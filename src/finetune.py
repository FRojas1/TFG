"""
Fine-tune ASR models on L2 verbatim transcripts.

Supports multiple modes:
  - lora     : LoRA fine-tuning (parameter-efficient)
  - gep      : LoRA + GEP error-only loss masking
  - full-gep : Full fine-tuning with per-token loss weighting (5x on errors)

Usage (full-gep):
    python src/finetune.py ^
        --config configs/whisper_small.yaml ^
        --mode full-gep ^
        --train-flist ... --val-flist ... --data-root ... ^
        --train-stm ... --val-stm ... ^
        --train-ref-cache refs/train-refs.json ^
        --val-ref-cache refs/eval-val-refs.json ^
        --train-gep-targets refs/train-gep-targets.json ^
        --output-dir checkpoints/whisper-small-full-gep ^
        --epochs 10 --lr 5e-6 --batch-size 2 --grad-accum 8
"""

import argparse
import json
import math
import re
from pathlib import Path

import librosa
import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import SANDiDataset

TARGET_SR = 16_000

# ------------------------------------------------------------------ #
# Reference normalization for training targets
# ------------------------------------------------------------------ #
# STM references contain structural markers that no ASR model produces
# natively.  We must clean these before using them as training targets,
# but the strategy differs by architecture:
#
#   Whisper (Seq2Seq) — produces cased, punctuated text.  Keep case and
#       punctuation.  Map (%hesitation%) -> "um" (a realistic filler the
#       model can learn).  Drop (partial-) fragments entirely (no
#       natural text equivalent).
#
#   CTC models — produce uppercase, unpunctuated text.  Lowercase,
#       strip punctuation.  Map (%hesitation%) -> "um".  Drop partials.
# ------------------------------------------------------------------ #

_HESITATION_RE = re.compile(r"\(%hesitation%\)", re.IGNORECASE)
_PARTIAL_RE = re.compile(r"\(\S+?-\)")
_PUNCT_RE = re.compile(r"[^\w\s']")
_SPACE_RE = re.compile(r"\s+")


def normalize_ref_whisper(text: str) -> str:
    """Normalize STM reference for Whisper training targets.

    Lowercases (STM refs are already lowercase) but keeps punctuation,
    which Whisper's decoder naturally produces.  Maps hesitation markers
    to 'um' and strips partial-word fragments.
    """
    text = _HESITATION_RE.sub(" um ", text)
    text = _PARTIAL_RE.sub(" ", text)
    text = text.lower()
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def normalize_ref_ctc(text: str) -> str:
    """Normalize STM reference for CTC training targets.

    Lowercases, strips punctuation, maps hesitation markers to 'um',
    and strips partial-word fragments.
    """
    text = _HESITATION_RE.sub(" um ", text)
    text = _PARTIAL_RE.sub(" ", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def load_audio(path, target_sr=TARGET_SR):
    audio, sr = sf.read(path, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


# ------------------------------------------------------------------ #
# Dataset wrappers
# ------------------------------------------------------------------ #

class WhisperFTDataset(Dataset):
    """Yields (input_features, labels) for Whisper fine-tuning.

    If ref_cache is provided (dict: file_id -> preprocessed text), uses
    the cached truecased+punctuated reference.  Otherwise falls back to
    normalize_ref_whisper() on the raw STM reference.
    """

    def __init__(self, sandi: SANDiDataset, processor, ref_cache=None,
                 max_audio_sec=30.0):
        self.items = []
        self.ref_cache = ref_cache or {}
        self.processor = processor
        self.max_samples = int(max_audio_sec * TARGET_SR)

        for utt in sandi:
            has_ref = (utt.file_id in self.ref_cache) or utt.reference
            if has_ref and Path(utt.audio_path).exists():
                self.items.append(utt)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        utt = self.items[idx]
        audio = load_audio(utt.audio_path)
        if len(audio) > self.max_samples:
            audio = audio[: self.max_samples]

        inputs = self.processor.feature_extractor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt"
        )
        input_features = inputs.input_features.squeeze(0)

        ref = self.ref_cache.get(utt.file_id) or normalize_ref_whisper(utt.reference)
        labels = self.processor.tokenizer(ref, return_tensors="pt").input_ids.squeeze(0)

        return {"input_features": input_features, "labels": labels}


class WhisperGEPDataset(Dataset):
    """Yields (input_features, labels) for targeted GEP training.

    Non-error token positions in labels are set to -100 so that
    cross-entropy loss only fires on grammatical-error tokens.
    EOS and all non-error tokens are masked, preventing the model
    from learning to change its stopping behavior or degrade
    general transcription.
    """

    def __init__(self, sandi: SANDiDataset, processor, gep_targets: dict,
                 ref_cache=None, max_audio_sec=30.0):
        self.items = []
        self.gep_targets = gep_targets
        self.ref_cache = ref_cache or {}
        self.processor = processor
        self.max_samples = int(max_audio_sec * TARGET_SR)

        for utt in sandi:
            if utt.file_id not in gep_targets:
                continue
            entry = gep_targets[utt.file_id]
            if not entry.get("error_word_indices"):
                continue
            has_ref = (utt.file_id in self.ref_cache) or utt.reference
            if has_ref and Path(utt.audio_path).exists():
                self.items.append(utt)

    def __len__(self):
        return len(self.items)

    def _map_words_to_tokens(self, text, tokenizer):
        """Map each word in text to its BPE token span.

        Returns list of (token_start, token_end) tuples, one per word.
        """
        full_ids = tokenizer(text, return_tensors="pt").input_ids.squeeze(0)

        words = text.split()
        word_token_spans = []
        token_pos = 0

        # Whisper tokenizer prepends special tokens. Find where text tokens start.
        # Decode token by token to find boundaries.
        all_tokens = full_ids.tolist()

        # Skip leading special tokens (SOT, language, task, etc.)
        # by finding where the actual text encoding begins
        special_prefix_len = 0
        for tid in all_tokens:
            decoded = tokenizer.decode([tid]).strip()
            if decoded == "" or tid in tokenizer.all_special_ids:
                special_prefix_len += 1
            else:
                break

        text_tokens = all_tokens[special_prefix_len:]

        # Greedily match words to consecutive token spans
        tok_idx = 0
        for word in words:
            word_lower = word.lower().strip()
            span_start = tok_idx
            accumulated = ""
            while tok_idx < len(text_tokens):
                piece = tokenizer.decode([text_tokens[tok_idx]]).strip().lower()
                accumulated = (accumulated + piece).strip()
                tok_idx += 1
                if accumulated == word_lower or accumulated.replace(" ", "") == word_lower.replace(" ", ""):
                    break
                if len(accumulated) >= len(word_lower):
                    break
            # Map back to full_ids indices (add special_prefix_len offset)
            word_token_spans.append(
                (span_start + special_prefix_len, tok_idx + special_prefix_len)
            )

        return word_token_spans, full_ids

    def __getitem__(self, idx):
        utt = self.items[idx]
        audio = load_audio(utt.audio_path)
        if len(audio) > self.max_samples:
            audio = audio[: self.max_samples]

        inputs = self.processor.feature_extractor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt"
        )
        input_features = inputs.input_features.squeeze(0)

        ref = self.ref_cache.get(utt.file_id) or normalize_ref_whisper(utt.reference)
        gep_entry = self.gep_targets[utt.file_id]
        error_word_indices = set(gep_entry["error_word_indices"])
        fluent_words = gep_entry["fluent_words"]

        # Align fluent words (from GEP annotations) to ref words (STM/cache).
        # The annotation transcript may cover a larger span than the STM
        # segment, so not all fluent words will have a match. We only need
        # the error words to align.
        ref_words = ref.split()
        ref_lower = [w.lower().strip(".,!?;:\"'") for w in ref_words]
        fluent_lower = [w.lower() for w in fluent_words]

        fluent_to_ref = {}
        ri = 0
        for fi, fw in enumerate(fluent_lower):
            if ri >= len(ref_lower):
                break
            # Allow skipping up to 3 ref words to handle minor insertions
            # from the punctuation model (e.g. "um" inserted)
            lookahead = min(ri + 4, len(ref_lower))
            for rj in range(ri, lookahead):
                if ref_lower[rj] == fw:
                    fluent_to_ref[fi] = rj
                    ri = rj + 1
                    break
            else:
                # No match in lookahead window; skip this fluent word
                pass

        # Find ref word indices that correspond to errors
        error_ref_indices = set()
        n_mapped = 0
        for fi in error_word_indices:
            if fi in fluent_to_ref:
                error_ref_indices.add(fluent_to_ref[fi])
                n_mapped += 1

        # If fewer than half the error words aligned, this sample is too
        # desynced to be useful -- return fully-masked labels.
        if n_mapped < len(error_word_indices) * 0.5:
            full_ids = self.processor.tokenizer(ref, return_tensors="pt").input_ids.squeeze(0)
            return {"input_features": input_features, "labels": torch.full_like(full_ids, -100)}

        # Tokenize and map words to BPE spans
        word_token_spans, full_ids = self._map_words_to_tokens(ref, self.processor.tokenizer)

        # Build masked labels: -100 everywhere, then unmask error token positions
        labels = torch.full_like(full_ids, -100)
        for ri in error_ref_indices:
            if ri < len(word_token_spans):
                start, end = word_token_spans[ri]
                labels[start:end] = full_ids[start:end]

        return {"input_features": input_features, "labels": labels}


class WhisperGEPWeightedDataset(Dataset):
    """Full-label dataset with per-token error weights for loss weighting.

    Instead of masking non-error tokens with -100, returns the full label
    sequence plus an ``error_mask`` tensor (1 at error token positions,
    0 elsewhere).  The training loop uses this to apply a higher weight
    (e.g. 5x) on error tokens while still learning from the whole sentence.
    """

    def __init__(self, sandi: SANDiDataset, processor, gep_targets: dict,
                 ref_cache=None, max_audio_sec=30.0):
        self.items = []
        self.gep_targets = gep_targets
        self.ref_cache = ref_cache or {}
        self.processor = processor
        self.max_samples = int(max_audio_sec * TARGET_SR)

        for utt in sandi:
            has_ref = (utt.file_id in self.ref_cache) or utt.reference
            if has_ref and Path(utt.audio_path).exists():
                self.items.append(utt)

    def __len__(self):
        return len(self.items)

    def _map_words_to_tokens(self, text, tokenizer):
        full_ids = tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
        words = text.split()
        word_token_spans = []
        all_tokens = full_ids.tolist()

        special_prefix_len = 0
        for tid in all_tokens:
            decoded = tokenizer.decode([tid]).strip()
            if decoded == "" or tid in tokenizer.all_special_ids:
                special_prefix_len += 1
            else:
                break

        text_tokens = all_tokens[special_prefix_len:]
        tok_idx = 0
        for word in words:
            word_lower = word.lower().strip()
            span_start = tok_idx
            accumulated = ""
            while tok_idx < len(text_tokens):
                piece = tokenizer.decode([text_tokens[tok_idx]]).strip().lower()
                accumulated = (accumulated + piece).strip()
                tok_idx += 1
                if accumulated == word_lower or accumulated.replace(" ", "") == word_lower.replace(" ", ""):
                    break
                if len(accumulated) >= len(word_lower):
                    break
            word_token_spans.append(
                (span_start + special_prefix_len, tok_idx + special_prefix_len)
            )
        return word_token_spans, full_ids

    def __getitem__(self, idx):
        utt = self.items[idx]
        audio = load_audio(utt.audio_path)
        if len(audio) > self.max_samples:
            audio = audio[: self.max_samples]

        inputs = self.processor.feature_extractor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt"
        )
        input_features = inputs.input_features.squeeze(0)

        ref = self.ref_cache.get(utt.file_id) or normalize_ref_whisper(utt.reference)
        labels = self.processor.tokenizer(ref, return_tensors="pt").input_ids.squeeze(0)

        # Default: no error tokens flagged (weight = 1.0 everywhere)
        error_mask = torch.zeros_like(labels, dtype=torch.float32)

        gep_entry = self.gep_targets.get(utt.file_id)
        if gep_entry and gep_entry.get("error_word_indices"):
            error_word_indices = set(gep_entry["error_word_indices"])
            fluent_words = gep_entry["fluent_words"]

            ref_words = ref.split()
            ref_lower = [w.lower().strip(".,!?;:\"'") for w in ref_words]
            fluent_lower = [w.lower() for w in fluent_words]

            fluent_to_ref = {}
            ri = 0
            for fi, fw in enumerate(fluent_lower):
                if ri >= len(ref_lower):
                    break
                lookahead = min(ri + 4, len(ref_lower))
                for rj in range(ri, lookahead):
                    if ref_lower[rj] == fw:
                        fluent_to_ref[fi] = rj
                        ri = rj + 1
                        break

            error_ref_indices = set()
            for fi in error_word_indices:
                if fi in fluent_to_ref:
                    error_ref_indices.add(fluent_to_ref[fi])

            if error_ref_indices:
                word_token_spans, full_ids = self._map_words_to_tokens(
                    ref, self.processor.tokenizer
                )
                labels = full_ids
                error_mask = torch.zeros_like(labels, dtype=torch.float32)
                for ri in error_ref_indices:
                    if ri < len(word_token_spans):
                        start, end = word_token_spans[ri]
                        error_mask[start:end] = 1.0

        return {
            "input_features": input_features,
            "labels": labels,
            "error_mask": error_mask,
        }


class CTCFTDataset(Dataset):
    """Yields (input_values, labels) for CTC fine-tuning."""

    def __init__(self, sandi: SANDiDataset, processor, max_audio_sec=30.0):
        self.items = []
        self.processor = processor
        self.max_samples = int(max_audio_sec * TARGET_SR)

        for utt in sandi:
            if utt.reference and Path(utt.audio_path).exists():
                self.items.append(utt)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        utt = self.items[idx]
        audio = load_audio(utt.audio_path)
        if len(audio) > self.max_samples:
            audio = audio[: self.max_samples]

        ref = normalize_ref_ctc(utt.reference)
        inputs = self.processor(audio=audio, sampling_rate=TARGET_SR, return_tensors="pt")
        labels = self.processor.tokenizer(ref, return_tensors="pt").input_ids.squeeze(0)

        input_tensor = (
            inputs.input_features.squeeze(0)
            if hasattr(inputs, "input_features")
            else inputs.input_values.squeeze(0)
        )

        return {"input_values": input_tensor, "labels": labels}


# ------------------------------------------------------------------ #
# Collators
# ------------------------------------------------------------------ #

def whisper_collate(batch):
    input_features = torch.stack([b["input_features"] for b in batch])
    label_lengths = [b["labels"].size(0) for b in batch]
    max_len = max(label_lengths)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, : b["labels"].size(0)] = b["labels"]
    return {"input_features": input_features, "labels": labels}


def whisper_weighted_collate(batch):
    """Collate for WhisperGEPWeightedDataset: pads labels AND error_mask."""
    input_features = torch.stack([b["input_features"] for b in batch])
    label_lengths = [b["labels"].size(0) for b in batch]
    max_len = max(label_lengths)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    error_mask = torch.zeros((len(batch), max_len), dtype=torch.float32)
    for i, b in enumerate(batch):
        n = b["labels"].size(0)
        labels[i, :n] = b["labels"]
        error_mask[i, :n] = b["error_mask"]
    return {"input_features": input_features, "labels": labels, "error_mask": error_mask}


def ctc_collate(batch):
    # input_values is (time, feat_dim) for spectral features or (time,) for waveform
    ndim = batch[0]["input_values"].ndim
    if ndim == 2:
        # Spectral features: (time, feat_dim) -- pad along time axis
        input_lengths = [b["input_values"].shape[0] for b in batch]
        feat_dim = batch[0]["input_values"].shape[1]
        max_time = max(input_lengths)
        padded = torch.zeros(len(batch), max_time, feat_dim)
        for i, b in enumerate(batch):
            t = b["input_values"].shape[0]
            padded[i, :t, :] = b["input_values"]
    else:
        # Raw waveform: (time,)
        input_lengths = [b["input_values"].shape[0] for b in batch]
        max_time = max(input_lengths)
        padded = torch.zeros(len(batch), max_time)
        for i, b in enumerate(batch):
            padded[i, : b["input_values"].size(0)] = b["input_values"]

    label_lengths = [b["labels"].size(0) for b in batch]
    max_label = max(label_lengths)
    labels = torch.full((len(batch), max_label), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, : b["labels"].size(0)] = b["labels"]

    return {
        "input_features": padded,
        "labels": labels,
        "attention_mask": torch.tensor(
            [[1] * l + [0] * (max_time - l) for l in input_lengths],
            dtype=torch.long,
        ),
    }


# ------------------------------------------------------------------ #
# LR schedule with linear warmup + cosine decay
# ------------------------------------------------------------------ #

class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        super().__init__(optimizer, lr_lambda)


# ------------------------------------------------------------------ #
# Validation
# ------------------------------------------------------------------ #

_FILLER_WORDS = frozenset({
    "uh", "uhh", "uhhh", "uhhhh", "uhhhhh", "uhm",
    "um", "umm", "ummm", "ummmmm",
    "ah", "ahh", "ahm",
    "eh", "ehh", "ehm",
    "er", "erm", "err",
    "hm", "hmm", "hmmm",
    "mm", "mmm", "mmmm", "mmmmm", "mmmmmm",
    "mhm", "mmhm", "mmhmm",
    "huh", "uhhuh",
})


def _normalize_for_wer(text: str, is_hyp: bool = False) -> str:
    """Normalize text for lexical WER: lowercase, strip punct, remove fillers from hyp."""
    text = _HESITATION_RE.sub(" ", text)
    text = _PARTIAL_RE.sub(" ", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    if is_hyp:
        words = [w for w in text.split() if w not in _FILLER_WORDS]
        text = " ".join(words)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


@torch.no_grad()
def validate_whisper(model, processor, val_dataset, device, max_val=200):
    """Quick WER validation on a subset using lexical normalization.

    Transcription goes through the HuggingFace ASR pipeline (chunked long-form
    decoding) to match the inference path used by ``transcribe_lora.py`` /
    ``transcribe_ft.py`` -- raw ``model.generate(max_new_tokens=225)`` silently
    truncates utterances longer than ~20s and gives phantom validation WER.
    """
    import jiwer
    from whisper_pipe import wrap_model_pipeline, transcribe_audio
    model.eval()
    pipe = wrap_model_pipeline(model, processor, device=device, dtype=torch.float32)
    refs, hyps = [], []
    items = val_dataset.items[:max_val]
    for utt in tqdm(items, desc="val", leave=False):
        hyp, _, _ = transcribe_audio(pipe, utt.audio_path)
        ref = _normalize_for_wer(utt.reference, is_hyp=False)
        hyp = _normalize_for_wer(hyp, is_hyp=True)
        if ref:
            refs.append(ref)
            hyps.append(hyp)
    model.train()
    if not refs:
        return 1.0
    return jiwer.wer(refs, hyps)


@torch.no_grad()
def validate_ctc(model, processor, val_dataset, device, max_val=200):
    import jiwer
    model.eval()
    refs, hyps = [], []
    items = val_dataset.items[:max_val]
    for utt in tqdm(items, desc="val", leave=False):
        audio = load_audio(utt.audio_path)
        inputs = processor(audio=audio, sampling_rate=TARGET_SR, return_tensors="pt")
        feats = inputs.input_features.to(device, dtype=torch.float32)
        outputs = model.generate(feats)
        hyp = processor.batch_decode(outputs)[0].strip()
        ref = normalize_ref_ctc(utt.reference)
        if ref:
            refs.append(ref)
            hyps.append(hyp.lower())
    model.train()
    if not refs:
        return 1.0
    return jiwer.wer(refs, hyps)


# ------------------------------------------------------------------ #
# Fine-tuning loops
# ------------------------------------------------------------------ #

def finetune_whisper(config, args):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from peft import LoraConfig, get_peft_model

    device = args.device
    model_id = config["model"]["model_id"]
    print(f"Loading Whisper model: {model_id}")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(device)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = WhisperFTDataset(args._train_sandi, processor,
                                ref_cache=args._train_ref_cache)
    val_ds = WhisperFTDataset(args._val_sandi, processor,
                              ref_cache=args._val_ref_cache)
    print(f"Train: {len(train_ds)} utterances, Val: {len(val_ds)} utterances")

    # Sanity check: validate before any training to confirm baseline
    print("Running pre-training validation (baseline) ...")
    baseline_wer = validate_whisper(model, processor, val_ds, device, max_val=args.max_val)
    print(f"  Baseline val WER: {baseline_wer:.4f}")

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=whisper_collate,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_wer = baseline_wer
    log = [{"epoch": 0, "loss": None, "val_wer": baseline_wer}]

    model.train()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            epoch_loss += outputs.loss.item()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = epoch_loss / len(loader)
        val_wer = validate_whisper(model, processor, val_ds, device, max_val=args.max_val)
        print(f"  Epoch {epoch}: loss={avg_loss:.4f}, val_wer={val_wer:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
        log.append({"epoch": epoch, "loss": avg_loss, "val_wer": val_wer})

        if val_wer < best_wer:
            best_wer = val_wer
            model.save_pretrained(output_dir / "best")
            processor.save_pretrained(output_dir / "best")
            print(f"  -> New best WER: {val_wer:.4f}")

    model.save_pretrained(output_dir / "last")
    processor.save_pretrained(output_dir / "last")

    with open(output_dir / "train_log.json", "w") as f:
        json.dump({"config": config, "args": vars(args), "log": log, "best_wer": best_wer}, f, indent=2, default=str)

    print(f"\nDone. Best val WER: {best_wer:.4f} (baseline was {baseline_wer:.4f})")
    return output_dir / "best"


def finetune_whisper_gep(config, args):
    """LoRA fine-tuning with GEP-masked loss: only error tokens receive gradient."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from peft import LoraConfig, get_peft_model

    device = args.device
    model_id = config["model"]["model_id"]
    print(f"Loading Whisper model (GEP mode): {model_id}")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(device)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = WhisperGEPDataset(
        args._train_sandi, processor,
        gep_targets=args._train_gep_targets,
        ref_cache=args._train_ref_cache,
    )
    val_ds = WhisperFTDataset(
        args._val_sandi, processor,
        ref_cache=args._val_ref_cache,
    )
    print(f"Train (GEP): {len(train_ds)} utterances, Val: {len(val_ds)} utterances")

    print("Running pre-training validation (baseline) ...")
    baseline_wer = validate_whisper(model, processor, val_ds, device, max_val=args.max_val)
    print(f"  Baseline val WER: {baseline_wer:.4f}")

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=whisper_collate,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_wer = baseline_wer
    log = [{"epoch": 0, "loss": None, "val_wer": baseline_wer}]

    model.train()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        n_batches_with_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            # Skip batches where all labels are masked (no error tokens)
            if (labels == -100).all():
                continue

            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            epoch_loss += outputs.loss.item()
            n_batches_with_loss += 1

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = epoch_loss / max(1, n_batches_with_loss)
        val_wer = validate_whisper(model, processor, val_ds, device, max_val=args.max_val)
        print(f"  Epoch {epoch}: loss={avg_loss:.4f}, val_wer={val_wer:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
        log.append({"epoch": epoch, "loss": avg_loss, "val_wer": val_wer})

        if val_wer < best_wer:
            best_wer = val_wer
            model.save_pretrained(output_dir / "best")
            processor.save_pretrained(output_dir / "best")
            print(f"  -> New best WER: {val_wer:.4f}")

    model.save_pretrained(output_dir / "last")
    processor.save_pretrained(output_dir / "last")

    with open(output_dir / "train_log.json", "w") as f:
        json.dump({"config": config, "args": vars(args), "log": log, "best_wer": best_wer}, f, indent=2, default=str)

    print(f"\nDone. Best val WER: {best_wer:.4f} (baseline was {baseline_wer:.4f})")
    return output_dir / "best"


def finetune_whisper_full_gep(config, args):
    """Full fine-tuning with per-token loss weighting (no LoRA).

    Uses gradient checkpointing + FP16 mixed precision to fit in 12 GB.
    The encoder is frozen by default (--freeze-encoder) since it already
    extracts audio features well -- only the decoder (which carries the
    language model bias that "corrects" grammar) is fine-tuned.
    Error tokens receive ``args.error_weight`` times the normal loss.
    """
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device = args.device
    model_id = config["model"]["model_id"]
    print(f"Loading Whisper model (full-gep mode): {model_id}")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(device)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if args.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
        print("Encoder frozen -- only decoder parameters will be updated")

    model.gradient_checkpointing_enable()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} parameters")

    train_ds = WhisperGEPWeightedDataset(
        args._train_sandi, processor,
        gep_targets=args._train_gep_targets,
        ref_cache=args._train_ref_cache,
    )
    val_ds = WhisperFTDataset(
        args._val_sandi, processor,
        ref_cache=args._val_ref_cache,
    )
    print(f"Train: {len(train_ds)} utterances, Val: {len(val_ds)} utterances")

    print("Running pre-training validation (baseline) ...")
    baseline_wer = validate_whisper(model, processor, val_ds, device, max_val=args.max_val)
    print(f"  Baseline val WER: {baseline_wer:.4f}")

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=whisper_weighted_collate,
        num_workers=0,
        pin_memory=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    scaler = torch.amp.GradScaler("cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_wer = baseline_wer
    log = [{"epoch": 0, "loss": None, "val_wer": baseline_wer}]
    error_w = args.error_weight
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    model.train()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            error_mask = batch["error_mask"].to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(input_features=input_features, labels=labels)
                logits = outputs.logits

                # Shift for autoregressive loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_error = error_mask[..., 1:].contiguous()

                raw_loss = loss_fct(
                    shift_logits.view(-1, logits.size(-1)),
                    shift_labels.view(-1),
                )
                raw_loss = raw_loss.view(shift_labels.size())

                # Build per-token weights: 1.0 for normal, error_w for errors
                weights = torch.ones_like(shift_labels, dtype=torch.float32)
                weights[shift_error > 0.5] = error_w

                # Zero out padding (-100 positions)
                padding_mask = (shift_labels != -100).float()
                weights = weights * padding_mask

                weighted_loss = (raw_loss * weights).sum() / padding_mask.sum().clamp(min=1)

            loss = weighted_loss / args.grad_accum
            scaler.scale(loss).backward()
            epoch_loss += weighted_loss.item()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = epoch_loss / len(loader)

        # Disable gradient checkpointing for generation during validation
        model.gradient_checkpointing_disable()
        val_wer = validate_whisper(model, processor, val_ds, device, max_val=args.max_val)
        model.gradient_checkpointing_enable()

        print(f"  Epoch {epoch}: loss={avg_loss:.4f}, val_wer={val_wer:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
        log.append({"epoch": epoch, "loss": avg_loss, "val_wer": val_wer})

        if val_wer < best_wer:
            best_wer = val_wer
            save_dir = output_dir / "best"
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            print(f"  -> New best WER: {val_wer:.4f}")

    save_dir = output_dir / "last"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    with open(output_dir / "train_log.json", "w") as f:
        json.dump({"config": config, "args": vars(args), "log": log, "best_wer": best_wer}, f, indent=2, default=str)

    print(f"\nDone. Best val WER: {best_wer:.4f} (baseline was {baseline_wer:.4f})")
    return output_dir / "best"


def finetune_ctc(config, args):
    from transformers import AutoModelForCTC, AutoProcessor
    from peft import LoraConfig, get_peft_model

    device = args.device
    model_id = config["model"]["model_id"]
    print(f"Loading CTC model: {model_id}")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(device)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = CTCFTDataset(args._train_sandi, processor)
    val_ds = CTCFTDataset(args._val_sandi, processor)
    print(f"Train: {len(train_ds)} utterances, Val: {len(val_ds)} utterances")

    print("Running pre-training validation (baseline) ...")
    baseline_wer = validate_ctc(model, processor, val_ds, device, max_val=args.max_val)
    print(f"  Baseline val WER: {baseline_wer:.4f}")

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ctc_collate,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_wer = baseline_wer
    log = [{"epoch": 0, "loss": None, "val_wer": baseline_wer}]

    model.train()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")):
            feats = batch["input_features"].to(device, dtype=torch.float32)
            labels = batch["labels"].to(device)
            attn = batch["attention_mask"].to(device)

            outputs = model(input_features=feats, labels=labels, attention_mask=attn)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            epoch_loss += outputs.loss.item()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = epoch_loss / len(loader)
        val_wer = validate_ctc(model, processor, val_ds, device, max_val=args.max_val)
        print(f"  Epoch {epoch}: loss={avg_loss:.4f}, val_wer={val_wer:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
        log.append({"epoch": epoch, "loss": avg_loss, "val_wer": val_wer})

        if val_wer < best_wer:
            best_wer = val_wer
            model.save_pretrained(output_dir / "best")
            processor.save_pretrained(output_dir / "best")
            print(f"  -> New best WER: {val_wer:.4f}")

    model.save_pretrained(output_dir / "last")
    processor.save_pretrained(output_dir / "last")

    with open(output_dir / "train_log.json", "w") as f:
        json.dump({"config": config, "args": vars(args), "log": log, "best_wer": best_wer}, f, indent=2, default=str)

    print(f"\nDone. Best val WER: {best_wer:.4f} (baseline was {baseline_wer:.4f})")
    return output_dir / "best"


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune ASR model with LoRA")
    p.add_argument("--config", required=True)
    p.add_argument("--train-flist", required=True)
    p.add_argument("--val-flist", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--train-stm", required=True)
    p.add_argument("--val-stm", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode", choices=["lora", "gep", "full-gep"], default="lora",
                   help="Training mode: lora | gep (LoRA + masking) | full-gep (full FT + loss weighting)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-val", type=int, default=200, help="Max utterances for validation WER")
    p.add_argument("--error-weight", type=float, default=5.0,
                   help="Loss multiplier for error tokens (full-gep mode, default 5.0)")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="Freeze encoder weights, only fine-tune the decoder")
    p.add_argument("--train-ref-cache", default=None,
                   help="JSON cache of preprocessed train refs (from preprocess_refs.py)")
    p.add_argument("--val-ref-cache", default=None,
                   help="JSON cache of preprocessed val refs (from preprocess_refs.py)")
    p.add_argument("--train-gep-targets", default=None,
                   help="JSON cache of GEP error indices (from build_gep_targets.py)")
    p.add_argument("--val-gep-targets", default=None,
                   help="JSON cache of GEP error indices for validation")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    backend = config["model"]["backend"]

    print("Loading train dataset...")
    args._train_sandi = SANDiDataset(
        args.train_flist, args.data_root, stm_path=args.train_stm
    )
    print(f"  {len(args._train_sandi)} utterances")

    print("Loading val dataset...")
    args._val_sandi = SANDiDataset(
        args.val_flist, args.data_root, stm_path=args.val_stm
    )
    print(f"  {len(args._val_sandi)} utterances")

    args._train_ref_cache = None
    args._val_ref_cache = None
    if args.train_ref_cache:
        with open(args.train_ref_cache, encoding="utf-8") as f:
            args._train_ref_cache = json.load(f)
        print(f"Loaded train ref cache: {len(args._train_ref_cache)} entries")
    if args.val_ref_cache:
        with open(args.val_ref_cache, encoding="utf-8") as f:
            args._val_ref_cache = json.load(f)
        print(f"Loaded val ref cache: {len(args._val_ref_cache)} entries")

    args._train_gep_targets = None
    args._val_gep_targets = None
    if args.train_gep_targets:
        with open(args.train_gep_targets, encoding="utf-8") as f:
            args._train_gep_targets = json.load(f)
        print(f"Loaded train GEP targets: {len(args._train_gep_targets)} entries")
    if args.val_gep_targets:
        with open(args.val_gep_targets, encoding="utf-8") as f:
            args._val_gep_targets = json.load(f)
        print(f"Loaded val GEP targets: {len(args._val_gep_targets)} entries")

    if backend == "whisper" and args.mode == "full-gep":
        if not args._train_gep_targets:
            raise ValueError("full-gep mode requires --train-gep-targets")
        finetune_whisper_full_gep(config, args)
    elif backend == "whisper" and args.mode == "gep":
        if not args._train_gep_targets:
            raise ValueError("GEP mode requires --train-gep-targets")
        finetune_whisper_gep(config, args)
    elif backend == "whisper":
        finetune_whisper(config, args)
    elif backend == "fastconformer_ctc":
        finetune_ctc(config, args)
    else:
        raise ValueError(f"Fine-tuning not supported for backend: {backend}")


if __name__ == "__main__":
    main()
