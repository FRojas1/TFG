"""
DPO fine-tuning for Grammar Error Preservation.

Trains a Whisper model with Direct Preference Optimization to prefer
error-preserving transcriptions over error-correcting ones. Uses LoRA
on all linear layers for parameter efficiency.

Usage:
    python src/finetune_dpo.py ^
        --config configs/whisper_small.yaml ^
        --train-flist info/sandi-corpus-2025/reference-materials/dpo-pool/finetune-train.tsv ^
        --val-flist info/sandi-corpus-2025/reference-materials/dpo-pool/finetune-val.tsv ^
        --data-root info/sandi-corpus-2025 ^
        --train-dpo-pairs info/sandi-corpus-2025/reference-materials/dpo-pool/finetune-train-dpo.json ^
        --val-dpo-pairs info/sandi-corpus-2025/reference-materials/dpo-pool/finetune-val-dpo.json ^
        --val-stm info/sandi-corpus-2025/reference-materials/stms/train-asr.stm ^
        --val-annotations info/sandi-corpus-2025/reference-materials/annotations/train-trans-ref.json ^
        --val-gec-annotations info/sandi-corpus-2025/reference-materials/annotations/train-gec-ref.json ^
        --output-dir checkpoints/whisper-small-dpo ^
        --epochs 5 --lr 5e-5 --batch-size 2 --grad-accum 8
"""

import argparse
import functools
import json
import math
import re
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

TARGET_SR = 16_000


def load_audio(path, target_sr=TARGET_SR):
    audio, sr = sf.read(path, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    assert sr == target_sr, f"Expected {target_sr}Hz, got {sr}Hz for {path}"
    return audio


# ------------------------------------------------------------------ #
# Text normalization (matches evaluate.py)
# ------------------------------------------------------------------ #

_PUNCT_RE = re.compile(r"[^\w\s']")

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


def normalize_hyp(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    words = [w for w in text.split() if w not in _FILLER_WORDS]
    return " ".join(words)


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class DPODataset(Dataset):
    """Yields (audio, chosen_text, rejected_text) for DPO training.

    The DPO pairs JSON provides pre-normalized chosen/rejected texts.
    """

    def __init__(self, flist_path, data_root, dpo_pairs_path, max_audio_sec=30.0):
        self.data_root = Path(data_root)
        self.max_samples = int(max_audio_sec * TARGET_SR)

        with open(dpo_pairs_path, encoding="utf-8") as f:
            dpo_pairs = json.load(f)

        self.items = []
        with open(flist_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                file_id, rel_path = line.split("\t", 1)
                if file_id not in dpo_pairs:
                    continue
                audio_path = str(self.data_root / rel_path)
                if not Path(audio_path).exists():
                    continue
                self.items.append({
                    "file_id": file_id,
                    "audio_path": audio_path,
                    "chosen": dpo_pairs[file_id]["chosen"],
                    "rejected": dpo_pairs[file_id]["rejected"],
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio = load_audio(item["audio_path"])
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        return {
            "audio": audio,
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "file_id": item["file_id"],
        }


def dpo_collate(batch, *, processor):
    """Collate DPO batch: process audio features and tokenize chosen/rejected."""
    chosens = [b["chosen"] for b in batch]
    rejecteds = [b["rejected"] for b in batch]

    # Process each audio individually (Whisper feature extractor auto-pads to 3000 frames)
    features = []
    for b in batch:
        inputs = processor.feature_extractor(
            b["audio"], sampling_rate=TARGET_SR, return_tensors="pt"
        )
        features.append(inputs.input_features.squeeze(0))
    input_features = torch.stack(features)

    chosen_enc = processor.tokenizer(
        chosens, return_tensors="pt", padding=True, truncation=True, max_length=448
    )
    rejected_enc = processor.tokenizer(
        rejecteds, return_tensors="pt", padding=True, truncation=True, max_length=448
    )

    return {
        "input_features": input_features,
        "chosen_ids": chosen_enc.input_ids,
        "chosen_mask": chosen_enc.attention_mask,
        "rejected_ids": rejected_enc.input_ids,
        "rejected_mask": rejected_enc.attention_mask,
    }


# ------------------------------------------------------------------ #
# DPO Loss
# ------------------------------------------------------------------ #

def encode_audio(model, input_features):
    """Run the encoder and return cached encoder outputs.

    For PEFT-wrapped models (PeftModel -> WhisperForConditionalGeneration ->
    WhisperModel -> WhisperEncoder), the encoder lives at model.model.model.encoder.
    Returns a BaseModelOutput that can be passed to forward() as encoder_outputs.
    """
    encoder = model.model.model.encoder
    return encoder(input_features)


def compute_seq_logprobs(model, label_ids, label_mask, encoder_outputs):
    """Compute per-sequence sum of log-probabilities via teacher forcing.

    Whisper's forward with `labels` internally shifts to create decoder_input_ids,
    so logits[i] directly predicts labels[i] — no additional shifting needed.

    Args:
        model: Whisper model (encoder-decoder), possibly PEFT-wrapped
        label_ids: (batch, seq_len) token IDs (pad positions = pad_token_id)
        label_mask: (batch, seq_len) attention mask (1=real token, 0=pad)
        encoder_outputs: pre-computed encoder hidden states (BaseModelOutput)

    Returns:
        (batch,) tensor of summed log-probs per sequence
    """
    labels_for_model = label_ids.clone()
    labels_for_model[label_mask == 0] = -100

    outputs = model(
        encoder_outputs=encoder_outputs, 
        labels=labels_for_model,
        decoder_attention_mask=label_mask
    )
    logits = outputs.logits

    log_probs = F.log_softmax(logits, dim=-1)

    gather_ids = label_ids.clone()
    gather_ids[label_mask == 0] = 0

    per_token_logps = torch.gather(
        log_probs, dim=-1, index=gather_ids.unsqueeze(-1)
    ).squeeze(-1)

    per_token_logps = per_token_logps * label_mask.float()
    seq_logps = per_token_logps.sum(dim=-1)

    return seq_logps


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """Compute DPO loss.

    loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
    """
    chosen_rewards = policy_chosen_logps - ref_chosen_logps
    rejected_rewards = policy_rejected_logps - ref_rejected_logps
    logits = beta * (chosen_rewards - rejected_rewards)
    loss = -F.logsigmoid(logits).mean()

    # Metrics for logging
    with torch.no_grad():
        chosen_reward_mean = chosen_rewards.mean().item()
        rejected_reward_mean = rejected_rewards.mean().item()
        reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

    return loss, {
        "chosen_reward": chosen_reward_mean,
        "rejected_reward": rejected_reward_mean,
        "reward_margin": reward_margin,
        "accuracy": accuracy,
    }


# ------------------------------------------------------------------ #
# LR schedule
# ------------------------------------------------------------------ #

class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        super().__init__(optimizer, lr_lambda)


# ------------------------------------------------------------------ #
# GEP Validation
# ------------------------------------------------------------------ #

@torch.no_grad()
def validate_gep(model, processor, val_dataset, device,
                 dataset_obj, gec_path, max_val=200):
    """Validate by computing full GEP preservation rate and WER on a subset.

    Transcription goes through the HuggingFace pipeline (chunked long-form
    decoding with ``return_timestamps=True``) to match the inference path
    used by ``transcribe_lora.py`` -- raw ``model.generate(max_new_tokens=225)``
    silently truncates utterances longer than ~20s and produces phantom WER.
    """
    import jiwer
    from evaluate import compute_gep
    from whisper_pipe import wrap_model_pipeline, transcribe_audio

    model.enable_adapter_layers()
    model.eval()

    pipe = wrap_model_pipeline(model, processor, device=device)

    items = val_dataset.items[:max_val]
    results = []
    refs, hyps = [], []

    for item in tqdm(items, desc="val-gep", leave=False):
        hyp, _, _ = transcribe_audio(pipe, item["audio_path"])

        results.append({
            "file_id": item["file_id"],
            "hypothesis": hyp,
        })

        refs.append(normalize_hyp(item["chosen"]))
        hyps.append(normalize_hyp(hyp))

    gep_metrics = compute_gep(results, dataset_obj, gec_path)
    overall = gep_metrics["summary"].get("_overall", {})
    preservation_rate = overall.get("preservation_rate", 0.0)

    wer = jiwer.wer(refs, hyps) if refs else 0.0

    model.train()
    return {"gep": preservation_rate, "wer": wer}


# ------------------------------------------------------------------ #
# Training loop
# ------------------------------------------------------------------ #

def train(config, args):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from peft import LoraConfig, get_peft_model
    from dataset import SANDiDataset

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
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Datasets
    train_ds = DPODataset(args.train_flist, args.data_root, args.train_dpo_pairs)
    val_ds = DPODataset(args.val_flist, args.data_root, args.val_dpo_pairs)
    print(f"Train: {len(train_ds)} pairs, Val: {len(val_ds)} pairs")

    # SANDiDataset for full GEP validation
    val_sandi = SANDiDataset(
        flist_path=args.val_flist,
        data_root=args.data_root,
        stm_path=args.val_stm,
        annotations_path=args.val_annotations,
    )

    if len(train_ds) == 0:
        raise ValueError("No training data found! Check paths and DPO pairs.")

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=functools.partial(dpo_collate, processor=processor),
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mixed precision setup
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.dtype]
    use_amp = amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda") if args.dtype == "fp16" else None
    print(f"Mixed precision: {args.dtype}" + (" (with GradScaler)" if scaler else ""))

    # Baseline validation
    print("Running pre-training validation...")
    baseline_metrics = validate_gep(
        model, processor, val_ds, device,
        val_sandi, args.val_gec_annotations, max_val=args.max_val,
    )
    print(f"  Baseline GEP: {baseline_metrics['gep']:.4f}, WER: {baseline_metrics['wer']:.4f}")

    best_gep = baseline_metrics["gep"]
    log = [{"epoch": 0, "loss": None, **baseline_metrics}]

    model.train()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_metrics = {"chosen_reward": 0, "rejected_reward": 0,
                         "reward_margin": 0, "accuracy": 0}
        n_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")):
            input_features = batch["input_features"].to(device)
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)

            # Policy forward (LoRA enabled): encode once, decode twice
            model.enable_adapter_layers()
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                policy_enc = encode_audio(model, input_features)
                policy_chosen_logps = compute_seq_logprobs(
                    model, chosen_ids, chosen_mask, policy_enc
                )
                policy_rejected_logps = compute_seq_logprobs(
                    model, rejected_ids, rejected_mask, policy_enc
                )

            # Reference forward (LoRA disabled, no grad): encode once, decode twice
            model.disable_adapter_layers()
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    ref_enc = encode_audio(model, input_features)
                    ref_chosen_logps = compute_seq_logprobs(
                        model, chosen_ids, chosen_mask, ref_enc
                    )
                    ref_rejected_logps = compute_seq_logprobs(
                        model, rejected_ids, rejected_mask, ref_enc
                    )

            # Re-enable for backward pass
            model.enable_adapter_layers()

            loss, metrics = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=args.beta,
            )

            scaled_loss = loss / args.grad_accum
            if scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            epoch_loss += loss.item()
            n_batches += 1

            for k, v in metrics.items():
                epoch_metrics[k] += v

            if (step + 1) % args.grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = epoch_loss / max(1, n_batches)
        avg_metrics = {k: v / max(1, n_batches) for k, v in epoch_metrics.items()}

        val_metrics = validate_gep(
            model, processor, val_ds, device,
            val_sandi, args.val_gec_annotations, max_val=args.max_val,
        )
        val_gep = val_metrics["gep"]

        print(f"  Epoch {epoch}: loss={avg_loss:.4f}, val_gep={val_gep:.4f}, "
              f"wer={val_metrics['wer']:.4f}, "
              f"accuracy={avg_metrics['accuracy']:.3f}, "
              f"margin={avg_metrics['reward_margin']:.3f}, "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        log.append({
            "epoch": epoch, "loss": avg_loss, **val_metrics,
            **avg_metrics,
        })

        if val_gep > best_gep:
            best_gep = val_gep
            model.save_pretrained(output_dir / "best")
            processor.save_pretrained(output_dir / "best")
            print(f"  -> New best GEP: {val_gep:.4f}")

    model.save_pretrained(output_dir / "last")
    processor.save_pretrained(output_dir / "last")

    with open(output_dir / "train_log.json", "w") as f:
        json.dump({"config": config, "args": vars(args), "log": log,
                   "best_gep": best_gep}, f, indent=2, default=str)

    print(f"\nDone. Best val GEP: {best_gep:.4f} (baseline was {baseline_metrics['gep']:.4f})")
    return output_dir / "best"


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="DPO fine-tuning for GEP")
    p.add_argument("--config", required=True, help="Model YAML config")
    p.add_argument("--train-flist", required=True, help="Training file list (from build_finetune_pool.py)")
    p.add_argument("--val-flist", required=True, help="Validation file list")
    p.add_argument("--data-root", required=True, help="Corpus root directory")
    p.add_argument("--train-dpo-pairs", required=True, help="Training DPO pairs JSON")
    p.add_argument("--val-dpo-pairs", required=True, help="Validation DPO pairs JSON")
    p.add_argument("--output-dir", required=True, help="Checkpoint output directory")
    p.add_argument("--val-stm", required=True, help="STM for full GEP validation")
    p.add_argument("--val-annotations", required=True, help="Annotations JSON for full GEP validation")
    p.add_argument("--val-gec-annotations", required=True, help="GEC annotations for full GEP validation")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta (preference strength)")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                   help="Mixed precision dtype (bf16 recommended, fp16 requires GradScaler)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker processes (0=main process only)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-val", type=int, default=100, help="Max utterances for validation")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config["model"]["backend"] != "whisper":
        raise ValueError("DPO fine-tuning currently only supports Whisper models")

    train(config, args)


if __name__ == "__main__":
    main()
