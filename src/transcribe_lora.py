"""Transcribe with a LoRA-finetuned Whisper model.

Uses the HuggingFace ASR pipeline (chunked long-form decoding, ``return_timestamps=True``)
via the shared helper in ``whisper_pipe.py``. This matches the decoding path used for
the base Whisper baseline -- raw ``model.generate(max_new_tokens=225)`` would silently
truncate any utterance longer than ~20s and was responsible for ~12-15pp of phantom
WER in the original LoRA/DPO runs.

Usage:
    python src/transcribe_lora.py ^
        --base-model openai/whisper-small.en ^
        --lora-path checkpoints/whisper-small-lora/best ^
        --flist info/sandi-corpus-2025/reference-materials/flists.flac/eval-test-asr.tsv ^
        --data-root info/sandi-corpus-2025 ^
        --stm info/sandi-corpus-2025/reference-materials/stms/eval-test-asr.stm ^
        --output results/whisper-small-en-lora/eval-test.json
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from dataset import SANDiDataset
from whisper_pipe import load_inference_pipeline, transcribe_audio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--lora-path", required=True)
    p.add_argument("--flist", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--stm", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N utterances (smoke check)")
    args = p.parse_args()

    print(f"Loading {args.base_model} + LoRA from {args.lora_path} ...")
    _, pipe = load_inference_pipeline(
        base_model=args.base_model,
        lora_path=args.lora_path,
        device=args.device,
        dtype=torch.float16,
    )
    print("Pipeline ready")

    ds = SANDiDataset(args.flist, args.data_root, stm_path=args.stm)
    utts = list(ds)
    if args.limit:
        utts = utts[: args.limit]
    print(f"{len(utts)} utterances")

    results = []
    skipped = 0
    total_audio = 0.0
    total_infer = 0.0

    for utt in tqdm(utts, desc="whisper-small-lora"):
        ap = Path(utt.audio_path)
        if not ap.exists():
            skipped += 1
            continue
        hyp, dur, elapsed = transcribe_audio(pipe, ap)
        total_audio += dur
        total_infer += elapsed
        results.append({
            "file_id": utt.file_id,
            "hypothesis": hyp,
            "reference": utt.reference,
            "audio_duration_s": round(dur, 3),
            "inference_time_s": round(elapsed, 3),
        })

    rtf = total_infer / total_audio if total_audio > 0 else 0
    out = {
        "model_id": f"{args.base_model}+lora({args.lora_path})",
        "model_name": Path(args.lora_path).parent.name or "whisper-small.en-lora",
        "backend": "whisper-pipeline",
        "dataset": args.flist,
        "num_utterances": len(results),
        "num_skipped": skipped,
        "total_audio_seconds": round(total_audio, 1),
        "total_inference_seconds": round(total_infer, 1),
        "real_time_factor": round(rtf, 4),
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"Done: {len(results)} transcribed, {skipped} skipped, RTF={rtf:.3f}")


if __name__ == "__main__":
    main()
