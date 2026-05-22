"""Transcribe with a fully fine-tuned Whisper model (no LoRA).

Uses the HuggingFace ASR pipeline (chunked long-form decoding) via the shared
helper in ``whisper_pipe.py``, matching the base Whisper baseline's decoding path.

Usage:
    python src/transcribe_ft.py ^
        --model-path checkpoints/whisper-small-full-gep/best ^
        --flist info/sandi-corpus-2025/reference-materials/flists.flac/eval-test-asr.tsv ^
        --data-root info/sandi-corpus-2025 ^
        --stm info/sandi-corpus-2025/reference-materials/stms/eval-test-asr.stm ^
        --output results/whisper-small-en-full-gep/eval-test.json
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
    p.add_argument("--model-path", required=True,
                   help="Path to the saved full model checkpoint directory")
    p.add_argument("--flist", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--stm", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    print(f"Loading {args.model_path} ...")
    _, pipe = load_inference_pipeline(
        full_model_path=args.model_path,
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

    for utt in tqdm(utts, desc="whisper-full-gep"):
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
        "model_id": args.model_path,
        "model_name": "whisper-small.en-full-gep",
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
