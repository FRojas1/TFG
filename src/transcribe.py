"""
Unified ASR transcription benchmark runner.

Runs a configured ASR model on every utterance in a Speak & Improve file list,
saves per-utterance results (hypothesis + reference) as JSON for downstream
evaluation.

Usage example (from project root):

    python src/transcribe.py ^
        --config configs/whisper_small.yaml ^
        --flist info/sandi-corpus-2025/reference-materials/flists.flac/dev-subset-asr.tsv ^
        --data-root info/sandi-corpus-2025 ^
        --stm info/sandi-corpus-2025/reference-materials/stms/dev-subset-asr.stm ^
        --output results/whisper-small-en/dev-subset.json
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import librosa
import soundfile as sf
import yaml
from tqdm import tqdm

from dataset import SANDiDataset
from models import load_model

TARGET_SAMPLE_RATE = 16_000


def load_audio(path: str, target_sr: int = TARGET_SAMPLE_RATE):
    audio, sr = sf.read(path, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr


def parse_args():
    p = argparse.ArgumentParser(description="Run ASR benchmark on S&I corpus")
    p.add_argument("--config", required=True, help="Model YAML config file")
    p.add_argument("--flist", required=True, help="TSV file list (file_id \\t audio_path)")
    p.add_argument("--data-root", required=True, help="Corpus root (audio paths resolve relative to this)")
    p.add_argument("--stm", default=None, help="STM file with reference transcripts")
    p.add_argument("--annotations", default=None, help="JSON annotation file (per-word marks)")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N utterances (for quick testing)")
    p.add_argument("--device", default=None, help="Override device from config (e.g. cpu, cuda, cuda:0)")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.device:
        config.setdefault("inference", {})["device"] = args.device

    print(f"Loading dataset from {args.flist}")
    dataset = SANDiDataset(
        flist_path=args.flist,
        data_root=args.data_root,
        stm_path=args.stm,
        annotations_path=args.annotations,
    )
    print(f"  {len(dataset)} utterances loaded")

    model_name = config["model"]["model_id"]
    print(f"Loading model: {model_name}")
    model = load_model(config)
    print(f"  Model ready: {model.name}")

    utterances = list(dataset)
    if args.limit:
        utterances = utterances[: args.limit]

    results = []
    skipped = 0
    total_audio_sec = 0.0
    total_inference_sec = 0.0

    print(f"Transcribing {len(utterances)} utterances ...")
    for utt in tqdm(utterances, desc=model.name):
        audio_path = Path(utt.audio_path)
        if not audio_path.exists():
            skipped += 1
            continue

        audio, sr = load_audio(str(audio_path))
        audio_duration = len(audio) / sr

        t0 = time.perf_counter()
        hypothesis = model.transcribe(audio, sr)
        inference_time = time.perf_counter() - t0

        total_audio_sec += audio_duration
        total_inference_sec += inference_time

        results.append({
            "file_id": utt.file_id,
            "hypothesis": hypothesis,
            "reference": utt.reference,
            "audio_duration_s": round(audio_duration, 3),
            "inference_time_s": round(inference_time, 3),
        })

    rtf = total_inference_sec / total_audio_sec if total_audio_sec > 0 else 0

    output_data = {
        "model_id": config["model"]["model_id"],
        "model_name": config["model"]["name"],
        "backend": config["model"]["backend"],
        "dataset": str(args.flist),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_utterances": len(results),
        "num_skipped": skipped,
        "total_audio_seconds": round(total_audio_sec, 1),
        "total_inference_seconds": round(total_inference_sec, 1),
        "real_time_factor": round(rtf, 4),
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nDone — {len(results)} transcribed, {skipped} skipped (audio not found)")
    print(f"Total audio: {total_audio_sec:.1f}s | Inference: {total_inference_sec:.1f}s | RTF: {rtf:.3f}")
    print(f"Results saved to {output_path}")

    if skipped > 0 and len(results) == 0:
        print(
            "\nAll utterances were skipped — the audio files are probably not downloaded yet.",
            file=sys.stderr,
        )
        print(
            "See info/sandi-corpus-2025/README-install-dataset.txt for download links.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
