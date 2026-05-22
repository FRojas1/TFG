"""Preprocess STM references for Whisper fine-tuning.

Applies two transformations to produce training targets that match
Whisper's native output style (cased, punctuated) while preserving
the speaker's actual words (including grammatical errors):

  1. Strip structural markers: (%hesitation%) -> "um", (partial-) -> removed
  2. Truecase + punctuate using a token-classification model (no word changes)

The result is cached as a JSON mapping {file_id: processed_reference} so
that fine-tuning can load it without reprocessing.

Usage:
    python src/preprocess_refs.py ^
        --flist info/sandi-corpus-2025/reference-materials/flists.flac/train-asr.tsv ^
        --data-root info/sandi-corpus-2025 ^
        --stm info/sandi-corpus-2025/reference-materials/stms/train-asr.stm ^
        --output info/sandi-corpus-2025/reference-materials/cache/train-whisper-refs.json
"""

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

from dataset import SANDiDataset

_HESITATION_RE = re.compile(r"\(%hesitation%\)", re.IGNORECASE)
_PARTIAL_RE = re.compile(r"\(\S+?-\)")
_SPACE_RE = re.compile(r"\s+")


def strip_markers(text: str) -> str:
    """Replace hesitation markers with 'um' and remove partial-word fragments."""
    text = _HESITATION_RE.sub(" um ", text)
    text = _PARTIAL_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="Preprocess STM refs for Whisper FT")
    parser.add_argument("--flist", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--stm", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for the punctuation model")
    args = parser.parse_args()

    print("Loading dataset...")
    ds = SANDiDataset(args.flist, args.data_root, stm_path=args.stm)

    items = [(utt.file_id, utt.reference) for utt in ds if utt.reference]
    print(f"  {len(items)} utterances with references")

    print("\nStep 1: Stripping structural markers...")
    stripped = {fid: strip_markers(ref) for fid, ref in items}

    print("Step 2: Loading truecasing + punctuation model...")
    from punctuators.models import PunctCapSegModelONNX
    pcs = PunctCapSegModelONNX.from_pretrained("pcs_en")

    print(f"Step 3: Processing {len(stripped)} references in batches of {args.batch_size}...")
    file_ids = list(stripped.keys())
    texts = [stripped[fid] for fid in file_ids]

    processed = {}
    for i in tqdm(range(0, len(texts), args.batch_size), desc="truecase+punct"):
        batch_ids = file_ids[i : i + args.batch_size]
        batch_texts = texts[i : i + args.batch_size]
        results = pcs.infer(batch_texts)
        for fid, segments in zip(batch_ids, results):
            processed[fid] = " ".join(segments)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(processed)} processed references to {out_path}")

    print("\n--- Sample comparisons ---")
    import random
    random.seed(42)
    sample_ids = random.sample(file_ids, min(8, len(file_ids)))
    for fid in sample_ids:
        raw = dict(items)[fid]
        print(f"\n  FILE: {fid}")
        print(f"  RAW:       {raw}")
        print(f"  STRIPPED:  {stripped[fid]}")
        print(f"  PROCESSED: {processed[fid]}")


if __name__ == "__main__":
    main()
