"""Re-run evaluate.py over every transcription JSON in results/.

Used after a change to the evaluation methodology (e.g. updating
GEP_EXCLUDED_TYPES in evaluate.py) so all *-eval.json files reflect
the new metric definitions.

The script knows about the four splits used in this project:
  - dev          (full 3,249-utt development set)
  - dev-subset   (small smoke-test subset of dev)
  - eval         (full 3,209-utt eval set)
  - eval-test    (80% subset of eval used for fine-tuning evaluation)

For each results/<model>/<split>.json that exists, the matching
*-eval.json is regenerated using the corresponding TSV/STM/JSON
reference files.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent
DATA = ROOT / "info" / "sandi-corpus-2025"
REF = DATA / "reference-materials"
RESULTS = ROOT / "results"

SPLITS = {
    "dev": {
        "flist": REF / "flists.flac" / "dev-asr.tsv",
        "stm":   REF / "stms"        / "dev-asr.stm",
        "trans": REF / "annotations" / "dev-trans-ref.json",
        "gec":   REF / "annotations" / "dev-gec-ref.json",
    },
    "dev-subset": {
        "flist": REF / "flists.flac" / "dev-subset-asr.tsv",
        "stm":   REF / "stms"        / "dev-subset-asr.stm",
        "trans": REF / "annotations" / "dev-subset-trans-ref.json",
        "gec":   REF / "annotations" / "dev-subset-gec-ref.json",
    },
    "eval": {
        "flist": REF / "flists.flac" / "eval-asr.tsv",
        "stm":   REF / "stms"        / "eval-asr.stm",
        "trans": REF / "annotations" / "eval-trans-ref.json",
        "gec":   REF / "annotations" / "eval-gec-ref.json",
    },
    "eval-test": {
        "flist": REF / "flists.flac" / "eval-test-asr.tsv",
        "stm":   REF / "stms"        / "eval-test-asr.stm",
        "trans": REF / "annotations" / "eval-test-trans-ref.json",
        "gec":   REF / "annotations" / "eval-test-gec-ref.json",
    },
}


def main():
    jobs = []
    for model_dir in sorted(RESULTS.iterdir()):
        if not model_dir.is_dir():
            continue
        for split_name in SPLITS:
            src = model_dir / f"{split_name}.json"
            if not src.exists():
                continue
            dst = model_dir / f"{split_name}-eval.json"
            jobs.append((model_dir.name, split_name, src, dst))

    print(f"Found {len(jobs)} evaluations to re-run\n")

    for i, (model, split, src, dst) in enumerate(jobs, 1):
        cfg = SPLITS[split]
        cmd = [
            sys.executable, str(SRC / "evaluate.py"),
            "--results",        str(src),
            "--flist",          str(cfg["flist"]),
            "--data-root",      str(DATA),
            "--stm",            str(cfg["stm"]),
            "--annotations",    str(cfg["trans"]),
            "--gec-annotations", str(cfg["gec"]),
            "--output",         str(dst),
        ]
        t0 = time.time()
        print(f"[{i}/{len(jobs)}] {model} / {split} ... ", end="", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        if result.returncode != 0:
            print(f"FAILED ({dt:.1f}s)")
            print(result.stdout[-500:])
            print("STDERR:", result.stderr[-500:])
        else:
            try:
                ov = json.loads(dst.read_text(encoding="utf-8"))["gep"]["summary"]["_overall"]
                print(f"OK ({dt:.1f}s) | gep n={ov['total']} pres={ov['preservation_rate']*100:.2f}%")
            except Exception:
                print(f"OK ({dt:.1f}s)")


if __name__ == "__main__":
    main()
