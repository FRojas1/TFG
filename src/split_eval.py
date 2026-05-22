"""Split the eval set into val (20%) and test (80%) subsets.

Produces filtered TSV file lists, STM references, and annotation JSONs
that can be passed directly to transcribe.py and evaluate.py.

Usage:
    python src/split_eval.py --data-root info/sandi-corpus-2025 --seed 42
"""

import argparse
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Split eval set into val/test")
    p.add_argument("--data-root", required=True, help="Corpus root directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.2)
    args = p.parse_args()

    root = Path(args.data_root)
    ref = root / "reference-materials"
    flist_dir = ref / "flists.flac"
    stm_dir = ref / "stms"
    ann_dir = ref / "annotations"

    # 1. Load eval file IDs
    eval_ids = []
    eval_lines = {}
    with open(flist_dir / "eval-asr.tsv", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fid = line.split("\t", 1)[0]
            eval_ids.append(fid)
            eval_lines[fid] = line

    random.seed(args.seed)
    shuffled = eval_ids[:]
    random.shuffle(shuffled)

    n_val = int(len(shuffled) * args.val_ratio)
    val_ids = set(shuffled[:n_val])
    test_ids = set(shuffled[n_val:])

    print(f"Total eval: {len(eval_ids)}")
    print(f"Val: {len(val_ids)}, Test: {len(test_ids)}")

    # 2. Write TSV splits
    for name, ids in [("eval-val", val_ids), ("eval-test", test_ids)]:
        out = flist_dir / f"{name}-asr.tsv"
        with open(out, "w", encoding="utf-8") as f:
            for fid in eval_ids:
                if fid in ids:
                    f.write(eval_lines[fid] + "\n")
        print(f"Wrote {out} ({sum(1 for fid in eval_ids if fid in ids)} lines)")

    # 3. Filter STM
    for stm_name in ["eval-asr.stm", "eval-fluent.stm", "eval-gec.stm"]:
        src = stm_dir / stm_name
        if not src.exists():
            continue
        for name, ids in [("eval-val", val_ids), ("eval-test", test_ids)]:
            out_name = stm_name.replace("eval-", f"{name}-")
            out = stm_dir / out_name
            with open(src, encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
                for line in fin:
                    if line.startswith(";;") or not line.strip():
                        fout.write(line)
                        continue
                    fid = line.split(None, 1)[0]
                    if fid in ids:
                        fout.write(line)
            print(f"Wrote {out}")

    # 4. Filter annotation JSONs
    for ann_name in ["eval-trans-ref.json", "eval-gec-ref.json"]:
        src = ann_dir / ann_name
        if not src.exists():
            continue
        with open(src, encoding="utf-8") as f:
            data = json.load(f)

        for name, ids in [("eval-val", val_ids), ("eval-test", test_ids)]:
            out_name = ann_name.replace("eval-", f"{name}-")
            out = ann_dir / out_name
            filtered = {
                "ref-type": data["ref-type"],
                "files": [e for e in data["files"] if e["File-id"] in ids],
            }
            with open(out, "w", encoding="utf-8") as f:
                json.dump(filtered, f, ensure_ascii=False)
            print(f"Wrote {out} ({len(filtered['files'])} entries)")

    # 5. Summary
    gec_src = ann_dir / "eval-gec-ref.json"
    with open(gec_src, encoding="utf-8") as f:
        gec = json.load(f)
    gec_ids = {e["File-id"] for e in gec["files"]}
    print(f"\nGEC coverage: val={len(gec_ids & val_ids)}/{len(val_ids)}, "
          f"test={len(gec_ids & test_ids)}/{len(test_ids)}")


if __name__ == "__main__":
    main()
