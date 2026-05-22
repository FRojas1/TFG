"""Build a curated DPO training pool for GEP-targeted fine-tuning.

Pools all available data (train + eval), filters to utterances where the
baseline model corrects at least one grammatical error of a whitelisted
ERRANT type, constructs DPO pairs (chosen=verbatim, rejected=hypothesis),
and splits 80/10/10 into train/val/test.

Usage:
    python src/build_finetune_pool.py ^
        --train-eval results/whisper-small-en/train-eval.json ^
        --train-results results/whisper-small-en/train.json ^
        --eval-eval results/whisper-small-en/eval-eval.json ^
        --eval-results results/whisper-small-en/eval.json ^
        --train-flist info/sandi-corpus-2025/reference-materials/flists.flac/train-asr.tsv ^
        --eval-flist info/sandi-corpus-2025/reference-materials/flists.flac/eval-asr.tsv ^
        --output-dir info/sandi-corpus-2025/reference-materials/dpo-pool ^
        --seed 42
"""

import argparse
import json
import random
import re
from pathlib import Path

GEP_WHITELIST = frozenset({
    "M:ADJ", "M:ADV", "M:CONJ", "M:DET", "M:NOUN", "M:PART",
    "M:PREP", "M:PRON", "M:VERB", "M:VERB:FORM", "M:VERB:TENSE",
    "R:ADJ", "R:ADJ:FORM", "R:ADV", "R:CONJ", "R:DET", "R:MORPH",
    "R:NOUN", "R:NOUN:INFL", "R:NOUN:NUM", "R:PART", "R:PREP",
    "R:PRON", "R:VERB", "R:VERB:FORM", "R:VERB:INFL", "R:VERB:SVA",
    "R:VERB:TENSE",
    "U:ADJ", "U:ADV", "U:CONJ", "U:DET", "U:NOUN", "U:PART",
    "U:PREP", "U:PRON", "U:VERB", "U:VERB:FORM", "U:VERB:TENSE",
})

_HESITATION_RE = re.compile(r"\(%hesitation%\)", re.IGNORECASE)
_PARTIAL_RE = re.compile(r"\(\S+?-\)")
_PUNCT_RE = re.compile(r"[^\w\s']")
_SPACE_RE = re.compile(r"\s+")

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


def normalize_ref(text: str) -> str:
    """Normalize STM reference: strip hesitations, partials, punct, lowercase."""
    text = _HESITATION_RE.sub(" ", text)
    text = _PARTIAL_RE.sub(" ", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def normalize_hyp(text: str) -> str:
    """Normalize hypothesis: lowercase, strip punct, remove fillers."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    words = [w for w in text.split() if w not in _FILLER_WORDS]
    return " ".join(words)


def load_flist(path: str) -> dict[str, str]:
    """Load TSV flist, return {file_id: relative_audio_path}."""
    entries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            file_id, rel_path = line.split("\t", 1)
            entries[file_id] = rel_path
    return entries


def main():
    p = argparse.ArgumentParser(description="Build DPO training pool for GEP fine-tuning")
    p.add_argument("--train-eval", required=True, help="Train GEP evaluation JSON")
    p.add_argument("--train-results", required=True, help="Train transcription results JSON")
    p.add_argument("--eval-eval", required=True, help="Eval GEP evaluation JSON")
    p.add_argument("--eval-results", required=True, help="Eval transcription results JSON")
    p.add_argument("--train-flist", required=True, help="Train TSV file list")
    p.add_argument("--eval-flist", required=True, help="Eval TSV file list")
    p.add_argument("--output-dir", required=True, help="Output directory for pool files")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    args = p.parse_args()

    assert abs(args.train_ratio + args.val_ratio - 0.9) < 1e-6, \
        "train + val must equal 0.9 (test gets the remaining 0.1)"

    # Load evaluation results (GEP per-utterance)
    print("Loading evaluation results...")
    with open(args.train_eval, encoding="utf-8") as f:
        train_eval = json.load(f)
    with open(args.eval_eval, encoding="utf-8") as f:
        eval_eval = json.load(f)

    # Load transcription results (for hypotheses and references)
    with open(args.train_results, encoding="utf-8") as f:
        train_results = json.load(f)
    with open(args.eval_results, encoding="utf-8") as f:
        eval_results = json.load(f)

    # Build hypothesis/reference lookup from results
    hyp_by_id = {}
    ref_by_id = {}
    for r in train_results["results"]:
        hyp_by_id[r["file_id"]] = r.get("hypothesis", "")
        ref_by_id[r["file_id"]] = r.get("reference", "")
    for r in eval_results["results"]:
        hyp_by_id[r["file_id"]] = r.get("hypothesis", "")
        ref_by_id[r["file_id"]] = r.get("reference", "")

    # Load flists for audio paths
    train_flist = load_flist(args.train_flist)
    eval_flist = load_flist(args.eval_flist)
    all_flist = {**train_flist, **eval_flist}

    # Combine GEP per-utterance from both sets
    gep_per_utt = {}
    for entry in train_eval.get("gep", {}).get("per_utterance", []):
        gep_per_utt[entry["file_id"]] = entry
    for entry in eval_eval.get("gep", {}).get("per_utterance", []):
        gep_per_utt[entry["file_id"]] = entry

    print(f"Total utterances with GEP data: {len(gep_per_utt)}")
    print(f"Total utterances with hypotheses: {len(hyp_by_id)}")
    print(f"Total utterances in flists: {len(all_flist)}")

    # Filter: utterances with at least one corrected error of a whitelisted type
    pool = []
    for file_id, gep_entry in gep_per_utt.items():
        if file_id not in hyp_by_id or file_id not in ref_by_id:
            continue
        if file_id not in all_flist:
            continue

        corrected_types = []
        for edit in gep_entry.get("edits", []):
            if edit["type"] in GEP_WHITELIST and edit["status"] == "corrected":
                corrected_types.append(edit["type"])

        if not corrected_types:
            continue

        ref_raw = ref_by_id[file_id]
        hyp_raw = hyp_by_id[file_id]

        if not ref_raw or not hyp_raw:
            continue

        chosen = normalize_ref(ref_raw)
        rejected = normalize_hyp(hyp_raw)

        if not chosen or not rejected:
            continue
        if chosen == rejected:
            continue

        pool.append({
            "file_id": file_id,
            "audio_path": all_flist[file_id],
            "chosen": chosen,
            "rejected": rejected,
            "corrected_error_types": corrected_types,
        })

    print(f"\nFiltered pool: {len(pool)} utterances with corrected errors")

    # Shuffle and split 80/10/10
    random.seed(args.seed)
    random.shuffle(pool)

    n = len(pool)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_pool = pool[:n_train]
    val_pool = pool[n_train:n_train + n_val]
    test_pool = pool[n_train + n_val:]

    print(f"Split: train={len(train_pool)}, val={len(val_pool)}, test={len(test_pool)}")

    # Write outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train_pool), ("val", val_pool), ("test", test_pool)]:
        # Write flist
        flist_path = output_dir / f"finetune-{split_name}.tsv"
        with open(flist_path, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(f"{item['file_id']}\t{item['audio_path']}\n")

        # Write DPO pairs JSON
        pairs_path = output_dir / f"finetune-{split_name}-dpo.json"
        pairs = {}
        for item in split_data:
            pairs[item["file_id"]] = {
                "chosen": item["chosen"],
                "rejected": item["rejected"],
                "corrected_error_types": item["corrected_error_types"],
            }
        with open(pairs_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)

    # Print error type distribution
    type_counts = {}
    for item in pool:
        for t in item["corrected_error_types"]:
            type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\nCorrected error type distribution (across all {len(pool)} utterances):")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {etype:<16} {count:>5}")

    print(f"\nOutputs written to {output_dir}/")
    print(f"  finetune-train.tsv ({len(train_pool)} utterances)")
    print(f"  finetune-val.tsv   ({len(val_pool)} utterances)")
    print(f"  finetune-test.tsv  ({len(test_pool)} utterances)")
    print(f"  finetune-{{train,val,test}}-dpo.json (DPO pairs)")


if __name__ == "__main__":
    main()
