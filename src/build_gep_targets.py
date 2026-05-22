"""Build per-utterance error word indices for targeted GEP training.

Runs ERRANT on (fluent_verbatim, GEC_reference) pairs from the S&I corpus
to identify which words in the verbatim transcript are grammatical errors.
Outputs a JSON cache mapping file_id -> list of error word indices (in the
fluent-verbatim word list, i.e. after stripping disfluencies/partials).

Usage:
    python src/build_gep_targets.py ^
        --trans-annotations info/sandi-corpus-2025/reference-materials/annotations/train-trans-ref.json ^
        --gec-annotations info/sandi-corpus-2025/reference-materials/annotations/train-gec-ref.json ^
        --output info/sandi-corpus-2025/reference-materials/cache/train-gep-targets.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import errant

from evaluate import GEP_EXCLUDED_TYPES


def load_annotations(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    by_id = {}
    for entry in data.get("files", []):
        by_id[entry["File-id"]] = entry
    return by_id


def extract_words_and_marks(transcript_items):
    words = []
    marks = []
    word_idx = 0
    for item in transcript_items:
        if "word" in item:
            words.append(item["word"])
            if "marks" in item:
                marks.append({"word_idx": word_idx, "marks": item["marks"]})
            word_idx += 1
    return words, marks


def build_fluent_words(words, marks):
    """Strip disfluency- and partial-marked words, return (lowercase list, index mapping).

    Returns:
        fluent_words: list of lowercase words with disfluencies/partials removed
        orig_to_fluent: dict mapping original word index -> fluent word index
    """
    skip = set()
    for m in marks:
        if any(t in ("disfluency", "partial") for t in m["marks"]):
            skip.add(m["word_idx"])

    fluent_words = []
    orig_to_fluent = {}
    fluent_idx = 0
    for i, w in enumerate(words):
        if i not in skip:
            fluent_words.append(w.lower())
            orig_to_fluent[i] = fluent_idx
            fluent_idx += 1

    return fluent_words, orig_to_fluent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trans-annotations", required=True)
    p.add_argument("--gec-annotations", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    trans_by_id = load_annotations(args.trans_annotations)
    gec_by_id = load_annotations(args.gec_annotations)

    common_ids = sorted(set(trans_by_id) & set(gec_by_id))
    print(f"Trans: {len(trans_by_id)}, GEC: {len(gec_by_id)}, Paired: {len(common_ids)}")

    annotator = errant.load("en")

    result = {}
    stats = defaultdict(int)
    total_errors = 0
    total_utterances_with_errors = 0

    for fid in common_ids:
        trans_entry = trans_by_id[fid]
        gec_entry = gec_by_id[fid]

        all_words, marks = extract_words_and_marks(trans_entry["Transcript"])
        fluent_words, orig_to_fluent = build_fluent_words(all_words, marks)
        gec_words = [w.lower() for w in
                     [item["word"] for item in gec_entry["Transcript"] if "word" in item]]

        fluent_text = " ".join(fluent_words)
        gec_text = " ".join(gec_words)

        if not fluent_text.strip() or not gec_text.strip():
            continue

        fluent_parsed = annotator.parse(fluent_text)
        gec_parsed = annotator.parse(gec_text)
        edits = annotator.annotate(fluent_parsed, gec_parsed)

        if not edits:
            result[fid] = {"error_word_indices": [], "edits": []}
            continue

        error_indices = set()
        edit_details = []

        for edit in edits:
            etype = edit.type
            if etype in GEP_EXCLUDED_TYPES:
                stats[f"_skipped:{etype}"] += 1
                continue
            stats[etype] += 1

            if edit.o_start == edit.o_end:
                # M:xxx (missing word) — the error is absence. Unmask tokens
                # bracketing the insertion point so the model learns NOT to
                # insert the corrected form at this position.
                if edit.o_start > 0:
                    error_indices.add(edit.o_start - 1)
                if edit.o_start < len(fluent_words):
                    error_indices.add(edit.o_start)
                edit_details.append({
                    "type": etype,
                    "category": "missing",
                    "fluent_span": [edit.o_start, edit.o_end],
                    "original": edit.o_str or "",
                    "correction": edit.c_str or "",
                })
            else:
                # R:xxx (replacement) or U:xxx (unnecessary) — the error
                # word(s) in the verbatim ARE the ones at o_start..o_end
                for i in range(edit.o_start, edit.o_end):
                    error_indices.add(i)
                edit_details.append({
                    "type": etype,
                    "category": "unnecessary" if not edit.c_str else "replacement",
                    "fluent_span": [edit.o_start, edit.o_end],
                    "original": edit.o_str or "",
                    "correction": edit.c_str or "",
                })

        error_list = sorted(error_indices)
        result[fid] = {
            "fluent_words": fluent_words,
            "error_word_indices": error_list,
            "edits": edit_details,
        }
        total_errors += len(error_list)
        if error_list:
            total_utterances_with_errors += 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(result)} entries to {args.output}")
    print(f"Utterances with errors: {total_utterances_with_errors}/{len(result)}")
    print(f"Total error word positions: {total_errors}")
    print(f"Avg errors per utterance (with errors): "
          f"{total_errors / max(1, total_utterances_with_errors):.1f}")
    print(f"\nTop error types:")
    for etype, count in sorted(stats.items(), key=lambda x: -x[1])[:15]:
        print(f"  {etype:<20} {count:>5}")


if __name__ == "__main__":
    main()
