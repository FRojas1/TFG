"""
Evaluation script: WER + EPR + GEP.

Takes a transcription results JSON (from transcribe.py) and the S&I corpus
annotations to compute:

  1. Word Error Rate (WER) — overall and per utterance
  2. Error Preservation Rate (EPR) — how often annotated disfluencies /
     pronunciation errors / partial words survive ASR transcription
  3. Grammatical Error Preservation (GEP) — uses ERRANT to identify real
     grammatical errors (by diffing the fluent verbatim against the GEC
     reference), then checks whether the ASR preserved or corrected each

Usage (from project root):

    python src/evaluate.py ^
        --results results/whisper-small-en/dev-subset.json ^
        --flist info/sandi-corpus-2025/reference-materials/flists.flac/dev-subset-asr.tsv ^
        --data-root info/sandi-corpus-2025 ^
        --stm info/sandi-corpus-2025/reference-materials/stms/dev-subset-asr.stm ^
        --annotations info/sandi-corpus-2025/reference-materials/annotations/dev-subset-trans-ref.json ^
        --gec-annotations info/sandi-corpus-2025/reference-materials/annotations/dev-subset-gec-ref.json ^
        --output results/whisper-small-en/dev-subset-eval.json
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import jiwer

from dataset import SANDiDataset


# ------------------------------------------------------------------ #
# Text normalisation
# ------------------------------------------------------------------ #
#
# The S&I STM references contain two special marker types that no ASR
# model will ever produce and that the official NIST SCTK scorer treats
# as "optionally deletable":
#
#   (%hesitation%)   Filler sounds (uh, um, hmm).  In the annotation
#                    JSON these are {"tag": "hesitation"} — structural
#                    markers, not spoken words.
#
#   (word-)          Partial / incomplete words the speaker started but
#                    abandoned, e.g. (univers-), (i-), (visi-).  In the
#                    annotation JSON: {"word": "UNIVERS", "marks": ["partial"]}.
#
# Both are removed from the reference before WER computation so that
# models are not penalised for failing to produce them.  Their
# preservation is tracked separately by the EPR metric using the
# annotation JSON (mark type "partial"; hesitations are tags, not
# words, so they never enter EPR).
#
# The official S&I GLM also maps common filler words in the *hypothesis*
# (uh, um, er, hmm …) to %HESITATION% so they don't count as
# insertions.  We apply the same mapping to the hypothesis.
# ------------------------------------------------------------------ #

_HESITATION_RE = re.compile(r"\(%hesitation%\)", re.IGNORECASE)
_PARTIAL_RE = re.compile(r"\(\S+?-\)")
_PUNCT_RE = re.compile(r"[^\w\s']")
_SPACE_RE = re.compile(r"\s+")

# Filler words that ASR models sometimes emit but that the S&I GLM
# treats as optionally deletable (mapped to %HESITATION%).
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
    """Normalise an STM reference for WER scoring.

    Removes (%hesitation%) markers, (partial-) word fragments, punctuation,
    and lowercases everything.
    """
    text = _HESITATION_RE.sub(" ", text)
    text = _PARTIAL_RE.sub(" ", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def normalize_hyp(text: str) -> str:
    """Normalise an ASR hypothesis for WER scoring.

    Lowercases, strips punctuation, and removes filler words that the
    S&I GLM maps to %HESITATION% (optionally deletable).
    """
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    words = text.split()
    words = [w for w in words if w not in _FILLER_WORDS]
    return " ".join(words)


# ------------------------------------------------------------------ #
# WER
# ------------------------------------------------------------------ #

def compute_wer_metrics(results: list[dict]) -> dict:
    """Compute overall and per-utterance WER from results list."""
    refs, hyps, file_ids = [], [], []

    for r in results:
        ref = r.get("reference")
        hyp = r.get("hypothesis", "")
        if ref is None:
            continue
        refs.append(normalize_ref(ref))
        hyps.append(normalize_hyp(hyp))
        file_ids.append(r["file_id"])

    wer_output = jiwer.process_words(refs, hyps)

    per_utterance = []
    for i, fid in enumerate(file_ids):
        utt_out = jiwer.process_words(refs[i], hyps[i])
        per_utterance.append({
            "file_id": fid,
            "wer": round(utt_out.wer, 4),
            "substitutions": utt_out.substitutions,
            "deletions": utt_out.deletions,
            "insertions": utt_out.insertions,
            "ref_words": len(refs[i].split()),
        })

    return {
        "overall_wer": round(wer_output.wer, 4),
        "total_substitutions": wer_output.substitutions,
        "total_deletions": wer_output.deletions,
        "total_insertions": wer_output.insertions,
        "total_ref_words": sum(u["ref_words"] for u in per_utterance),
        "per_utterance": per_utterance,
    }


# ------------------------------------------------------------------ #
# Error Preservation Rate (EPR)
# ------------------------------------------------------------------ #

def _build_ref_alignment_map(ref_words, hyp_words):
    """Align ref to hyp and return a dict: ref_word_idx -> (type, hyp_word|None).

    Types: 'equal', 'substitute', 'delete'.
    """
    if not ref_words or not hyp_words:
        return {}

    output = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
    ref_map = {}

    for chunk in output.alignments[0]:
        ref_span = range(chunk.ref_start_idx, chunk.ref_end_idx)
        hyp_span = range(chunk.hyp_start_idx, chunk.hyp_end_idx)

        if chunk.type == "equal":
            for offset, ri in enumerate(ref_span):
                hi = chunk.hyp_start_idx + offset
                ref_map[ri] = ("equal", hyp_words[hi])
        elif chunk.type == "substitute":
            ref_list = list(ref_span)
            hyp_list = list(hyp_span)
            for offset, ri in enumerate(ref_list):
                if offset < len(hyp_list):
                    ref_map[ri] = ("substitute", hyp_words[hyp_list[offset]])
                else:
                    ref_map[ri] = ("delete", None)
        elif chunk.type == "delete":
            for ri in ref_span:
                ref_map[ri] = ("delete", None)
        # 'insert' chunks don't consume reference words

    return ref_map


def compute_epr(results: list[dict], dataset: SANDiDataset) -> dict:
    """Compute Error Preservation Rate per annotation mark type.

    For each word in the reference that carries a mark (disfluency,
    pronunciation, partial), checks whether the ASR hypothesis preserved,
    deleted, or substituted it.

    Returns aggregate stats and per-utterance breakdown.
    """
    totals = defaultdict(lambda: {
        "preserved": 0, "deleted": 0, "substituted": 0, "total": 0
    })
    per_utterance = []

    for r in results:
        file_id = r["file_id"]
        hyp = r.get("hypothesis", "")

        try:
            utt = dataset[file_id]
        except KeyError:
            continue
        if not utt.annotations:
            continue

        info = SANDiDataset.extract_transcript_info(utt.annotations["Transcript"])
        if not info["marks"]:
            continue

        ref_words = [w.lower() for w in info["words"]]
        hyp_words = normalize_hyp(hyp).split()
        ref_map = _build_ref_alignment_map(ref_words, hyp_words)

        utt_marks = defaultdict(lambda: {
            "preserved": 0, "deleted": 0, "substituted": 0, "total": 0
        })

        for m in info["marks"]:
            wi = m["word_idx"]
            alignment = ref_map.get(wi, ("delete", None))

            for mark_type in m["marks"]:
                utt_marks[mark_type]["total"] += 1
                totals[mark_type]["total"] += 1

                if alignment[0] == "equal":
                    utt_marks[mark_type]["preserved"] += 1
                    totals[mark_type]["preserved"] += 1
                elif alignment[0] == "substitute":
                    utt_marks[mark_type]["substituted"] += 1
                    totals[mark_type]["substituted"] += 1
                else:
                    utt_marks[mark_type]["deleted"] += 1
                    totals[mark_type]["deleted"] += 1

        per_utterance.append({
            "file_id": file_id,
            "marks": {k: dict(v) for k, v in utt_marks.items()},
        })

    summary = {}
    for mark_type, counts in totals.items():
        total = counts["total"]
        summary[mark_type] = {
            **counts,
            "preservation_rate": round(counts["preserved"] / total, 4) if total else 0,
            "deletion_rate": round(counts["deleted"] / total, 4) if total else 0,
        }

    all_total = sum(c["total"] for c in totals.values())
    all_preserved = sum(c["preserved"] for c in totals.values())
    summary["_overall"] = {
        "total": all_total,
        "preserved": all_preserved,
        "deleted": sum(c["deleted"] for c in totals.values()),
        "substituted": sum(c["substituted"] for c in totals.values()),
        "preservation_rate": round(all_preserved / all_total, 4) if all_total else 0,
    }

    return {"summary": summary, "per_utterance": per_utterance}


# ------------------------------------------------------------------ #
# Grammatical Error Preservation (GEP)
# ------------------------------------------------------------------ #
#
# Approach:
#   1. Build a "fluent" form of the verbatim transcript by stripping
#      words marked as disfluency or partial (these are speech
#      production phenomena, not grammar).
#   2. Run ERRANT(fluent, GEC) to identify grammatical errors and
#      their types (R:PREP, R:DET, R:VERB:FORM, M:DET, …).
#   3. Filter out edit types that are not meaningful for L2 spoken
#      grammatical-error preservation (see GEP_EXCLUDED_TYPES below).
#   4. Align the fluent text to the ASR hypothesis with edit-distance.
#   5. For each remaining ERRANT edit, compare what the ASR produced
#      at the corresponding position against the erroneous and
#      corrected forms to classify as preserved / corrected / mutated.
# ------------------------------------------------------------------ #


# ERRANT edit types excluded from GEP scoring.
#
# Rationale:
#   - R:SPELL, R:ORTH       written-language artefacts. If the ASR
#                           transcribes a word differently from the
#                           verbatim annotation that's a WER concern,
#                           not a grammatical preservation concern.
#   - M:PUNCT, U:PUNCT      punctuation is not uniformly produced by
#                           ASR models (CTC backends emit none), so
#                           cross-model comparison is meaningless.
#   - M/R/U:OTHER           ERRANT's catch-all for multi-word lexical
#                           or paraphrastic edits. These do not model
#                           single grammatical errors and inflate the
#                           mutation rate mechanically because the ASR
#                           is unlikely to transcribe a long span verbatim.
#   - R:WO                  word-order edits cannot be scored reliably
#                           with edit-distance alignment (preserved-only
#                           in practice, never corrected).
GEP_EXCLUDED_TYPES = frozenset({
    "R:SPELL", "R:ORTH",
    "M:PUNCT", "U:PUNCT",
    "M:OTHER", "R:OTHER", "U:OTHER",
    "R:WO",
})

def _load_gec_words(gec_path: str) -> dict[str, list[str]]:
    """Load GEC annotation JSON, return {file_id: [word, …]}."""
    with open(gec_path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for entry in data.get("files", []):
        fid = entry["File-id"]
        words = [item["word"] for item in entry["Transcript"] if "word" in item]
        out[fid] = words
    return out


def _build_fluent_words(info: dict) -> list[str]:
    """Strip disfluency- and partial-marked words, return lowercase list."""
    skip = set()
    for m in info["marks"]:
        if any(t in ("disfluency", "partial") for t in m["marks"]):
            skip.add(m["word_idx"])
    return [w.lower() for i, w in enumerate(info["words"]) if i not in skip]


def _hyp_span_at(ref_map, hyp_words, start, end):
    """Get the ASR words aligned to fluent positions [start, end)."""
    words = []
    for i in range(start, end):
        aln = ref_map.get(i)
        if aln and aln[0] in ("equal", "substitute"):
            words.append(aln[1])
    return " ".join(words)


def _hyp_has_insertion_near(ref_map, hyp_words, pos, target):
    """Heuristic for Missing-type edits: check whether *target* appears in
    the hypothesis between the words aligned to *pos-1* and *pos*."""
    left_hyp = None
    right_hyp = None
    for i in range(pos - 1, -1, -1):
        aln = ref_map.get(i)
        if aln and aln[0] in ("equal", "substitute"):
            left_hyp = hyp_words.index(aln[1]) if aln[1] in hyp_words else None
            break
    for i in range(pos, max(ref_map.keys()) + 1 if ref_map else 0):
        aln = ref_map.get(i)
        if aln and aln[0] in ("equal", "substitute"):
            right_hyp = hyp_words.index(aln[1]) if aln[1] in hyp_words else None
            break

    if left_hyp is not None and right_hyp is not None and right_hyp > left_hyp:
        between = " ".join(hyp_words[left_hyp + 1 : right_hyp])
        return target.lower() in between.lower()
    return False


def compute_gep(
    results: list[dict],
    dataset: SANDiDataset,
    gec_path: str,
) -> dict:
    """Compute Grammatical Error Preservation per ERRANT error type."""
    import errant

    gec_by_id = _load_gec_words(gec_path)
    annotator = errant.load("en")

    totals = defaultdict(
        lambda: {"preserved": 0, "corrected": 0, "mutated": 0, "total": 0}
    )
    per_utterance = []

    for r in results:
        file_id = r["file_id"]
        hyp = r.get("hypothesis", "")

        try:
            utt = dataset[file_id]
        except KeyError:
            continue
        if not utt.annotations or file_id not in gec_by_id:
            continue

        info = SANDiDataset.extract_transcript_info(utt.annotations["Transcript"])
        fluent_words = _build_fluent_words(info)
        gec_words = [w.lower() for w in gec_by_id[file_id]]

        fluent_text = " ".join(fluent_words)
        gec_text = " ".join(gec_words)
        if not fluent_text.strip() or not gec_text.strip():
            continue

        fluent_parsed = annotator.parse(fluent_text)
        gec_parsed = annotator.parse(gec_text)
        edits = annotator.annotate(fluent_parsed, gec_parsed)
        if not edits:
            continue

        hyp_words = normalize_hyp(hyp).split()
        ref_map = _build_ref_alignment_map(fluent_words, hyp_words)

        utt_edits = []
        utt_counts = defaultdict(
            lambda: {"preserved": 0, "corrected": 0, "mutated": 0, "total": 0}
        )

        for edit in edits:
            etype = edit.type
            if etype in GEP_EXCLUDED_TYPES:
                continue
            orig = (edit.o_str or "").lower()
            corr = (edit.c_str or "").lower()

            if edit.o_start == edit.o_end:
                # Missing-word edit (M:xxx) — the error is the ABSENCE of a word.
                # If the ASR also lacks it → preserved; if ASR inserted it → corrected.
                inserted = _hyp_has_insertion_near(
                    ref_map, hyp_words, edit.o_start, corr
                )
                status = "corrected" if inserted else "preserved"
            elif not corr:
                # Unnecessary-word edit (U:xxx) — word should not be there.
                asr_span = _hyp_span_at(ref_map, hyp_words, edit.o_start, edit.o_end)
                if not asr_span:
                    status = "corrected"
                elif asr_span == orig:
                    status = "preserved"
                else:
                    status = "mutated"
            else:
                # Replacement edit (R:xxx) — the core case.
                asr_span = _hyp_span_at(ref_map, hyp_words, edit.o_start, edit.o_end)
                if asr_span == orig:
                    status = "preserved"
                elif asr_span == corr:
                    status = "corrected"
                else:
                    status = "mutated"

            utt_counts[etype]["total"] += 1
            utt_counts[etype][status] += 1
            totals[etype]["total"] += 1
            totals[etype][status] += 1

            utt_edits.append({
                "type": etype,
                "original": orig,
                "correction": corr,
                "asr_produced": _hyp_span_at(
                    ref_map, hyp_words, edit.o_start, edit.o_end
                ) if edit.o_start != edit.o_end else "(missing-type)",
                "status": status,
            })

        per_utterance.append({
            "file_id": file_id,
            "edits": utt_edits,
            "counts": {k: dict(v) for k, v in utt_counts.items()},
        })

    summary = {}
    for etype, counts in sorted(totals.items()):
        t = counts["total"]
        summary[etype] = {
            **counts,
            "preservation_rate": round(counts["preserved"] / t, 4) if t else 0,
            "correction_rate": round(counts["corrected"] / t, 4) if t else 0,
        }

    at = sum(c["total"] for c in totals.values())
    ap = sum(c["preserved"] for c in totals.values())
    ac = sum(c["corrected"] for c in totals.values())
    am = sum(c["mutated"] for c in totals.values())
    summary["_overall"] = {
        "total": at,
        "preserved": ap,
        "corrected": ac,
        "mutated": am,
        "preservation_rate": round(ap / at, 4) if at else 0,
        "correction_rate": round(ac / at, 4) if at else 0,
    }

    return {"summary": summary, "per_utterance": per_utterance}


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ASR results: WER + EPR + GEP")
    p.add_argument("--results", required=True, help="Transcription results JSON (from transcribe.py)")
    p.add_argument("--flist", required=True, help="TSV file list")
    p.add_argument("--data-root", required=True, help="Corpus root directory")
    p.add_argument("--stm", default=None, help="STM reference file")
    p.add_argument("--annotations", default=None, help="JSON annotation file (needed for EPR and GEP)")
    p.add_argument("--gec-annotations", default=None, help="GEC annotation JSON (needed for GEP)")
    p.add_argument("--output", default=None, help="Output JSON path (prints to stdout if omitted)")
    return p.parse_args()


def _fill_missing_references(results: list[dict], dataset: SANDiDataset) -> int:
    """Copy STM references into results when transcribe omitted --stm."""
    filled = 0
    for r in results:
        if r.get("reference") is not None:
            continue
        utt = dataset.utterances.get(r["file_id"])
        if utt and utt.reference:
            r["reference"] = utt.reference
            filled += 1
    return filled


def main():
    args = parse_args()

    with open(args.results, encoding="utf-8") as f:
        result_data = json.load(f)
    results = result_data["results"]
    model_name = result_data.get("model_name", "unknown")

    print(f"Evaluating {len(results)} utterances from model: {model_name}")

    dataset = None
    if args.stm or args.annotations:
        dataset = SANDiDataset(
            flist_path=args.flist,
            data_root=args.data_root,
            stm_path=args.stm,
            annotations_path=args.annotations,
        )
        if args.stm:
            filled = _fill_missing_references(results, dataset)
            if filled:
                print(f"Filled {filled} missing references from STM")

    # --- WER ---
    wer_metrics = compute_wer_metrics(results)
    print(f"\n{'='*50}")
    print(f"WER: {wer_metrics['overall_wer']:.2%}")
    print(f"  Substitutions: {wer_metrics['total_substitutions']}")
    print(f"  Deletions:     {wer_metrics['total_deletions']}")
    print(f"  Insertions:    {wer_metrics['total_insertions']}")
    print(f"  Ref words:     {wer_metrics['total_ref_words']}")

    # --- EPR ---
    epr_metrics = None
    if args.annotations:
        if dataset is None:
            dataset = SANDiDataset(
                flist_path=args.flist,
                data_root=args.data_root,
                stm_path=args.stm,
                annotations_path=args.annotations,
            )
        epr_metrics = compute_epr(results, dataset)

        print(f"\n{'='*50}")
        print("Error Preservation Rate (EPR) by mark type:\n")
        print(f"  {'Mark Type':<16} {'Total':>6} {'Preserved':>10} {'Deleted':>8} {'Subst':>6} {'EPR':>8}")
        print(f"  {'-'*56}")
        for mark_type, counts in sorted(epr_metrics["summary"].items()):
            if mark_type == "_overall":
                continue
            print(
                f"  {mark_type:<16} {counts['total']:>6} "
                f"{counts['preserved']:>10} {counts['deleted']:>8} "
                f"{counts['substituted']:>6} {counts['preservation_rate']:>7.1%}"
            )
        ov = epr_metrics["summary"]["_overall"]
        print(f"  {'-'*56}")
        print(
            f"  {'OVERALL':<16} {ov['total']:>6} "
            f"{ov['preserved']:>10} {ov['deleted']:>8} "
            f"{ov['substituted']:>6} {ov['preservation_rate']:>7.1%}"
        )
    else:
        print("\nSkipping EPR (no --annotations provided)")

    # --- GEP ---
    gep_metrics = None
    if args.annotations and args.gec_annotations:
        gep_metrics = compute_gep(results, dataset, args.gec_annotations)

        print(f"\n{'='*50}")
        print("Grammatical Error Preservation (GEP) by ERRANT type:\n")
        print(
            f"  {'Error Type':<16} {'Total':>6} {'Presrv':>7} "
            f"{'Corrct':>7} {'Mutatd':>7} {'GEP%':>7} {'Corr%':>7}"
        )
        print(f"  {'-'*60}")
        for etype, counts in sorted(gep_metrics["summary"].items()):
            if etype == "_overall":
                continue
            print(
                f"  {etype:<16} {counts['total']:>6} "
                f"{counts['preserved']:>7} {counts['corrected']:>7} "
                f"{counts['mutated']:>7} "
                f"{counts['preservation_rate']:>6.1%} "
                f"{counts['correction_rate']:>6.1%}"
            )
        ov = gep_metrics["summary"]["_overall"]
        print(f"  {'-'*60}")
        print(
            f"  {'OVERALL':<16} {ov['total']:>6} "
            f"{ov['preserved']:>7} {ov['corrected']:>7} "
            f"{ov['mutated']:>7} "
            f"{ov['preservation_rate']:>6.1%} "
            f"{ov['correction_rate']:>6.1%}"
        )
    elif not args.gec_annotations:
        print("\nSkipping GEP (no --gec-annotations provided)")

    # --- Save ---
    output_data = {
        "model_name": model_name,
        "source_results": args.results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wer": wer_metrics,
        "epr": epr_metrics,
        "gep": gep_metrics,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nFull results saved to {out_path}")
    else:
        print("\n(Use --output to save full results as JSON)")


if __name__ == "__main__":
    main()
