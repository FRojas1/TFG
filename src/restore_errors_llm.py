"""Restore grammatical errors in ASR hypotheses using an LLM.

Takes the existing DPO pairs, re-runs ERRANT on the fluent/GEC references
to identify each corrected grammatical error WITH its position, asks
Gemini 3 Flash (via OpenRouter) to make a minimum edit to the raw ASR
hypothesis restoring those errors, then verifies the result by re-running
ERRANT against the edited text. Only utterances where every target error
flips back to "preserved" are kept; the rest are discarded. Overwrites
each DPO pair so that:

    chosen   = LLM-edited raw ASR hypothesis (Whisper's natural style,
               punctuation/casing intact, grammar error restored)
    rejected = raw ASR hypothesis (same style, error NOT restored)

Both fields use the same surface form, so the DPO diff is concentrated
on the error site and the model is not pulled away from its native
output distribution.

Usage:
    python src/restore_errors_llm.py ^
        --dpo-dir info/sandi-corpus-2025/reference-materials/dpo-pool ^
        --trans-ref info/sandi-corpus-2025/reference-materials/dpo-pool/combined-trans-ref.json ^
        --gec-ref info/sandi-corpus-2025/reference-materials/dpo-pool/combined-gec-ref.json ^
        --train-results results/whisper-small-en/train.json ^
        --eval-results results/whisper-small-en/eval.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

from evaluate import (
    GEP_EXCLUDED_TYPES,
    _build_ref_alignment_map,
    _hyp_has_insertion_near,
    _hyp_span_at,
    normalize_hyp,
)

MODEL = "google/gemini-3-flash-preview"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


SYSTEM_PROMPT = """\
You are a precise text editor. You will be given an ASR (speech recognition) \
transcript, a human reference transcript, and a list of grammatical errors that \
the ASR incorrectly "fixed". Your task is to restore those specific errors in \
the ASR transcript with the MINIMUM edits possible. Do NOT change anything \
else — keep the ASR's wording, casing, style, and word order intact. Only \
modify the specific words listed in the errors, and apply EACH listed edit \
exactly once at the position indicated."""

USER_TEMPLATE = """\
ASR transcript:
"{hypothesis}"

Reference transcript (stripped of hesitations and partials; this is the text ERRANT used to detect the errors):
"{reference}"

Grammatical errors the speaker actually made (the ASR fixed them, but we want them restored).
Each item lists the exact words that appear around the edit site in the reference, \
so you can locate the matching position in the ASR transcript:
{error_list}

Use the surrounding-context words to align each error to the right position in the ASR \
transcript, then make ONLY the listed edits. If the same word appears multiple times in \
the ASR transcript, only edit the occurrences that align with the listed errors."""

STRONGER_RETRY_SUFFIX = """\

Your previous attempt did not restore the errors correctly. Re-read the error list \
carefully. Make exactly one edit per listed error, at the position indicated by the \
surrounding-context words. Do not invent additional edits, do not duplicate edits, \
and do not change any other part of the ASR transcript."""

RESPONSE_SCHEMA = {
    "name": "edited_transcript",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "edited_transcript": {
                "type": "string",
                "description": "The ASR transcript with ONLY the listed errors restored, nothing else changed",
            },
        },
        "required": ["edited_transcript"],
        "additionalProperties": False,
    },
}


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #

def _build_fluent_words(transcript_items):
    """Return lowercase word list, dropping disfluency / partial tokens.

    Mirrors evaluate._build_fluent_words but operates on the raw JSON
    transcript list directly so we don't need to instantiate SANDiDataset.
    """
    words = []
    for item in transcript_items:
        if "word" not in item:
            continue
        marks = item.get("marks") or []
        if any(m in ("disfluency", "partial") for m in marks):
            continue
        words.append(item["word"].lower())
    return words


def _build_gec_words(transcript_items):
    return [item["word"].lower() for item in transcript_items if "word" in item]


# ------------------------------------------------------------------ #
# ERRANT-based target detection + verification
# ------------------------------------------------------------------ #

def _classify_edit(edit, fluent_words, hyp_words, ref_map):
    """Return the status string for an ERRANT edit against the given hypothesis.

    Replicates the per-edit classification logic in evaluate.compute_gep.
    """
    orig = (edit.o_str or "").lower()
    corr = (edit.c_str or "").lower()

    if edit.o_start == edit.o_end:
        inserted = _hyp_has_insertion_near(ref_map, hyp_words, edit.o_start, corr)
        return "corrected" if inserted else "preserved"
    if not corr:
        asr_span = _hyp_span_at(ref_map, hyp_words, edit.o_start, edit.o_end)
        if not asr_span:
            return "corrected"
        if asr_span == orig:
            return "preserved"
        return "mutated"
    asr_span = _hyp_span_at(ref_map, hyp_words, edit.o_start, edit.o_end)
    if asr_span == orig:
        return "preserved"
    if asr_span == corr:
        return "corrected"
    return "mutated"


def _edit_anchors(edit, fluent_words):
    """Return (left_context, right_context) words from the fluent reference."""
    left = fluent_words[edit.o_start - 1] if edit.o_start > 0 else None
    right = fluent_words[edit.o_end] if edit.o_end < len(fluent_words) else None
    return left, right


def _format_anchor(left, right):
    if left and right:
        return f' between "{left}" and "{right}"'
    if left:
        return f' after "{left}"'
    if right:
        return f' before "{right}"'
    return ""


def format_error_list(targets):
    """Format target edits into human-readable, positionally-anchored instructions."""
    lines = []
    for t in targets:
        etype = t["type"]
        prefix = etype.split(":")[0]
        anchor = _format_anchor(t["left"], t["right"])
        if prefix == "R":
            lines.append(
                f'- {etype}: Change "{t["asr_produced"] or t["correction"]}" to '
                f'"{t["original"]}"{anchor}.'
            )
        elif prefix == "M":
            lines.append(
                f'- {etype}: Remove the inserted word "{t["correction"]}"{anchor} '
                f"(the speaker never said it)."
            )
        elif prefix == "U":
            lines.append(
                f'- {etype}: Insert "{t["original"]}"{anchor} '
                f"(the speaker said this word, but the ASR dropped it)."
            )
    return "\n".join(lines)


def compute_targets(annotator, fluent_words, gec_words, hypothesis):
    """Identify `corrected` grammatical errors and produce LLM target descriptors.

    Returns (targets, edits) where:
      * targets is the list of edits with status="corrected" plus positional anchors
      * edits is the full ERRANT edit list (kept for reuse during verification)
    """
    if not fluent_words or not gec_words:
        return [], []

    fluent_text = " ".join(fluent_words)
    gec_text = " ".join(gec_words)
    if not fluent_text.strip() or not gec_text.strip():
        return [], []

    fluent_parsed = annotator.parse(fluent_text)
    gec_parsed = annotator.parse(gec_text)
    edits = annotator.annotate(fluent_parsed, gec_parsed)

    hyp_words = normalize_hyp(hypothesis).split()
    ref_map = _build_ref_alignment_map(fluent_words, hyp_words)

    targets = []
    for e in edits:
        if e.type in GEP_EXCLUDED_TYPES:
            continue
        status = _classify_edit(e, fluent_words, hyp_words, ref_map)
        if status != "corrected":
            continue
        left, right = _edit_anchors(e, fluent_words)
        asr_produced = (
            _hyp_span_at(ref_map, hyp_words, e.o_start, e.o_end)
            if e.o_start != e.o_end else ""
        )
        targets.append({
            "type": e.type,
            "original": (e.o_str or "").lower(),
            "correction": (e.c_str or "").lower(),
            "asr_produced": asr_produced,
            "left": left,
            "right": right,
            "o_start": e.o_start,
            "o_end": e.o_end,
        })
    return targets, edits


def verify_targets(fluent_words, edits, targets, edited_hyp):
    """Return True iff every target edit is now preserved in `edited_hyp`."""
    hyp_words = normalize_hyp(edited_hyp).split()
    ref_map = _build_ref_alignment_map(fluent_words, hyp_words)
    new_status = {}
    for e in edits:
        if e.type in GEP_EXCLUDED_TYPES:
            continue
        new_status[(e.type, e.o_start, e.o_end)] = _classify_edit(
            e, fluent_words, hyp_words, ref_map
        )
    for t in targets:
        key = (t["type"], t["o_start"], t["o_end"])
        if new_status.get(key) != "preserved":
            return False, new_status
    return True, new_status


# ------------------------------------------------------------------ #
# LLM call
# ------------------------------------------------------------------ #

def call_gemini(client, hypothesis, reference, targets, stronger=False, file_id="?"):
    """Call Gemini via OpenRouter once. Returns edited transcript or None."""
    error_list = format_error_list(targets)
    user_msg = USER_TEMPLATE.format(
        hypothesis=hypothesis,
        reference=reference,
        error_list=error_list,
    )
    if stronger:
        user_msg = user_msg + STRONGER_RETRY_SUFFIX

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=1.0,
            response_format={
                "type": "json_schema",
                "json_schema": RESPONSE_SCHEMA,
            },
            extra_body={
                "reasoning": {"effort": "low"},
                "provider": {"require_parameters": True},
            },
        )
        content = response.choices[0].message.content or ""
        return json.loads(content).get("edited_transcript", "").strip()
    except Exception as e:
        err_str = str(e)
        if "DEADLINE_EXCEEDED" in err_str or "504" in err_str or "timeout" in err_str.lower():
            log(f"    [{file_id}] Server timeout")
            return None
        log(f"    [{file_id}] LLM call failed: {e}")
        return None


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    p = argparse.ArgumentParser(description="Restore grammatical errors via LLM")
    p.add_argument("--dpo-dir", required=True,
                   help="Directory with finetune-{train,val,test}-dpo.json files")
    p.add_argument("--trans-ref", required=True,
                   help="Combined verbatim/transcript reference JSON (with marks)")
    p.add_argument("--gec-ref", required=True,
                   help="Combined GEC-corrected reference JSON")
    p.add_argument("--train-results", required=True,
                   help="Train transcription results JSON (raw ASR hypotheses)")
    p.add_argument("--eval-results", required=True,
                   help="Eval transcription results JSON (raw ASR hypotheses)")
    p.add_argument("--dry-run", type=int, default=0,
                   help="Process only this many items per split for testing (0 = all)")
    args = p.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    import errant
    log("Loading ERRANT annotator...")
    annotator = errant.load("en")

    log("Loading references...")
    with open(args.trans_ref, encoding="utf-8") as f:
        trans_data = json.load(f)
    with open(args.gec_ref, encoding="utf-8") as f:
        gec_data = json.load(f)

    fluent_by_id = {
        e["File-id"]: _build_fluent_words(e["Transcript"])
        for e in trans_data.get("files", [])
    }
    gec_by_id = {
        e["File-id"]: _build_gec_words(e["Transcript"])
        for e in gec_data.get("files", [])
    }

    log("Loading transcription results...")
    with open(args.train_results, encoding="utf-8") as f:
        train_results = json.load(f)
    with open(args.eval_results, encoding="utf-8") as f:
        eval_results = json.load(f)

    hyp_by_id = {}
    for r in train_results["results"]:
        hyp_by_id[r["file_id"]] = r.get("hypothesis", "")
    for r in eval_results["results"]:
        hyp_by_id[r["file_id"]] = r.get("hypothesis", "")

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=60.0,
    )

    dpo_dir = Path(args.dpo_dir)
    splits = ["train", "val", "test"]
    total_stats = {
        "processed": 0, "success": 0, "verify_failed": 0,
        "llm_failed": 0, "skipped": 0,
    }

    for split in splits:
        dpo_path = dpo_dir / f"finetune-{split}-dpo.json"
        if not dpo_path.exists():
            log(f"Skipping {split}: {dpo_path} not found")
            continue

        log(f"\n{'='*60}")
        log(f"Processing {split} split: {dpo_path}")
        log(f"{'='*60}")

        with open(dpo_path, encoding="utf-8") as f:
            dpo_pairs = json.load(f)

        items = list(dpo_pairs.items())
        if args.dry_run > 0:
            items = items[:args.dry_run]

        updated = 0
        verify_failed = 0
        llm_failed = 0
        skipped = 0

        for i, (file_id, pair) in enumerate(items):
            hypothesis = hyp_by_id.get(file_id, "")
            if not hypothesis:
                skipped += 1
                continue

            fluent_words = fluent_by_id.get(file_id)
            gec_words = gec_by_id.get(file_id)
            if not fluent_words or not gec_words:
                pair["chosen"] = hypothesis
                pair["rejected"] = hypothesis
                skipped += 1
                continue

            try:
                targets, edits = compute_targets(
                    annotator, fluent_words, gec_words, hypothesis
                )
            except Exception as e:
                log(f"  [{file_id}] ERRANT failed: {e}")
                pair["chosen"] = hypothesis
                pair["rejected"] = hypothesis
                skipped += 1
                continue

            if not targets:
                if pair.get("chosen") != hypothesis:
                    log(f"  [{file_id}] NO targets. Setting chosen/rejected to raw ASR hypothesis.")
                pair["chosen"] = hypothesis
                pair["rejected"] = hypothesis
                skipped += 1
                continue

            reference = " ".join(fluent_words)

            existing_chosen = pair.get("chosen", "")
            existing_rejected = pair.get("rejected", "")
            is_llm_edited = bool(existing_chosen and existing_rejected and existing_chosen != existing_rejected)

            if is_llm_edited:
                ok, _ = verify_targets(fluent_words, edits, targets, existing_chosen)
                if ok:
                    updated += 1
                    continue
                else:
                    log(f"  [{file_id}] Previously passed but failed NEW verification. Retrying...")

            attempts = []
            accepted = None
            for attempt_idx in range(2):
                edited = call_gemini(
                    client,
                    hypothesis,
                    reference,
                    targets,
                    stronger=(attempt_idx == 1),
                    file_id=file_id,
                )
                attempts.append(edited)
                if not edited:
                    continue
                ok, _ = verify_targets(fluent_words, edits, targets, edited)
                if ok:
                    accepted = edited
                    break

            if accepted is None:
                if all(a is None for a in attempts):
                    llm_failed += 1
                else:
                    verify_failed += 1
                
                pair["chosen"] = hypothesis
                pair["rejected"] = hypothesis

                if args.dry_run > 0 or i < 5:
                    log(f"\n  [{file_id}] DISCARDED after {len(attempts)} attempts")
                    log(f"    Hypothesis: {hypothesis}")
                    for t in targets:
                        log(f"    Target: {t['type']} | "
                            f"'{t['asr_produced'] or t['correction']}' -> '{t['original']}' "
                            f"| anchor=({t['left']} | {t['right']})")
                    for j, a in enumerate(attempts):
                        log(f"    Attempt {j+1}: {a}")
            else:
                dpo_pairs[file_id]["chosen"] = accepted
                dpo_pairs[file_id]["rejected"] = hypothesis
                updated += 1
                if args.dry_run > 0 or i < 5:
                    log(f"\n  [{file_id}] ACCEPTED (attempt {attempts.index(accepted)+1})")
                    log(f"    rejected (raw_hyp): {hypothesis}")
                    log(f"    chosen   (edited):  {accepted}")
                    for t in targets:
                        log(f"    Target: {t['type']} | "
                            f"'{t['asr_produced'] or t['correction']}' -> '{t['original']}' "
                            f"| anchor=({t['left']} | {t['right']})")

            if (i + 1) % 50 == 0:
                log(f"  Progress: {i+1}/{len(items)} "
                    f"(updated={updated}, verify_failed={verify_failed}, "
                    f"llm_failed={llm_failed}, skipped={skipped})")

            time.sleep(0.1)

        total_stats["processed"] += len(items) - skipped
        total_stats["success"] += updated
        total_stats["verify_failed"] += verify_failed
        total_stats["llm_failed"] += llm_failed
        total_stats["skipped"] += skipped

        log(f"\n  {split} results: updated={updated}, "
            f"verify_failed={verify_failed}, llm_failed={llm_failed}, "
            f"skipped={skipped}")

        if args.dry_run == 0:
            with open(dpo_path, "w", encoding="utf-8") as f:
                json.dump(dpo_pairs, f, ensure_ascii=False, indent=2)
            log(f"  Written: {dpo_path}")

    log(f"\n{'='*60}")
    log(f"TOTAL: processed={total_stats['processed']}, "
        f"success={total_stats['success']}, "
        f"verify_failed={total_stats['verify_failed']}, "
        f"llm_failed={total_stats['llm_failed']}, "
        f"skipped={total_stats['skipped']}")
    acceptance = total_stats["success"] / max(1, total_stats["processed"])
    log(f"Acceptance rate: {acceptance:.1%}")

    if args.dry_run == 0:
        stats_path = dpo_dir / "restoration_stats.json"
        with open(stats_path, "w") as f:
            json.dump(total_stats, f, indent=2)
        log(f"Stats saved: {stats_path}")


if __name__ == "__main__":
    main()
