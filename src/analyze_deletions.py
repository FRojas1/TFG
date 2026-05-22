"""
Analyze word-level deletion regressions introduced by fine-tuning.

Compares Whisper Small (base) vs LoRA fine-tuned on the same utterances to find
words that were correctly transcribed by the base model but *deleted* (omitted)
by the fine-tuned model. This is the deletion counterpart to the substitution
confusion analysis in analyze_finetuning_confusions.py.

Hypothesis: fine-tuning on L2 speech raises the model's acoustic detection
threshold, causing it to omit words that it previously transcribed correctly —
especially short, unstressed function words that L2 speakers under-articulate.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jiwer
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Libertinus Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

COLOR_BLUE = "#2563EB"
COLOR_GOLD = "#FFB71F"
COLOR_GREEN = "#059669"
COLOR_RED = "#C82333"
COLOR_NEUTRAL = "#64748b"

FUNCTION_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "it", "its",
    "it's", "i", "you", "he", "she", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "our", "their", "this", "that",
    "these", "those", "there", "here", "not", "no", "so", "if", "than",
    "then", "very", "just", "also", "too", "as", "about",
})

# ------------------------------------------------------------------ #
# Text normalisation (same as evaluate.py / analyze_finetuning_confusions.py)
# ------------------------------------------------------------------ #

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
    text = _HESITATION_RE.sub(" ", text)
    text = _PARTIAL_RE.sub(" ", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def normalize_hyp(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    words = text.split()
    words = [w for w in words if w not in _FILLER_WORDS]
    return " ".join(words)


# ------------------------------------------------------------------ #
# Core analysis
# ------------------------------------------------------------------ #

def get_word_alignments(ref_words, hyp_words):
    """Return per-ref-word alignment: list of (ref_idx, type, hyp_word_or_None)."""
    if not ref_words:
        return []
    if not hyp_words:
        return [(i, "delete", None) for i in range(len(ref_words))]

    output = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
    alignments = []

    for chunk in output.alignments[0]:
        ref_span = range(chunk.ref_start_idx, chunk.ref_end_idx)
        hyp_span = range(chunk.hyp_start_idx, chunk.hyp_end_idx)

        if chunk.type == "equal":
            for offset, ri in enumerate(ref_span):
                hi = chunk.hyp_start_idx + offset
                alignments.append((ri, "equal", hyp_words[hi]))
        elif chunk.type == "substitute":
            ref_list = list(ref_span)
            hyp_list = list(hyp_span)
            for offset, ri in enumerate(ref_list):
                if offset < len(hyp_list):
                    alignments.append((ri, "substitute", hyp_words[hyp_list[offset]]))
                else:
                    alignments.append((ri, "delete", None))
        elif chunk.type == "delete":
            for ri in ref_span:
                alignments.append((ri, "delete", None))

    return alignments


def analyze_deletion_regressions(base_path, finetuned_path):
    """Find words correctly transcribed by base but deleted by fine-tuned model."""
    with open(base_path, encoding="utf-8") as f:
        base_data = json.load(f)
    with open(finetuned_path, encoding="utf-8") as f:
        ft_data = json.load(f)

    base_by_id = {r["file_id"]: r for r in base_data["results"]}
    ft_by_id = {r["file_id"]: r for r in ft_data["results"]}

    common_ids = set(base_by_id.keys()) & set(ft_by_id.keys())
    print(f"Common utterances: {len(common_ids)}")

    new_deletions = []
    ref_word_counter = Counter()
    base_correct_count = Counter()
    total_ref_words = 0
    base_correct_ft_correct = 0
    base_correct_ft_substituted = 0
    base_correct_ft_deleted = 0

    for fid in sorted(common_ids):
        ref_raw = base_by_id[fid].get("reference") or ft_by_id[fid].get("reference")
        if not ref_raw:
            continue
        ref_text = normalize_ref(ref_raw)
        base_hyp = normalize_hyp(base_by_id[fid].get("hypothesis", ""))
        ft_hyp = normalize_hyp(ft_by_id[fid].get("hypothesis", ""))

        ref_words = ref_text.split()
        base_hyp_words = base_hyp.split()
        ft_hyp_words = ft_hyp.split()

        if not ref_words:
            continue

        total_ref_words += len(ref_words)
        for w in ref_words:
            ref_word_counter[w] += 1

        base_aligns = get_word_alignments(ref_words, base_hyp_words)
        ft_aligns = get_word_alignments(ref_words, ft_hyp_words)

        base_status = {}
        for ri, atype, hword in base_aligns:
            base_status[ri] = (atype, hword)

        ft_status = {}
        for ri, atype, hword in ft_aligns:
            ft_status[ri] = (atype, hword)

        for ri, ref_word in enumerate(ref_words):
            b = base_status.get(ri, ("delete", None))
            f_ = ft_status.get(ri, ("delete", None))

            if b[0] == "equal":
                base_correct_count[ref_word] += 1
                if f_[0] == "equal":
                    base_correct_ft_correct += 1
                elif f_[0] == "substitute":
                    base_correct_ft_substituted += 1
                elif f_[0] == "delete":
                    base_correct_ft_deleted += 1
                    new_deletions.append(ref_word)

    print(f"\nTotal reference words: {total_ref_words:,}")
    print(f"Words correct in base model: {base_correct_ft_correct + base_correct_ft_substituted + base_correct_ft_deleted:,}")
    print(f"  Still correct in FT:  {base_correct_ft_correct:,}")
    print(f"  Substituted by FT:    {base_correct_ft_substituted:,}")
    print(f"  *Deleted by FT:       {base_correct_ft_deleted:,}")
    print(f"\nDeletion regression rate: "
          f"{base_correct_ft_deleted / (base_correct_ft_correct + base_correct_ft_substituted + base_correct_ft_deleted) * 100:.2f}%")

    return new_deletions, ref_word_counter, base_correct_count


def build_deletion_analysis(new_deletions, base_correct_count):
    """Build per-word deletion frequency and regression rate."""
    del_counter = Counter(new_deletions)

    del_rate = {}
    for word, count in del_counter.items():
        base_correct = base_correct_count.get(word, 0)
        if base_correct > 0:
            del_rate[word] = count / base_correct
    return del_counter, del_rate


# ------------------------------------------------------------------ #
# Figures
# ------------------------------------------------------------------ #

def generate_figures(del_counter, del_rate, base_correct_count, output_dir):
    """Generate figures for the deletion regression analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 41: Top 20 words deleted by fine-tuning (correct in base) ---
    top_n = 20
    top_deleted = del_counter.most_common(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    words = [w for w, _ in top_deleted]
    counts = [c for _, c in top_deleted]
    colors_bar = [COLOR_NEUTRAL if w in FUNCTION_WORDS else COLOR_RED for w in words]

    ax.barh(range(len(words)), counts, color=colors_bar, alpha=0.85)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=11, fontfamily="monospace")
    ax.set_xlabel("New Deletions (correct in base, deleted in fine-tuned)")
    ax.set_title("Top 20 Words: Correct in Base Whisper Small -> Deleted in Fine-tuned (LoRA)",
                 fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_NEUTRAL, alpha=0.85, label="Function word"),
        Patch(facecolor=COLOR_RED, alpha=0.85, label="Content word"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_dir / "fig41_deletion_top_words.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved: {output_dir / 'fig41_deletion_top_words.png'}")

    # --- Figure 42: Deletion regression rate for top words ---
    min_base_correct = 100
    words_with_rate = [
        (w, del_rate[w] * 100, del_counter[w], base_correct_count[w])
        for w in del_counter
        if base_correct_count.get(w, 0) >= min_base_correct and w in del_rate
    ]
    words_with_rate.sort(key=lambda x: x[1], reverse=True)
    top_rate = words_with_rate[:20]

    fig, ax = plt.subplots(figsize=(12, 6))
    rate_words = [w for w, _, _, _ in top_rate]
    rates = [r for _, r, _, _ in top_rate]
    rate_colors = [COLOR_NEUTRAL if w in FUNCTION_WORDS else COLOR_RED for w in rate_words]

    bars = ax.barh(range(len(rate_words)), rates, color=rate_colors, alpha=0.85)
    ax.set_yticks(range(len(rate_words)))

    ylabels = []
    for w, rate, count, total in top_rate:
        ylabels.append(f"{w}  ({count}/{total})")
    ax.set_yticklabels(ylabels, fontsize=10, fontfamily="monospace")

    ax.set_xlabel("Deletion Regression Rate (%)")
    ax.set_title("Highest Deletion Regression Rates: P(deleted by FT | correct in base)",
                 fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_dir / "fig42_deletion_by_model.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {output_dir / 'fig42_deletion_by_model.png'}")


# ------------------------------------------------------------------ #
# Console output
# ------------------------------------------------------------------ #

def print_analysis(del_counter, del_rate, base_correct_count, ref_word_counter):
    """Print detailed analysis."""
    total_new_deletions = sum(del_counter.values())

    func_deletions = sum(c for w, c in del_counter.items() if w in FUNCTION_WORDS)
    content_deletions = total_new_deletions - func_deletions

    print("\n" + "=" * 70)
    print("DELETION REGRESSION ANALYSIS: BASE WHISPER SMALL vs LORA")
    print("=" * 70)
    print(f"\nTotal new deletions (base correct -> FT deleted): {total_new_deletions:,}")
    print(f"  Function words: {func_deletions:,} ({func_deletions/total_new_deletions*100:.1f}%)")
    print(f"  Content words:  {content_deletions:,} ({content_deletions/total_new_deletions*100:.1f}%)")

    print(f"\n{'='*70}")
    print("TOP 30 MOST-DELETED WORDS (by absolute count)")
    print(f"{'='*70}")
    print(f"  {'Word':<15} {'New Del':>8} {'Base OK':>8} {'Regr Rate':>10} {'Ref Total':>10} {'Type':<10}")
    print(f"  {'-'*65}")
    for word, count in del_counter.most_common(30):
        base_ok = base_correct_count.get(word, 0)
        rate = del_rate.get(word, 0) * 100
        total = ref_word_counter.get(word, 0)
        wtype = "function" if word in FUNCTION_WORDS else "content"
        print(f"  {word:<15} {count:>8} {base_ok:>8} {rate:>9.1f}% {total:>10} {wtype:<10}")

    print(f"\n{'='*70}")
    print("HIGHEST DELETION REGRESSION RATES (min 100 base-correct occurrences)")
    print(f"{'='*70}")
    words_with_rate = [
        (w, del_rate[w] * 100, del_counter[w], base_correct_count[w])
        for w in del_counter
        if base_correct_count.get(w, 0) >= 100 and w in del_rate
    ]
    words_with_rate.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'Word':<15} {'Regr Rate':>10} {'New Del':>8} {'Base OK':>8} {'Type':<10}")
    print(f"  {'-'*55}")
    for word, rate, count, base_ok in words_with_rate[:25]:
        wtype = "function" if word in FUNCTION_WORDS else "content"
        print(f"  {word:<15} {rate:>9.1f}% {count:>8} {base_ok:>8} {wtype:<10}")

    print(f"\n{'='*70}")
    print("CONTENT WORDS WITH HIGHEST DELETION COUNTS")
    print(f"{'='*70}")
    content_words = [(w, c) for w, c in del_counter.items() if w not in FUNCTION_WORDS]
    content_words.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'Word':<15} {'New Del':>8} {'Base OK':>8} {'Regr Rate':>10} {'Ref Total':>10}")
    print(f"  {'-'*55}")
    for word, count in content_words[:20]:
        base_ok = base_correct_count.get(word, 0)
        rate = del_rate.get(word, 0) * 100
        total = ref_word_counter.get(word, 0)
        print(f"  {word:<15} {count:>8} {base_ok:>8} {rate:>9.1f}% {total:>10}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    figures_dir = REPO_ROOT / "doc" / "figures"
    results_dir = REPO_ROOT / "results"

    base_path = results_dir / "whisper-small-en" / "eval-test.json"
    ft_path = results_dir / "whisper-small-en-lora" / "eval-test.json"

    if not base_path.exists():
        print(f"ERROR: {base_path} not found")
        sys.exit(1)
    if not ft_path.exists():
        print(f"ERROR: {ft_path} not found")
        sys.exit(1)

    print("Deletion Regression Analysis: Base Whisper Small vs LoRA Fine-tuned")
    print("=" * 70)
    print(f"Base model:      {base_path}")
    print(f"Fine-tuned model: {ft_path}")

    new_deletions, ref_word_counter, base_correct_count = \
        analyze_deletion_regressions(base_path, ft_path)

    del_counter, del_rate = build_deletion_analysis(new_deletions, base_correct_count)

    print_analysis(del_counter, del_rate, base_correct_count, ref_word_counter)
    generate_figures(del_counter, del_rate, base_correct_count, figures_dir)

    json_out = results_dir / "deletion_regression_analysis.json"
    output_data = {
        "total_new_deletions": sum(del_counter.values()),
        "function_word_deletions": sum(c for w, c in del_counter.items() if w in FUNCTION_WORDS),
        "content_word_deletions": sum(c for w, c in del_counter.items() if w not in FUNCTION_WORDS),
        "top_30_by_count": [
            {"word": w, "new_deletions": c, "base_correct": base_correct_count.get(w, 0),
             "regression_rate": del_rate.get(w, 0), "ref_total": ref_word_counter.get(w, 0),
             "type": "function" if w in FUNCTION_WORDS else "content"}
            for w, c in del_counter.most_common(30)
        ],
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nRaw data saved to: {json_out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
