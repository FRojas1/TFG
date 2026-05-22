"""
Analyze word-level confusions introduced by fine-tuning.

Compares Whisper Small (base) vs LoRA fine-tuned on the same utterances to find
words that were correctly transcribed by the base model but incorrectly by the
fine-tuned model. Builds confusion distributions for the top-N worst cases.

Hypothesis: fine-tuning on L2 speech teaches the model "bad pronunciation"
patterns, causing it to hallucinate incorrect words where it previously used
language-model context to produce the right word.
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

# Match doc/generate_figures.py styling (Libertinus Serif; DPI; bbox).
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

# Document palette + neutrals (same base hues as generate_figures.COLORS).
COLOR_BLUE = "#2563EB"
COLOR_GOLD = "#FFB71F"
COLOR_GREEN = "#059669"
COLOR_RED = "#C82333"
COLOR_NEUTRAL = "#64748b"

# ------------------------------------------------------------------ #
# Text normalisation (same as evaluate.py)
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
    """Return per-ref-word alignment: list of (type, hyp_word_or_None)."""
    if not ref_words or not hyp_words:
        return []

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


def analyze_confusions(base_path, finetuned_path):
    """Find words correct in base but wrong in finetuned."""
    with open(base_path, encoding="utf-8") as f:
        base_data = json.load(f)
    with open(finetuned_path, encoding="utf-8") as f:
        ft_data = json.load(f)

    base_by_id = {r["file_id"]: r for r in base_data["results"]}
    ft_by_id = {r["file_id"]: r for r in ft_data["results"]}

    common_ids = set(base_by_id.keys()) & set(ft_by_id.keys())
    print(f"Common utterances: {len(common_ids)}")

    # Track: for each reference word that was CORRECT in base but WRONG in finetuned,
    # record (ref_word, ft_produced_word)
    new_substitutions = []  # (ref_word, ft_word)
    new_deletions = []  # ref_word that base got right but ft deleted

    # Also track overall stats
    base_correct_ft_wrong = 0
    base_correct_ft_correct = 0
    total_ref_words = 0

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
                if f_[0] == "equal":
                    base_correct_ft_correct += 1
                elif f_[0] == "substitute":
                    base_correct_ft_wrong += 1
                    new_substitutions.append((ref_word, f_[1]))
                elif f_[0] == "delete":
                    base_correct_ft_wrong += 1
                    new_deletions.append(ref_word)

    print(f"\nTotal reference words: {total_ref_words}")
    print(f"Base correct & FT correct: {base_correct_ft_correct}")
    print(f"Base correct & FT wrong:   {base_correct_ft_wrong}")
    print(f"  - Substitutions: {len(new_substitutions)}")
    print(f"  - Deletions:     {len(new_deletions)}")
    print(f"\nRegression rate (words correct->wrong): "
          f"{base_correct_ft_wrong / (base_correct_ft_correct + base_correct_ft_wrong) * 100:.2f}%")

    return new_substitutions, new_deletions


def build_confusion_analysis(new_substitutions):
    """Build per-reference-word confusion distributions."""
    # Count how often each reference word was newly substituted
    ref_word_counts = Counter(ref for ref, _ in new_substitutions)

    # For each reference word, what did the FT model produce instead?
    confusion_map = defaultdict(Counter)
    for ref, ft in new_substitutions:
        confusion_map[ref][ft] += 1

    return ref_word_counts, confusion_map


def generate_figures(ref_word_counts, confusion_map, output_dir):
    """Generate figures for the analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # --- Figure 1: Top 20 words that regressed most (color-coded) ---
    top_n = 20
    top_words = ref_word_counts.most_common(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    words = [w for w, _ in top_words]
    counts = [c for _, c in top_words]
    colors_bar = [COLOR_NEUTRAL if w in FUNCTION_WORDS else COLOR_RED for w in words]
    bars = ax.barh(range(len(words)), counts, color=colors_bar, alpha=0.85)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=11, fontfamily="monospace")
    ax.set_xlabel("Times Newly Incorrect (was correct in base model)", fontsize=12)
    ax.set_title("Top 20 Words: Correct in Base Whisper Small -> Wrong in Fine-tuned (LoRA)",
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
    fig.savefig(output_dir / "fig38_regression_top_words.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved: {output_dir / 'fig38_regression_top_words.png'}")

    # --- Figure 2: Top 3 confusion distributions ---
    interesting_words = []
    if "it's" in confusion_map:
        interesting_words.append(("it's", ref_word_counts["it's"]))
    for w, c in ref_word_counts.most_common(20):
        if w not in [x[0] for x in interesting_words]:
            interesting_words.append((w, c))
            if len(interesting_words) >= 3:
                break

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    subplot_colors = [COLOR_BLUE, COLOR_GOLD, COLOR_GREEN]

    for idx, (ref_word, total_count) in enumerate(interesting_words[:3]):
        ax = axes[idx]
        confusions = confusion_map[ref_word].most_common(7)
        sub_words = [w for w, _ in confusions]
        sub_counts = [c for _, c in confusions]

        ax.barh(range(len(sub_words)), sub_counts, color=subplot_colors[idx], alpha=0.85)
        ax.set_yticks(range(len(sub_words)))
        ax.set_yticklabels(sub_words, fontsize=10, fontfamily="monospace")
        ax.set_title(f'"{ref_word}" -> ??\n(n={total_count})',
                     fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        if idx == 0:
            ax.set_xlabel("Count", fontsize=10)

    plt.suptitle("Confusion Distributions: What Did the Fine-tuned Model Predict Instead?",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "fig39_confusion_distributions.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {output_dir / 'fig39_confusion_distributions.png'}")

    # --- Figure 3: Focused confusion pairs (Sankey-style horizontal) ---
    # Show the strongest confusion pairs as a directed graph
    top_pairs = Counter()
    for ref_w in ref_word_counts:
        for sub_w, cnt in confusion_map[ref_w].items():
            top_pairs[(ref_w, sub_w)] += cnt

    top15_pairs = top_pairs.most_common(15)

    fig, ax = plt.subplots(figsize=(12, 7))
    labels = [f"{ref} -> {sub}" for (ref, sub), _ in top15_pairs]
    values = [c for _, c in top15_pairs]
    pair_colors = [
        COLOR_RED if (ref not in FUNCTION_WORDS or sub not in FUNCTION_WORDS)
        else COLOR_NEUTRAL
        for (ref, sub), _ in top15_pairs
    ]

    ax.barh(range(len(labels)), values, color=pair_colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11, fontfamily="monospace")
    ax.set_xlabel("Occurrences", fontsize=12)
    ax.set_title("Top 15 Confusion Pairs: Regressions Introduced by Fine-tuning",
                 fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "fig40_confusion_pairs.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {output_dir / 'fig40_confusion_pairs.png'}")

    return interesting_words[:5]


def print_detailed_analysis(ref_word_counts, confusion_map):
    """Print detailed findings for the document."""
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS: TOP 5 WORD CONFUSIONS")
    print("=" * 70)

    top5 = ref_word_counts.most_common(5)
    for rank, (ref_word, total) in enumerate(top5, 1):
        print(f"\n{'-' * 50}")
        print(f"#{rank}: \"{ref_word}\" -- {total} new errors introduced by fine-tuning")
        print(f"{'-' * 50}")
        confusions = confusion_map[ref_word].most_common(10)
        print(f"  {'Predicted':<20} {'Count':>6} {'%':>8}")
        for sub_word, count in confusions:
            pct = count / total * 100
            print(f"  {sub_word:<20} {count:>6} {pct:>7.1f}%")

    print("\n" + "=" * 70)
    print("TOP 20 MOST REGRESSED WORDS")
    print("=" * 70)
    print(f"  {'Ref Word':<20} {'New Errors':>10} {'Top Confusion':>20}")
    for word, count in ref_word_counts.most_common(20):
        top_conf = confusion_map[word].most_common(1)[0] if confusion_map[word] else ("?", 0)
        print(f"  {word:<20} {count:>10} {top_conf[0]:>15} ({top_conf[1]}x)")


def main():
    base_path = REPO_ROOT / "results/whisper-small-en/eval.json"
    ft_path = REPO_ROOT / "results/whisper-small-en-lora-v2/eval-test.json"
    figures_dir = REPO_ROOT / "doc/figures"
    json_out = REPO_ROOT / "results/finetuning_confusion_analysis.json"

    if not base_path.exists() or not ft_path.exists():
        print("ERROR: Missing files:")
        if not base_path.exists():
            print(f"  {base_path}")
        if not ft_path.exists():
            print(f"  {ft_path}")
        sys.exit(1)

    print("Analyzing word-level regressions: base Whisper Small vs LoRA fine-tuned")
    print("=" * 70)

    new_substitutions, new_deletions = analyze_confusions(base_path, ft_path)

    ref_word_counts, confusion_map = build_confusion_analysis(new_substitutions)

    print_detailed_analysis(ref_word_counts, confusion_map)

    top5 = generate_figures(ref_word_counts, confusion_map, figures_dir)

    # Save raw data for reference
    output_data = {
        "total_new_substitutions": len(new_substitutions),
        "total_new_deletions": len(new_deletions),
        "top_20_regressed": [
            {"word": w, "count": c, "confusions": dict(confusion_map[w].most_common(10))}
            for w, c in ref_word_counts.most_common(20)
        ],
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nRaw data saved to {json_out}")


if __name__ == "__main__":
    main()
