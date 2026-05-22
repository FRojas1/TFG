"""
Generate figures comparing Parakeet TDT-CTC 110M vs Parakeet CTC 1.1B
across proficiency levels (WER and GEP).

Reuses the proficiency-band infrastructure from generate_proficiency_figures.py.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Libertinus Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR = Path(__file__).parent.parent / "results"
MARKS_DIR = (
    Path(__file__).parent.parent
    / "info"
    / "sandi-corpus-2025"
    / "reference-materials"
    / "sla-marks"
)

MODELS = {
    "parakeet-ctc-1.1b": "Parakeet CTC 1.1B",
    "parakeet-tdt-ctc-110m": "Parakeet TDT-CTC 110M",
}
COLORS = {
    "Parakeet CTC 1.1B": "#4CAF50",
    "Parakeet TDT-CTC 110M": "#81C784",
}
HATCHES = {
    "Parakeet CTC 1.1B": "",
    "Parakeet TDT-CTC 110M": "//",
}

BANDS = [
    ("Low (<3.5)", lambda s: s < 3.5),
    ("Medium (3.5-4.5)", lambda s: 3.5 <= s <= 4.5),
    ("High (>4.5)", lambda s: s > 4.5),
]
BAND_LABELS = [b[0] for b in BANDS]
BAND_COLORS = ["#e74c3c", "#f39c12", "#2ecc71"]


def load_sla_marks(split="eval"):
    path = MARKS_DIR / f"{split}-sla-overall.tsv"
    marks = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            marks[parts[0]] = float(parts[1])
    return marks


def session_id_from_file_id(file_id: str) -> str:
    parts = file_id.split("-")
    return "-".join(parts[:2])


def assign_band(score):
    for i, (_, pred) in enumerate(BANDS):
        if pred(score):
            return i
    return len(BANDS) - 1


def load_eval(model_key, split="eval"):
    path = RESULTS_DIR / model_key / f"{split}-eval.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_utterance_band_map(eval_data, sla_marks):
    fid_to_band = {}
    for utt in eval_data["wer"]["per_utterance"]:
        fid = utt["file_id"]
        sid = session_id_from_file_id(fid)
        if sid in sla_marks:
            fid_to_band[fid] = assign_band(sla_marks[sid])
    return fid_to_band


def compute_wer_by_band(eval_data, fid_to_band):
    band_subs = defaultdict(int)
    band_dels = defaultdict(int)
    band_ins = defaultdict(int)
    band_ref = defaultdict(int)
    band_utt_wers = defaultdict(list)

    for utt in eval_data["wer"]["per_utterance"]:
        fid = utt["file_id"]
        if fid not in fid_to_band:
            continue
        band = fid_to_band[fid]
        band_subs[band] += utt["substitutions"]
        band_dels[band] += utt["deletions"]
        band_ins[band] += utt["insertions"]
        band_ref[band] += utt["ref_words"]
        band_utt_wers[band].append(utt["wer"])

    results = {}
    for bi in range(len(BANDS)):
        total_errs = band_subs[bi] + band_dels[bi] + band_ins[bi]
        ref = band_ref[bi]
        results[bi] = {
            "wer": total_errs / ref if ref > 0 else 0,
            "ref_words": ref,
            "n_utterances": len(band_utt_wers[bi]),
            "median_wer": float(np.median(band_utt_wers[bi])) if band_utt_wers[bi] else 0,
        }
    return results


def compute_gep_by_band(eval_data, fid_to_band):
    band_counts = defaultdict(lambda: {"preserved": 0, "corrected": 0, "mutated": 0, "total": 0})

    for utt in eval_data["gep"]["per_utterance"]:
        fid = utt["file_id"]
        if fid not in fid_to_band:
            continue
        band = fid_to_band[fid]
        for etype, counts in utt["counts"].items():
            band_counts[band]["preserved"] += counts["preserved"]
            band_counts[band]["corrected"] += counts["corrected"]
            band_counts[band]["mutated"] += counts["mutated"]
            band_counts[band]["total"] += counts["total"]

    results = {}
    for bi in range(len(BANDS)):
        c = band_counts[bi]
        t = c["total"]
        results[bi] = {
            "preservation_rate": c["preserved"] / t if t > 0 else 0,
            "correction_rate": c["corrected"] / t if t > 0 else 0,
            "mutation_rate": c["mutated"] / t if t > 0 else 0,
            "total_errors": t,
        }
    return results


def fig35_parakeet_wer_by_proficiency(all_wer_by_band):
    """Side-by-side bar chart: Parakeet 110M vs 1.1B WER by proficiency."""
    x = np.arange(len(BAND_LABELS))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (key, label) in enumerate(MODELS.items()):
        wers = [all_wer_by_band[key][bi]["wer"] * 100 for bi in range(len(BANDS))]
        bars = ax.bar(x + i * width - width / 2, wers, width, label=label,
                      color=COLORS[label], hatch=HATCHES[label],
                      edgecolor="black", linewidth=0.6)
        for bar, val in zip(bars, wers):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS, fontsize=11)
    ax.set_ylabel("WER (%)")
    ax.set_title("WER by Proficiency Level: Parakeet 110M vs. 1.1B")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_wer = max(
        all_wer_by_band[k][bi]["wer"] * 100
        for k in MODELS for bi in range(len(BANDS))
    )
    ax.set_ylim(0, max_wer * 1.25)

    for bi in range(len(BANDS)):
        w_big = all_wer_by_band["parakeet-ctc-1.1b"][bi]["wer"] * 100
        w_small = all_wer_by_band["parakeet-tdt-ctc-110m"][bi]["wer"] * 100
        diff = w_small - w_big
        mid_x = x[bi]
        mid_y = max(w_big, w_small) + 2.5
        ax.annotate(f"$\\Delta$ = {diff:+.1f} pp",
                    xy=(mid_x, mid_y), ha="center", fontsize=8.5, color="#555")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig35_parakeet_wer_proficiency.png")
    plt.close(fig)
    print("  -> fig35_parakeet_wer_proficiency.png")


def fig36_parakeet_gep_by_proficiency(all_gep_by_band):
    """Side-by-side bar chart: Parakeet 110M vs 1.1B GEP by proficiency."""
    x = np.arange(len(BAND_LABELS))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (key, label) in enumerate(MODELS.items()):
        rates = [all_gep_by_band[key][bi]["preservation_rate"] * 100 for bi in range(len(BANDS))]
        bars = ax.bar(x + i * width - width / 2, rates, width, label=label,
                      color=COLORS[label], hatch=HATCHES[label],
                      edgecolor="black", linewidth=0.6)
        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 2.5,
                    f"{val:.1f}%", ha="center", va="top", fontsize=9, fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#333", alpha=0.7))

    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS, fontsize=11)
    ax.set_ylabel("GEP - Preservation (%)")
    ax.set_title("Grammatical Error Preservation by Level: Parakeet 110M vs. 1.1B")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(55, 82)

    for bi in range(len(BANDS)):
        g_big = all_gep_by_band["parakeet-ctc-1.1b"][bi]["preservation_rate"] * 100
        g_small = all_gep_by_band["parakeet-tdt-ctc-110m"][bi]["preservation_rate"] * 100
        diff = g_small - g_big
        mid_x = x[bi]
        mid_y = max(g_big, g_small) + 1.5
        ax.annotate(f"$\\Delta$ = {diff:+.1f} pp",
                    xy=(mid_x, mid_y), ha="center", fontsize=8.5, color="#555")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig36_parakeet_gep_proficiency.png")
    plt.close(fig)
    print("  -> fig36_parakeet_gep_proficiency.png")


def fig37_parakeet_correction_by_proficiency(all_gep_by_band):
    """Side-by-side bar chart: autocorrection rate by proficiency."""
    x = np.arange(len(BAND_LABELS))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (key, label) in enumerate(MODELS.items()):
        rates = [all_gep_by_band[key][bi]["correction_rate"] * 100 for bi in range(len(BANDS))]
        bars = ax.bar(x + i * width - width / 2, rates, width, label=label,
                      color=COLORS[label], hatch=HATCHES[label],
                      edgecolor="black", linewidth=0.6)
        for bar, val in zip(bars, rates):
            offset = -width / 2 if i == 0 else width / 2
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    max_rate = max(
        all_gep_by_band[k][bi]["correction_rate"] * 100
        for k in MODELS for bi in range(len(BANDS))
    )
    ax.set_ylim(0, max_rate * 1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS, fontsize=11)
    ax.set_ylabel("Correction Rate (%)")
    ax.set_title("Grammatical Auto-Correction by Level: Parakeet 110M vs. 1.1B")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig37_parakeet_correction_proficiency.png")
    plt.close(fig)
    print("  -> fig37_parakeet_correction_proficiency.png")


if __name__ == "__main__":
    sla_marks = load_sla_marks("eval")
    print(f"Loaded {len(sla_marks)} SLA marks")

    evals = {}
    for key in MODELS:
        evals[key] = load_eval(key, "eval")
        print(f"  Loaded {key}")

    all_wer_by_band = {}
    all_gep_by_band = {}

    for key, label in MODELS.items():
        fid_to_band = build_utterance_band_map(evals[key], sla_marks)
        all_wer_by_band[key] = compute_wer_by_band(evals[key], fid_to_band)
        all_gep_by_band[key] = compute_gep_by_band(evals[key], fid_to_band)

        print(f"\n{label}:")
        for bi, blabel in enumerate(BAND_LABELS):
            w = all_wer_by_band[key][bi]
            g = all_gep_by_band[key][bi]
            print(f"  {blabel}: WER={w['wer']:.2%} ({w['n_utterances']} utt, {w['ref_words']} words) | "
                  f"GEP pres={g['preservation_rate']:.2%}, corr={g['correction_rate']:.2%}, "
                  f"mut={g['mutation_rate']:.2%} ({g['total_errors']} errors)")

    print("\nGenerating comparison figures...")
    fig35_parakeet_wer_by_proficiency(all_wer_by_band)
    fig36_parakeet_gep_by_proficiency(all_gep_by_band)
    fig37_parakeet_correction_by_proficiency(all_gep_by_band)

    print("\nDone! Figures saved to", FIGURES_DIR)
