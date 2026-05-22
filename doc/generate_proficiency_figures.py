"""Generate figures showing metrics stratified by speaker proficiency level.

Uses the SLA overall marks (continuous 2-6 scale) from the S&I corpus to
group utterances into proficiency bands, then computes WER, GEP, and EPR
per band for each model.
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
    "whisper-small-en": "Whisper Small",
    "whisper-medium-en": "Whisper Medium",
    "wav2vec2-large-960h": "Wav2Vec2 Large",
    "parakeet-ctc-1.1b": "Parakeet CTC 1.1B",
}
COLORS = {
    "Whisper Small": "#2563EB",
    "Whisper Medium": "#FFB71F",
    "Wav2Vec2 Large": "#059669",
    "Parakeet CTC 1.1B": "#C82333",
}

BANDS = [
    ("Low (<3.5)", lambda s: s < 3.5),
    ("Medium (3.5-4.5)", lambda s: 3.5 <= s <= 4.5),
    ("High (>4.5)", lambda s: s > 4.5),
]
BAND_LABELS = [b[0] for b in BANDS]
# Proficiency bands (low / medium / high); aligned with document palette.
BAND_COLORS = ["#C82333", "#FFB71F", "#059669"]


def load_sla_marks(split="eval"):
    """Load SLA overall marks TSV → {session_id: score}."""
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
    """Extract session ID: 'SI114J-00054-P10006' → 'SI114J-00054'."""
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
    """Map file_id → band index using WER per_utterance file_ids."""
    fid_to_band = {}
    matched = 0
    for utt in eval_data["wer"]["per_utterance"]:
        fid = utt["file_id"]
        sid = session_id_from_file_id(fid)
        if sid in sla_marks:
            fid_to_band[fid] = assign_band(sla_marks[sid])
            matched += 1
    return fid_to_band


def compute_wer_by_band(eval_data, fid_to_band):
    """Compute corpus-level WER per proficiency band."""
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
            "utt_wers": band_utt_wers[bi],
        }
    return results


def compute_gep_by_band(eval_data, fid_to_band):
    """Compute GEP preservation / correction rates per proficiency band."""
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


def compute_epr_by_band(eval_data, fid_to_band):
    """Compute EPR preservation rate per proficiency band."""
    band_counts = defaultdict(lambda: {"preserved": 0, "deleted": 0, "substituted": 0, "total": 0})

    for utt in eval_data["epr"]["per_utterance"]:
        fid = utt["file_id"]
        if fid not in fid_to_band:
            continue
        band = fid_to_band[fid]
        for mark_type, counts in utt["marks"].items():
            band_counts[band]["preserved"] += counts["preserved"]
            band_counts[band]["deleted"] += counts["deleted"]
            band_counts[band]["substituted"] += counts["substituted"]
            band_counts[band]["total"] += counts["total"]

    results = {}
    for bi in range(len(BANDS)):
        c = band_counts[bi]
        t = c["total"]
        results[bi] = {
            "preservation_rate": c["preserved"] / t if t > 0 else 0,
            "total_marks": t,
        }
    return results


def fig20_wer_by_proficiency(all_wer_by_band):
    """Grouped bar chart of WER by proficiency band."""
    x = np.arange(len(BAND_LABELS))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (key, label) in enumerate(MODELS.items()):
        wers = [all_wer_by_band[key][bi]["wer"] * 100 for bi in range(len(BANDS))]
        bars = ax.bar(x + i * width, wers, width, label=label, color=COLORS[label],
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, wers):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(BAND_LABELS, fontsize=11)
    ax.set_ylabel("WER (%)")
    ax.set_title("WER by Speaker Proficiency Level")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(
        all_wer_by_band[k][bi]["wer"] * 100
        for k in MODELS for bi in range(len(BANDS))
    ) * 1.2)
    fig.savefig(FIGURES_DIR / "fig20_wer_by_proficiency.png")
    plt.close(fig)


def fig21_gep_by_proficiency(all_gep_by_band):
    """Grouped bar chart of GEP preservation rate by proficiency band."""
    x = np.arange(len(BAND_LABELS))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (key, label) in enumerate(MODELS.items()):
        rates = [all_gep_by_band[key][bi]["preservation_rate"] * 100 for bi in range(len(BANDS))]
        bars = ax.bar(x + i * width, rates, width, label=label, color=COLORS[label],
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(BAND_LABELS, fontsize=11)
    ax.set_ylabel("GEP - Preservation (%)")
    ax.set_title("Grammatical Error Preservation by Proficiency Level")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, framealpha=0.9, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 105)
    fig.savefig(FIGURES_DIR / "fig21_gep_by_proficiency.png")
    plt.close(fig)


def fig22_correction_by_proficiency(all_gep_by_band):
    """Grouped bar chart of GEP correction rate by proficiency band."""
    x = np.arange(len(BAND_LABELS))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (key, label) in enumerate(MODELS.items()):
        rates = [all_gep_by_band[key][bi]["correction_rate"] * 100 for bi in range(len(BANDS))]
        bars = ax.bar(x + i * width, rates, width, label=label, color=COLORS[label],
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(BAND_LABELS, fontsize=11)
    ax.set_ylabel("Correction Rate (%)")
    ax.set_title("Grammatical Auto-Correction Rate by Proficiency Level")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, framealpha=0.9, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(FIGURES_DIR / "fig22_correction_by_proficiency.png")
    plt.close(fig)


def fig23_combined_proficiency(all_wer_by_band, all_gep_by_band, all_epr_by_band):
    """3-panel figure: WER, GEP, EPR by proficiency, one subplot per metric."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    x = np.arange(len(BAND_LABELS))
    width = 0.2

    metric_fns = [
        ("WER (%)", lambda k, bi: all_wer_by_band[k][bi]["wer"] * 100),
        ("GEP - Preservation (%)", lambda k, bi: all_gep_by_band[k][bi]["preservation_rate"] * 100),
        ("EPR - Preservation (%)", lambda k, bi: all_epr_by_band[k][bi]["preservation_rate"] * 100),
    ]

    for mi, (metric_name, fn) in enumerate(metric_fns):
        ax = axes[mi]
        for i, (key, label) in enumerate(MODELS.items()):
            vals = [fn(key, bi) for bi in range(len(BANDS))]
            bars = ax.bar(x + i * width, vals, width, label=label, color=COLORS[label],
                          edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(BAND_LABELS, fontsize=9)
        ax.set_title(metric_name, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if mi == 0:
            ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

    fig.suptitle("ASR Metrics by Speaker Proficiency Level",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig23_combined_proficiency.png")
    plt.close(fig)


def fig24_proficiency_distribution(sla_marks):
    """Histogram of proficiency scores with band boundaries."""
    scores = list(sla_marks.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=20, color="#2563EB", edgecolor="white", alpha=0.8)
    ax.axvline(x=3.5, color="#C82333", linestyle="--", linewidth=1.5, label="Low/Medium Threshold (3.5)")
    ax.axvline(x=4.5, color="#059669", linestyle="--", linewidth=1.5, label="Medium/High Threshold (4.5)")
    ax.set_xlabel("SLA Score (continuous scale)")
    ax.set_ylabel("Number of Speakers")
    ax.set_title("Proficiency Level Distribution in the Eval Set")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(FIGURES_DIR / "fig24_proficiency_distribution.png")
    plt.close(fig)


def fig25_tradeoff_by_proficiency(all_wer_by_band, all_gep_by_band):
    """WER vs GEP scatter, one point per model per band."""
    markers = ["o", "s", "D"]
    fig, ax = plt.subplots(figsize=(8, 6))

    for bi, band_label in enumerate(BAND_LABELS):
        for key, label in MODELS.items():
            wer = all_wer_by_band[key][bi]["wer"] * 100
            gep = all_gep_by_band[key][bi]["preservation_rate"] * 100
            ax.scatter(wer, gep, s=160, color=COLORS[label], marker=markers[bi],
                       edgecolors="black", linewidths=0.6, zorder=5)

    for key, label in MODELS.items():
        ax.scatter([], [], color=COLORS[label], label=label, s=80)
    for bi, band_label in enumerate(BAND_LABELS):
        ax.scatter([], [], color="gray", marker=markers[bi], label=band_label, s=80)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    ax.set_xlabel("WER (%)")
    ax.set_ylabel("GEP - Preservation (%)")
    ax.set_title("WER vs. GEP Trade-off by Proficiency Level")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)
    fig.savefig(FIGURES_DIR / "fig25_tradeoff_by_proficiency.png")
    plt.close(fig)


if __name__ == "__main__":
    sla_marks = load_sla_marks("eval")
    print(f"Loaded {len(sla_marks)} SLA marks for eval set")
    scores = list(sla_marks.values())
    print(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")
    print(f"  Mean: {np.mean(scores):.3f}, Median: {np.median(scores):.3f}")

    band_counts = defaultdict(int)
    for s in scores:
        band_counts[assign_band(s)] += 1
    for bi, label in enumerate(BAND_LABELS):
        print(f"  {label}: {band_counts[bi]} speakers")

    evals = {k: load_eval(k, "eval") for k in MODELS}

    all_wer_by_band = {}
    all_gep_by_band = {}
    all_epr_by_band = {}

    for key, label in MODELS.items():
        fid_to_band = build_utterance_band_map(evals[key], sla_marks)
        all_wer_by_band[key] = compute_wer_by_band(evals[key], fid_to_band)
        all_gep_by_band[key] = compute_gep_by_band(evals[key], fid_to_band)
        all_epr_by_band[key] = compute_epr_by_band(evals[key], fid_to_band)

        print(f"\n{label}:")
        for bi, blabel in enumerate(BAND_LABELS):
            w = all_wer_by_band[key][bi]
            g = all_gep_by_band[key][bi]
            e = all_epr_by_band[key][bi]
            print(f"  {blabel}: WER={w['wer']:.2%} ({w['n_utterances']} utt, {w['ref_words']} words) | "
                  f"GEP={g['preservation_rate']:.2%} ({g['total_errors']} errors) | "
                  f"EPR={e['preservation_rate']:.2%} ({e['total_marks']} marks)")

    print("\nGenerating proficiency-stratified figures...")
    fig20_wer_by_proficiency(all_wer_by_band)
    print("  fig20_wer_by_proficiency.png")
    fig21_gep_by_proficiency(all_gep_by_band)
    print("  fig21_gep_by_proficiency.png")
    fig22_correction_by_proficiency(all_gep_by_band)
    print("  fig22_correction_by_proficiency.png")
    # fig23 removed from document; keeping function but not generating figure.
    # fig23_combined_proficiency(all_wer_by_band, all_gep_by_band, all_epr_by_band)
    # print("  fig23_combined_proficiency.png")
    fig24_proficiency_distribution(sla_marks)
    print("  fig24_proficiency_distribution.png")
    fig25_tradeoff_by_proficiency(all_wer_by_band, all_gep_by_band)
    print("  fig25_tradeoff_by_proficiency.png")

    print("\nDone! All proficiency figures saved to", FIGURES_DIR)
