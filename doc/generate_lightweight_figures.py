"""
Generate comparison figures for the lightweight models section.

Compares Parakeet TDT-CTC 110M, Canary 180M Flash, and Moonshine Base
against the previously evaluated models (Whisper Small, Whisper Medium,
Wav2Vec2 Large, Parakeet CTC 1.1B).
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Libertinus Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_eval(model_dir, filename="eval-eval.json"):
    path = RESULTS_DIR / model_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"No eval file found: {path}")
    with open(path) as f:
        return json.load(f)


BASELINE_MODELS = {
    "Whisper Small (244M)": "whisper-small-en",
    "Whisper Medium (769M)": "whisper-medium-en",
    "Wav2Vec2 Large (317M)": "wav2vec2-large-960h",
    "Parakeet CTC 1.1B": "parakeet-ctc-1.1b",
}

LIGHTWEIGHT_MODELS = {
    "Parakeet TDT 110M": "parakeet-tdt-ctc-110m",
    "Canary 180M Flash": "canary-180m-flash",
    "Moonshine Base (61M)": "moonshine-base",
}

BASELINE_COLORS = {
    "Whisper Small (244M)": "#2563EB",
    "Whisper Medium (769M)": "#FFB71F",
    "Wav2Vec2 Large (317M)": "#059669",
    "Parakeet CTC 1.1B": "#C82333",
}

LIGHTWEIGHT_COLORS = {
    "Parakeet TDT 110M": "#E88D98",
    "Canary 180M Flash": "#7B1FA2",
    "Moonshine Base (61M)": "#FF9800",
}

GRAY = "#9E9E9E"
BLUE = "#2563EB"
LIGHTWEIGHT_KEYS = {"Parakeet TDT 110M", "Canary 180M Flash", "Moonshine Base (61M)"}


def get_bar_style(name, is_best=False):
    color = BLUE if is_best else GRAY
    hatch = '//' if name in LIGHTWEIGHT_KEYS else ''
    return color, hatch


def get_metrics(eval_data):
    """Extract key metrics from an eval JSON."""
    metrics = {}
    metrics["wer"] = eval_data.get("wer", {}).get("overall_wer", None)
    if metrics["wer"] is not None:
        metrics["wer"] *= 100

    gep = eval_data.get("gep", {})
    if gep:
        overall = gep.get("summary", {}).get("_overall", {})
        metrics["gep_preserved"] = overall.get("preservation_rate", 0) * 100
        metrics["gep_corrected"] = overall.get("correction_rate", 0) * 100
        total = overall.get("total", 0)
        mutated = overall.get("mutated", 0)
        metrics["gep_mutated"] = (mutated / total * 100) if total > 0 else 0
    else:
        metrics["gep_preserved"] = None
        metrics["gep_corrected"] = None
        metrics["gep_mutated"] = None

    epr = eval_data.get("epr", {})
    if epr:
        overall = epr.get("summary", {}).get("_overall", {})
        metrics["epr"] = overall.get("preservation_rate", 0) * 100
    else:
        metrics["epr"] = None

    return metrics


def fig_wer_comparison(all_metrics):
    """Bubble chart comparing WER across all models (sorted worst to best).

    Circle size is proportional to model parameter count.
    """
    params = {
        "Whisper Small (244M)": 244,
        "Whisper Medium (769M)": 769,
        "Wav2Vec2 Large (317M)": 317,
        "Parakeet CTC 1.1B": 1063,
        "Parakeet TDT 110M": 114,
        "Canary 180M Flash": 182,
        "Moonshine Base (61M)": 61,
    }
    size_scale = 0.8

    n_models = len(all_metrics)
    fig_w = max(6, n_models * 1.1)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    sorted_items = sorted(all_metrics.items(), key=lambda x: x[1]["wer"], reverse=True)
    names = [item[0] for item in sorted_items]
    wers = [item[1]["wer"] for item in sorted_items]
    best_name = names[-1]

    for i, (name, wer) in enumerate(zip(names, wers)):
        is_best = name == best_name
        color = BLUE if is_best else GRAY
        s = params.get(name, 200) * size_scale
        ax.scatter(i, wer, s=s, color=color, edgecolors="dimgray",
                   linewidths=0.8, zorder=5, alpha=0.85)
        # Offset the label above the circle edge (radius in display points + padding)
        radius_pts = math.sqrt(s / math.pi)
        ax.annotate(f"{wer:.1f}%", (i, wer),
                    xytext=(0, radius_pts + 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_xlim(-0.7, len(names) - 0.3)
    ax.set_ylabel("WER (%)")
    ax.set_title("WER: Lightweight Models vs. Baselines (eval set)")
    ax.set_ylim(0, max(wers) * 1.3)

    other_handle = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GRAY,
                               markeredgecolor="dimgray", markersize=8, label="Other models")
    best_handle = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE,
                              markeredgecolor="dimgray", markersize=8, label="Best overall")
    ax.legend(handles=[other_handle, best_handle], loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig30_lightweight_wer.png", bbox_inches="tight")
    plt.close()
    print("  -> fig30_lightweight_wer.png")


def fig_gep_comparison(all_metrics):
    """Bubble chart comparing GEP preservation across all models (sorted worst to best).

    Circle size is proportional to model parameter count.
    """
    params = {
        "Whisper Small (244M)": 244,
        "Whisper Medium (769M)": 769,
        "Wav2Vec2 Large (317M)": 317,
        "Parakeet CTC 1.1B": 1063,
        "Parakeet TDT 110M": 114,
        "Canary 180M Flash": 182,
        "Moonshine Base (61M)": 61,
    }
    size_scale = 0.8

    n_models = len(all_metrics)
    fig_w = max(6, n_models * 1.1)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    sorted_items = sorted(all_metrics.items(), key=lambda x: x[1]["gep_preserved"])
    names = [item[0] for item in sorted_items]
    geps = [item[1]["gep_preserved"] for item in sorted_items]
    best_name = names[-1]

    for i, (name, gep) in enumerate(zip(names, geps)):
        is_best = name == best_name
        color = BLUE if is_best else GRAY
        s = params.get(name, 200) * size_scale
        ax.scatter(i, gep, s=s, color=color, edgecolors="dimgray",
                   linewidths=0.8, zorder=5, alpha=0.85)
        if gep is not None:
            radius_pts = math.sqrt(s / math.pi)
            ax.annotate(f"{gep:.1f}%", (i, gep),
                        xytext=(0, radius_pts + 4),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_xlim(-0.7, len(names) - 0.3)
    ax.set_ylabel("GEP Preserved (%)")
    ax.set_title("Grammatical Error Preservation: Lightweight Models vs. Baselines")
    ax.set_ylim(min(geps) - 5, 100)

    other_handle = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GRAY,
                               markeredgecolor="dimgray", markersize=8, label="Other models")
    best_handle = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE,
                              markeredgecolor="dimgray", markersize=8, label="Best overall")
    ax.legend(handles=[other_handle, best_handle], loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig31_lightweight_gep.png", bbox_inches="tight")
    plt.close()
    print("  -> fig31_lightweight_gep.png")


def fig_correction_rate(all_metrics):
    """Bar chart of auto-correction rate. (Disabled — not called from main.)"""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(all_metrics.keys())
    corrs = [all_metrics[n]["gep_corrected"] for n in names]
    colors = []
    for n in names:
        if n in BASELINE_COLORS:
            colors.append(BASELINE_COLORS[n])
        else:
            colors.append(LIGHTWEIGHT_COLORS[n])

    bars = ax.bar(range(len(names)), corrs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Correction Rate (%)")
    ax.set_title("Auto-correction Rate: Lightweight Models vs. Baselines")

    for bar, val in zip(bars, corrs):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig32_lightweight_correction.png", bbox_inches="tight")
    plt.close()
    print("  -> fig32_lightweight_correction.png")


def fig_tradeoff(all_metrics):
    """Scatter plot: WER vs GEP for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    all_colors = {**BASELINE_COLORS, **LIGHTWEIGHT_COLORS}

    for name, m in all_metrics.items():
        if m["wer"] is None or m["gep_preserved"] is None:
            continue
        color = all_colors.get(name, GRAY)
        if name in BASELINE_COLORS:
            marker = "o"
            size = 120
        else:
            marker = "D"
            size = 100

        ax.scatter(m["wer"], m["gep_preserved"], c=color, s=size,
                   marker=marker, edgecolors="black", linewidth=0.8, zorder=5)
        ax.annotate(name, (m["wer"], m["gep_preserved"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, ha="left")

    ax.set_xlabel("WER (%)")
    ax.set_ylabel("GEP Preserved (%)")
    ax.set_title("WER vs. GEP Trade-off: All Models")

    baseline_patch = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                                 markersize=10, label="Baselines")
    lightweight_patch = plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
                                    markersize=9, label="Lightweight")
    ax.legend(handles=[baseline_patch, lightweight_patch], loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig33_lightweight_tradeoff.png", bbox_inches="tight")
    plt.close()
    print("  -> fig33_lightweight_tradeoff.png")


def fig_params_vs_wer(all_metrics):
    """Scatter: parameter count vs accuracy (100 - WER), showing efficiency."""
    params = {
        "Whisper Small (244M)": 244,
        "Whisper Medium (769M)": 769,
        "Wav2Vec2 Large (317M)": 317,
        "Parakeet CTC 1.1B": 1063,
        "Parakeet TDT 110M": 114,
        "Canary 180M Flash": 182,
        "Moonshine Base (61M)": 61,
    }

    all_colors = {**BASELINE_COLORS, **LIGHTWEIGHT_COLORS}

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, m in all_metrics.items():
        if m["wer"] is None or name not in params:
            continue
        color = all_colors.get(name, GRAY)
        if name in BASELINE_COLORS:
            marker = "o"
        else:
            marker = "D"

        accuracy = 100 - m["wer"]
        ax.scatter(params[name], accuracy, c=color, s=120,
                   marker=marker, edgecolors="black", linewidth=0.8, zorder=5)
        ax.annotate(name, (params[name], accuracy),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Size vs. Accuracy: Efficiency Comparison")
    ax.set_xscale("log")

    baseline_patch = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                                 markersize=10, label="Baselines")
    lightweight_patch = plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
                                    markersize=9, label="Lightweight")
    ax.legend(handles=[baseline_patch, lightweight_patch], loc="lower right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig34_lightweight_params_wer.png", bbox_inches="tight")
    plt.close()
    print("  -> fig34_lightweight_params_wer.png")


def main():
    print("Loading evaluation results...")

    all_metrics = {}

    for name, dirname in BASELINE_MODELS.items():
        try:
            data = load_eval(dirname)
            all_metrics[name] = get_metrics(data)
            print(f"  {name}: WER={all_metrics[name]['wer']:.2f}%")
        except Exception as e:
            print(f"  {name}: SKIPPED ({e})")

    for name, dirname in LIGHTWEIGHT_MODELS.items():
        try:
            data = load_eval(dirname)
            all_metrics[name] = get_metrics(data)
            print(f"  {name}: WER={all_metrics[name]['wer']:.2f}%")
        except Exception as e:
            print(f"  {name}: SKIPPED ({e})")

    if len(all_metrics) < 5:
        print(f"\nWARNING: Only {len(all_metrics)} models loaded. "
              "Run evaluations first for missing models.")

    print("\nGenerating figures...")
    fig_wer_comparison(all_metrics)
    fig_gep_comparison(all_metrics)
    # fig_correction_rate disabled — Figure 23 removed from document
    # fig_correction_rate(all_metrics)
    fig_tradeoff(all_metrics)
    fig_params_vs_wer(all_metrics)
    print("\nDone!")


if __name__ == "__main__":
    main()
