"""Generate all figures for the TFG document from evaluation results.

Uses the eval set (3 209 utterances, 13 783 grammatical errors) as the
primary data source.  Dev-set results are loaded when available for
cross-validation plots.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
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

def load_eval(model_key, split="eval"):
    path = RESULTS_DIR / model_key / f"{split}-eval.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fig1_wer_comparison(evals):
    """Bar chart comparing overall WER across models."""
    names = list(MODELS.values())
    wers = [evals[k]["wer"]["overall_wer"] * 100 for k in MODELS]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar(names, wers, color=[COLORS[n] for n in names], width=0.55, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, wers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("WER (%)")
    ax.set_title("Word Error Rate (WER) by Model")
    ax.set_ylim(0, max(wers) * 1.25)
    ax.set_xticklabels(names, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(FIGURES_DIR / "fig1_wer_comparison.png")
    plt.close(fig)


def fig2_wer_per_utterance(evals):
    """Scatter/strip chart of per-utterance WER across models."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (key, label) in enumerate(MODELS.items()):
        per_utt = evals[key]["wer"]["per_utterance"]
        wers = [u["wer"] * 100 for u in per_utt]
        x = np.random.normal(i, 0.08, size=len(wers))
        ax.scatter(x, wers, alpha=0.6, s=30, color=COLORS[label], edgecolors="white", linewidths=0.3)
        median = np.median(wers)
        ax.hlines(median, i - 0.25, i + 0.25, color=COLORS[label], linewidth=2.5, zorder=5)

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS.values())
    ax.set_ylabel("WER (%)")
    ax.set_title("WER Distribution per Utterance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_wer_distribution.png", bbox_inches="tight")
    plt.close(fig)


def fig3_epr_by_mark(evals):
    """Grouped bar chart of EPR by mark type."""
    mark_types = ["disfluency", "pronunciation", "partial"]
    mark_labels = ["Disfluency", "Pronunciation", "Partial"]
    x = np.arange(len(mark_types))
    width = 0.22

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (key, label) in enumerate(MODELS.items()):
        epr_summary = evals[key]["epr"]["summary"]
        rates = [epr_summary.get(mt, {}).get("preservation_rate", 0) * 100 for mt in mark_types]
        bars = ax.bar(x + i * width, rates, width, label=label, color=COLORS[label], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(mark_labels)
    ax.set_ylabel("Preservation Rate (%)")
    ax.set_title("Speech Phenomena Preservation (EPR) by Type")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, framealpha=0.9, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(FIGURES_DIR / "fig3_epr_by_mark.png")
    plt.close(fig)


def fig4_epr_overall(evals):
    """Stacked bar chart showing preserved / deleted / substituted for EPR."""
    names = list(MODELS.values())
    preserved, deleted, substituted = [], [], []
    for key in MODELS:
        ov = evals[key]["epr"]["summary"]["_overall"]
        total = ov["total"]
        preserved.append(ov["preserved"] / total * 100)
        deleted.append(ov["deleted"] / total * 100)
        substituted.append(ov["substituted"] / total * 100)

    fig, ax = plt.subplots(figsize=(6, 4))
    p = np.array(preserved)
    d = np.array(deleted)
    s = np.array(substituted)
    ax.barh(names, p, color="#2ecc71", label="Preserved", edgecolor="white")
    ax.barh(names, d, left=p, color="#e74c3c", label="Deleted", edgecolor="white")
    ax.barh(names, s, left=p + d, color="#f39c12", label="Substituted", edgecolor="white")
    ax.set_xlabel("Percentage (%)")
    ax.set_title("Overall EPR Breakdown")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        ncol=1,
        framealpha=0.9,
        fontsize=9,
    )
    ax.set_xlim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(right=0.74)
    fig.savefig(FIGURES_DIR / "fig4_epr_overall.png", bbox_inches="tight")
    plt.close(fig)


def fig5_gep_overall(evals):
    """Stacked bar chart: GEP preserved / corrected / mutated."""
    names = list(MODELS.values())
    preserved, corrected, mutated = [], [], []
    for key in MODELS:
        ov = evals[key]["gep"]["summary"]["_overall"]
        total = ov["total"]
        preserved.append(ov["preserved"] / total * 100)
        corrected.append(ov["corrected"] / total * 100)
        mutated.append(ov["mutated"] / total * 100)

    fig, ax = plt.subplots(figsize=(6, 4))
    p = np.array(preserved)
    c = np.array(corrected)
    m = np.array(mutated)
    ax.barh(names, p, color="#2ecc71", label="Preserved", edgecolor="white")
    ax.barh(names, c, left=p, color="#3498db", label="Corrected", edgecolor="white")
    ax.barh(names, m, left=p + c, color="#e74c3c", label="Mutated", edgecolor="white")
    ax.set_xlabel("Percentage (%)")
    ax.set_title("Grammatical Error Preservation (GEP)")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        ncol=1,
        framealpha=0.9,
        fontsize=9,
    )
    ax.set_xlim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(right=0.74)
    fig.savefig(FIGURES_DIR / "fig5_gep_overall.png", bbox_inches="tight")
    plt.close(fig)


def fig6_gep_by_type(evals):
    """Heatmap of GEP preservation rate for a curated selection of ~14 ERRANT error types."""
    all_types = set()
    for key in MODELS:
        all_types |= {t for t in evals[key]["gep"]["summary"] if t != "_overall"}

    type_totals = {}
    for t in all_types:
        type_totals[t] = sum(
            evals[k]["gep"]["summary"].get(t, {}).get("total", 0) for k in MODELS
        )
    valid_types = [t for t in all_types if type_totals[t] >= 120]
    if not valid_types:
        return

    model_keys = list(MODELS.keys())
    avg_pres = {}
    per_model_pres = {k: {} for k in model_keys}
    for t in valid_types:
        rates = []
        for k in model_keys:
            r = evals[k]["gep"]["summary"].get(t, {}).get("preservation_rate", 0) * 100
            per_model_pres[k][t] = r
            rates.append(r)
        avg_pres[t] = np.mean(rates)

    selected = set()

    sorted_by_avg = sorted(valid_types, key=lambda t: avg_pres[t])
    for t in sorted_by_avg[:3]:
        selected.add(t)
    for t in sorted_by_avg[-3:]:
        selected.add(t)

    for k in model_keys:
        best_delta_type, best_delta = None, -np.inf
        worst_delta_type, worst_delta = None, np.inf
        for t in valid_types:
            others = [per_model_pres[ok][t] for ok in model_keys if ok != k]
            delta = per_model_pres[k][t] - np.mean(others)
            if delta > best_delta:
                best_delta = delta
                best_delta_type = t
            if delta < worst_delta:
                worst_delta = delta
                worst_delta_type = t
        if best_delta_type:
            selected.add(best_delta_type)
        if worst_delta_type:
            selected.add(worst_delta_type)

    selected_types = sorted(selected, key=lambda t: avg_pres[t])

    matrix = np.zeros((len(selected_types), len(MODELS)))
    for j, key in enumerate(model_keys):
        for i, t in enumerate(selected_types):
            entry = evals[key]["gep"]["summary"].get(t, {})
            matrix[i, j] = entry.get("preservation_rate", 0) * 100

    fig, ax = plt.subplots(figsize=(7, max(3, len(selected_types) * 0.45 + 1.5)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect=0.6, vmin=0, vmax=100)

    ax.set_xticks(range(len(MODELS)))
    ax.xaxis.tick_top()
    ax.set_xticklabels(MODELS.values(), fontsize=10, rotation=15, ha="left")
    ax.set_yticks(range(len(selected_types)))
    ax.set_yticklabels(selected_types, fontsize=9)

    for i in range(len(selected_types)):
        for j in range(len(MODELS)):
            val = matrix[i, j]
            color = "white" if val < 40 or val > 85 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8.5, color=color)

    ax.set_title("GEP (%) by ERRANT Error Type", pad=25)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.6, pad=0.12, aspect=30)
    cbar.set_label("Preservation (%)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_gep_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def fig6b_correction_rate_by_type(evals):
    """Heatmap of raw correction rate (corrected/total) by ERRANT error type.

    Complements fig6 (preservation rate): together they disambiguate whether
    non-preserved errors are due to intentional correction or random mutation.
    """
    all_types = set()
    for key in MODELS:
        all_types |= {t for t in evals[key]["gep"]["summary"] if t != "_overall"}

    type_totals = {}
    for t in all_types:
        type_totals[t] = sum(
            evals[k]["gep"]["summary"].get(t, {}).get("total", 0) for k in MODELS
        )
    valid_types = [t for t in all_types if type_totals[t] >= 120]
    if not valid_types:
        return

    model_keys = list(MODELS.keys())
    avg_corr = {}
    per_model_corr = {k: {} for k in model_keys}
    for t in valid_types:
        rates = []
        for k in model_keys:
            entry = evals[k]["gep"]["summary"].get(t, {})
            r = entry.get("correction_rate", 0) * 100
            per_model_corr[k][t] = r
            rates.append(r)
        avg_corr[t] = np.mean(rates)

    selected = set()

    sorted_by_avg = sorted(valid_types, key=lambda t: avg_corr[t])
    for t in sorted_by_avg[:3]:
        selected.add(t)
    for t in sorted_by_avg[-3:]:
        selected.add(t)

    for k in model_keys:
        best_delta_type, best_delta = None, -np.inf
        worst_delta_type, worst_delta = None, np.inf
        for t in valid_types:
            others = [per_model_corr[ok][t] for ok in model_keys if ok != k]
            delta = per_model_corr[k][t] - np.mean(others)
            if delta > best_delta:
                best_delta = delta
                best_delta_type = t
            if delta < worst_delta:
                worst_delta = delta
                worst_delta_type = t
        if best_delta_type:
            selected.add(best_delta_type)
        if worst_delta_type:
            selected.add(worst_delta_type)

    selected_types = sorted(selected, key=lambda t: avg_corr[t])

    matrix = np.zeros((len(selected_types), len(MODELS)))
    for j, key in enumerate(model_keys):
        for i, t in enumerate(selected_types):
            entry = evals[key]["gep"]["summary"].get(t, {})
            matrix[i, j] = entry.get("correction_rate", 0) * 100

    fig, ax = plt.subplots(figsize=(7, max(3, len(selected_types) * 0.45 + 1.5)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect=0.6, vmin=0, vmax=100)

    ax.set_xticks(range(len(MODELS)))
    ax.xaxis.tick_top()
    ax.set_xticklabels(MODELS.values(), fontsize=10, rotation=15, ha="left")
    ax.set_yticks(range(len(selected_types)))
    ax.set_yticklabels(selected_types, fontsize=9)

    for i in range(len(selected_types)):
        for j in range(len(MODELS)):
            val = matrix[i, j]
            color = "white" if val < 15 or val > 75 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8.5, color=color)

    ax.set_title("Correction Rate (%) by ERRANT Error Type", pad=25)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.6, pad=0.12, aspect=30)
    cbar.set_label("Corrected / Total (%)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6b_correction_rate_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def fig7_gep_correction_rate(evals):
    """Stacked bar chart: breakdown of non-preserved errors into corrections vs mutations."""
    names = list(MODELS.values())
    corr_pcts = []
    mut_pcts = []
    for key in MODELS:
        ov = evals[key]["gep"]["summary"]["_overall"]
        changed = ov["corrected"] + ov["mutated"]
        corr_pcts.append(ov["corrected"] / changed * 100 if changed > 0 else 0)
        mut_pcts.append(ov["mutated"] / changed * 100 if changed > 0 else 0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(names))
    width = 0.55
    bars_corr = ax.bar(x, corr_pcts, width, label="Corrections", color="#3498db", edgecolor="white", linewidth=0.5)
    bars_mut = ax.bar(x, mut_pcts, width, bottom=corr_pcts, label="Mutations", color="#e74c3c", edgecolor="white", linewidth=0.5)

    for i, (c, m) in enumerate(zip(corr_pcts, mut_pcts)):
        ax.text(i, c / 2, f"{c:.1f}%", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        ax.text(i, c + m / 2, f"{m:.1f}%", ha="center", va="center", fontsize=10, fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Percentage of Non-Preserved Errors (%)")
    ax.set_title("Breakdown of Non-Preserved Errors: Corrections vs. Mutations")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(FIGURES_DIR / "fig7_gep_correction_rate.png")
    plt.close(fig)


def fig8_wer_errors_breakdown(evals):
    """Stacked bar chart of substitutions / deletions / insertions."""
    names = list(MODELS.values())
    subs, dels, ins = [], [], []
    for key in MODELS:
        w = evals[key]["wer"]
        total_errs = w["total_substitutions"] + w["total_deletions"] + w["total_insertions"]
        subs.append(w["total_substitutions"] / total_errs * 100)
        dels.append(w["total_deletions"] / total_errs * 100)
        ins.append(w["total_insertions"] / total_errs * 100)

    fig, ax = plt.subplots(figsize=(6, 4))
    s = np.array(subs)
    d = np.array(dels)
    i_ = np.array(ins)
    ax.barh(names, s, color="#e74c3c", label="Substitutions", edgecolor="white")
    ax.barh(names, d, left=s, color="#3498db", label="Deletions", edgecolor="white")
    ax.barh(names, i_, left=s + d, color="#f39c12", label="Insertions", edgecolor="white")
    ax.set_xlabel("Percentage (%)")
    ax.set_title("WER Error Composition")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        ncol=1,
        framealpha=0.9,
        fontsize=9,
    )
    ax.set_xlim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(right=0.74)
    fig.savefig(FIGURES_DIR / "fig8_wer_breakdown.png", bbox_inches="tight")
    plt.close(fig)


def fig9_tradeoff_scatter(evals):
    """Scatter: WER vs GEP preservation rate — the key trade-off."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for key, label in MODELS.items():
        wer = evals[key]["wer"]["overall_wer"] * 100
        gep = evals[key]["gep"]["summary"]["_overall"]["preservation_rate"] * 100
        ax.scatter(wer, gep, s=180, color=COLORS[label], zorder=5, edgecolors="black", linewidths=0.8)
        ax.annotate(label, (wer, gep), textcoords="offset points",
                    xytext=(10, -5), fontsize=10)

    ax.set_xlabel("WER (%)")
    ax.set_ylabel("GEP - Preservation (%)")
    ax.set_title("Trade-off: Accuracy vs. Grammatical Preservation")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    all_wers = [evals[k]["wer"]["overall_wer"] * 100 for k in MODELS]
    all_geps = [evals[k]["gep"]["summary"]["_overall"]["preservation_rate"] * 100 for k in MODELS]
    ax.set_xlim(0, max(all_wers) * 1.2)
    ax.set_ylim(min(all_geps) - 5, max(all_geps) + 5)
    ax.grid(alpha=0.3)
    fig.savefig(FIGURES_DIR / "fig9_tradeoff.png")
    plt.close(fig)


def fig10_eval_stability(eval_evals):
    """Bootstrap stability analysis: resample the eval set into K folds and show metric variation."""
    K = 10
    rng = np.random.RandomState(42)

    metric_colors = {"WER": "#e74c3c", "EPR": "#3498db", "GEP": "#2ecc71"}

    fold_results = {metric: {label: [] for label in MODELS.values()} for metric in ["WER", "EPR", "GEP"]}

    for key, label in MODELS.items():
        ev = eval_evals[key]
        wer_utt = ev["wer"]["per_utterance"]
        epr_utt = ev["epr"]["per_utterance"]
        gep_utt = ev["gep"]["per_utterance"]

        n_wer = len(wer_utt)
        indices = rng.permutation(n_wer)
        folds = np.array_split(indices, K)

        for fold_idx in folds:
            sub, dele, ins, ref_w = 0, 0, 0, 0
            for i in fold_idx:
                u = wer_utt[i]
                sub += u["substitutions"]
                dele += u["deletions"]
                ins += u["insertions"]
                ref_w += u["ref_words"]
            fold_wer = (sub + dele + ins) / ref_w * 100 if ref_w > 0 else 0
            fold_results["WER"][label].append(fold_wer)

        epr_by_id = {u["file_id"]: u for u in epr_utt}
        epr_ids_ordered = [wer_utt[i]["file_id"] for i in range(n_wer)]
        folds_epr = np.array_split(rng.permutation(len(epr_ids_ordered)), K)
        for fold_idx in folds_epr:
            pres, total = 0, 0
            for i in fold_idx:
                fid = epr_ids_ordered[i]
                if fid in epr_by_id:
                    for mark_info in epr_by_id[fid]["marks"].values():
                        pres += mark_info["preserved"]
                        total += mark_info["total"]
            fold_results["EPR"][label].append(pres / total * 100 if total > 0 else 0)

        gep_by_id = {u["file_id"]: u for u in gep_utt}
        gep_ids_ordered = [wer_utt[i]["file_id"] for i in range(n_wer)]
        folds_gep = np.array_split(rng.permutation(len(gep_ids_ordered)), K)
        for fold_idx in folds_gep:
            pres, total = 0, 0
            for i in fold_idx:
                fid = gep_ids_ordered[i]
                if fid in gep_by_id:
                    counts = gep_by_id[fid]["counts"]
                    for type_info in counts.values():
                        pres += type_info["preserved"]
                        total += type_info["total"]
            fold_results["GEP"][label].append(pres / total * 100 if total > 0 else 0)

    # One column × three rows: readable bar spacing; moderate width (not full page when embedded).
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True, constrained_layout=True)

    labels_list = list(MODELS.values())
    short_labels = [l.replace(" ", "\n") for l in labels_list]
    x = np.arange(len(labels_list))

    for mi, metric in enumerate(["WER", "EPR", "GEP"]):
        ax = axes[mi]
        data = [fold_results[metric][l] for l in labels_list]
        means = [np.mean(d) for d in data]
        stds = [np.std(d) for d in data]

        ax.bar(x, means, color=metric_colors[metric], edgecolor="white", linewidth=0.6, alpha=0.88)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#333333", capsize=5, linewidth=1.4)

        ymax = max((m + s for m, s in zip(means, stds)), default=0)
        if ymax <= 0:
            ymax = 1
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + max(0.6, ymax * 0.02), f"\u00b1{s:.1f}", ha="center", fontsize=9, fontstyle="italic")
        ax.set_ylim(0, ymax * 1.18)

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=10)
        ax.set_title(f"{metric}: mean \u00b1 SD over {K} folds", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Model", fontsize=11)
    fig.suptitle("Eval-Set Stability: 10-Fold Cross-Validation", fontsize=13, fontweight="bold")
    fig.savefig(FIGURES_DIR / "fig10_dev_vs_eval.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig11_gep_top_types(evals):
    """Grouped bar chart of GEP preservation rate for the 5 most frequent error types."""
    type_totals = {}
    for key in MODELS:
        for t, info in evals[key]["gep"]["summary"].items():
            if t == "_overall":
                continue
            type_totals[t] = type_totals.get(t, 0) + info.get("total", 0)

    top_types = [t for t, _ in sorted(type_totals.items(), key=lambda x: -x[1])[:5]]
    x = np.arange(len(top_types))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (key, label) in enumerate(MODELS.items()):
        rates = []
        for t in top_types:
            entry = evals[key]["gep"]["summary"].get(t, {})
            rates.append(entry.get("preservation_rate", 0) * 100)
        ax.bar(x + i * width, rates, width, label=label, color=COLORS[label], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"{t}\n(n={type_totals[t]})" for t in top_types], fontsize=8.5, rotation=0)
    ax.set_ylabel("GEP - Preservation (%)")
    ax.set_title("Grammatical Preservation by Error Type (Top 5 by Frequency)")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, framealpha=0.9, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    fig.savefig(FIGURES_DIR / "fig11_gep_top_types.png")
    plt.close(fig)


def fig12_gep_correction_by_category(evals):
    """Grouped bar chart comparing correction rate for Missing / Replacement / Unnecessary."""
    categories = {"Missing (M:)": "M:", "Replacement (R:)": "R:", "Unnecessary (U:)": "U:"}
    x = np.arange(len(categories))
    width = 0.22

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (key, label) in enumerate(MODELS.items()):
        rates = []
        for cat_name, prefix in categories.items():
            total = corrected = 0
            for t, info in evals[key]["gep"]["summary"].items():
                if t.startswith(prefix):
                    total += info.get("total", 0)
                    corrected += info.get("corrected", 0)
            rates.append(corrected / total * 100 if total > 0 else 0)
        bars = ax.bar(x + i * width, rates, width, label=label, color=COLORS[label], edgecolor="white")
        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(categories.keys(), fontsize=10)
    ax.set_ylabel("Correction Rate (%)")
    ax.set_title("Correction by Grammatical Error Category")
    ax.set_ylim(0, max(40, 10))
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(FIGURES_DIR / "fig12_gep_correction_category.png")
    plt.close(fig)


def fig13_finetune_training_curves():
    """Training loss and val WER curves for both models."""
    whisper_log = json.load(open(RESULTS_DIR.parent / "checkpoints" / "whisper-small-lora" / "train_log.json"))
    parakeet_log = json.load(open(RESULTS_DIR.parent / "checkpoints" / "parakeet-ctc-lora" / "train_log.json"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Whisper
    wlog = whisper_log["log"]
    epochs_w = [e["epoch"] for e in wlog]
    wers_w = [e["val_wer"] * 100 for e in wlog]
    losses_w = [e["loss"] for e in wlog if e["loss"] is not None]
    loss_epochs_w = [e["epoch"] for e in wlog if e["loss"] is not None]

    ax1.plot(epochs_w, wers_w, "o-", color="#4C72B0", label="Val WER", linewidth=2, markersize=6)
    ax1.axhline(y=wers_w[0], color="#4C72B0", linestyle="--", alpha=0.4, label=f"Baseline ({wers_w[0]:.1f}%)")
    ax1.axvline(x=2, color="green", linestyle=":", alpha=0.6, label="Best checkpoint (ep. 2)")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(loss_epochs_w, losses_w, "s--", color="#DD8452", label="Train Loss", linewidth=1.5, markersize=5, alpha=0.7)
    ax1_twin.set_ylabel("Loss", color="#DD8452")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val WER (%)")
    ax1.set_title("Whisper Small + LoRA")
    ax1.legend(loc="upper left", fontsize=8)
    ax1_twin.legend(loc="upper right", fontsize=8)
    ax1.spines["top"].set_visible(False)

    # Parakeet
    plog = parakeet_log["log"]
    epochs_p = [e["epoch"] for e in plog]
    wers_p = [e["val_wer"] * 100 for e in plog]
    losses_p = [e["loss"] for e in plog if e["loss"] is not None]
    loss_epochs_p = [e["epoch"] for e in plog if e["loss"] is not None]

    ax2.plot(epochs_p, wers_p, "o-", color="#C44E52", label="Val WER", linewidth=2, markersize=6)
    ax2.axhline(y=wers_p[0], color="#C44E52", linestyle="--", alpha=0.4, label=f"Baseline ({wers_p[0]:.1f}%)")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(loss_epochs_p, losses_p, "s--", color="#DD8452", label="Train Loss", linewidth=1.5, markersize=5, alpha=0.7)
    ax2_twin.set_ylabel("Loss", color="#DD8452")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val WER (%)")
    ax2.set_title("Parakeet CTC 1.1B + LoRA")
    ax2.legend(loc="upper left", fontsize=8)
    ax2_twin.legend(loc="upper right", fontsize=8)
    ax2.spines["top"].set_visible(False)

    fig.suptitle("LoRA Training Curves", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig13_finetune_curves.png")
    plt.close(fig)


def fig14_finetune_comparison(test_evals):
    """Grouped bar chart comparing baseline vs fine-tuned Whisper on test set."""
    models_ft = {
        "whisper-small-en": "Whisper Small\n(base)",
        "whisper-small-en-lora": "Whisper Small\n+ LoRA",
        "parakeet-ctc-1.1b": "Parakeet CTC\n(base)",
    }
    colors_ft = ["#4C72B0", "#7BAFD4", "#C44E52"]

    metrics = {
        "WER (%)": lambda e: e["wer"]["overall_wer"] * 100,
        "GEP (%)": lambda e: e["gep"]["summary"]["_overall"]["preservation_rate"] * 100,
        "Corr (%)": lambda e: e["gep"]["summary"]["_overall"]["correction_rate"] * 100,
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for mi, (metric_name, fn) in enumerate(metrics.items()):
        ax = axes[mi]
        names = list(models_ft.values())
        vals = [fn(test_evals[k]) for k in models_ft]
        bars = ax.bar(names, vals, color=colors_ft, width=0.55, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax.set_title(metric_name, fontsize=13)
        ax.set_ylim(0, max(vals) * 1.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Impact of LoRA Fine-Tuning (Test Subset, 2568 utt)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig14_finetune_comparison.png")
    plt.close(fig)


def fig15_tradeoff_with_ft(test_evals):
    """Scatter: WER vs GEP including fine-tuned model."""
    all_models = {
        "whisper-small-en": "Whisper Small",
        "whisper-medium-en": "Whisper Medium",
        "wav2vec2-large-960h": "Wav2Vec2 Large",
        "parakeet-ctc-1.1b": "Parakeet CTC 1.1B",
        "whisper-small-en-lora": "Whisper Small + LoRA",
    }
    colors_all = {
        "Whisper Small": "#4C72B0",
        "Whisper Medium": "#DD8452",
        "Wav2Vec2 Large": "#55A868",
        "Parakeet CTC 1.1B": "#C44E52",
        "Whisper Small + LoRA": "#7BAFD4",
    }

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for key, label in all_models.items():
        if key not in test_evals:
            continue
        wer = test_evals[key]["wer"]["overall_wer"] * 100
        gep = test_evals[key]["gep"]["summary"]["_overall"]["preservation_rate"] * 100
        marker = "D" if "LoRA" in label else "o"
        ax.scatter(wer, gep, s=200, color=colors_all[label], zorder=5,
                   edgecolors="black", linewidths=0.8, marker=marker)
        offset = (10, -5) if "LoRA" not in label else (10, 8)
        ax.annotate(label, (wer, gep), textcoords="offset points",
                    xytext=offset, fontsize=9.5)

    ax.set_xlabel("WER (%)")
    ax.set_ylabel("GEP - Preservation (%)")
    ax.set_title("WER vs GEP Trade-off (including Fine-Tuning)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)
    fig.savefig(FIGURES_DIR / "fig15_tradeoff_with_ft.png")
    plt.close(fig)


def fig16_all_experiments_comparison(test_evals):
    """Comprehensive bar chart: WER + GEP for all experiments."""
    models_order = [
        ("whisper-small-en", "Whisper Small\n(baseline)"),
        ("whisper-small-en-lora", "LoRA v1\n(raw verbatim)"),
        ("whisper-small-en-lora-v2", "LoRA v2\n(normalized)"),
        ("whisper-small-en-lora-v2-eos", "LoRA v2\n+ EOS penalty"),
        ("whisper-small-en-lora-gep", "LoRA\nGEP-masked"),
        ("whisper-small-en-full-gep", "Full FT\nGEP-weighted"),
    ]
    available = [(k, l) for k, l in models_order if k in test_evals]
    if not available:
        return

    keys, labels = zip(*available)
    wers = [test_evals[k]["wer"]["overall_wer"] * 100 for k in keys]
    geps = [test_evals[k]["gep"]["summary"]["_overall"]["preservation_rate"] * 100 for k in keys]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width / 2, wers, width, label="WER (%)", color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, geps, width, label="GEP (%)", color="#2ecc71", alpha=0.85, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars1, wers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar, val in zip(bars2, geps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Comparison of All Fine-Tuning Experiments")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, max(max(wers), max(geps)) * 1.2)
    ax.axhline(y=wers[0], color="#e74c3c", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.axhline(y=geps[0], color="#2ecc71", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(FIGURES_DIR / "fig16_all_experiments.png")
    plt.close(fig)


def fig17_deletion_analysis(test_evals):
    """Bar chart showing deletion counts across all experiments."""
    models_order = [
        ("whisper-small-en", "Baseline"),
        ("whisper-small-en-lora", "LoRA v1"),
        ("whisper-small-en-lora-v2", "LoRA v2"),
        ("whisper-small-en-lora-v2-eos", "LoRA v2\n+EOS"),
        ("whisper-small-en-lora-gep", "LoRA\nGEP-mask"),
        ("whisper-small-en-full-gep", "Full FT\nGEP-wt"),
    ]
    available = [(k, l) for k, l in models_order if k in test_evals]
    if not available:
        return

    keys, labels = zip(*available)
    subs = [test_evals[k]["wer"]["total_substitutions"] for k in keys]
    dels = [test_evals[k]["wer"]["total_deletions"] for k in keys]
    ins = [test_evals[k]["wer"]["total_insertions"] for k in keys]

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, subs, width, label="Substitutions", color="#e74c3c", edgecolor="black", linewidth=0.5)
    ax.bar(x, dels, width, label="Deletions", color="#3498db", edgecolor="black", linewidth=0.5)
    ax.bar(x + width, ins, width, label="Insertions", color="#f39c12", edgecolor="black", linewidth=0.5)

    for i, (s, d, n) in enumerate(zip(subs, dels, ins)):
        ax.text(i, d + 300, f"{d:,}", ha="center", fontsize=8, color="#3498db", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Number of Errors")
    ax.set_title("WER Error Composition: All Experiments")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(FIGURES_DIR / "fig17_deletion_analysis.png")
    plt.close(fig)


def fig18_tradeoff_all_experiments(test_evals):
    """WER vs GEP scatter including ALL fine-tuned experiments."""
    all_models = {
        "whisper-small-en": ("Whisper Small (base)", "o", "#4C72B0"),
        "whisper-medium-en": ("Whisper Medium (base)", "o", "#DD8452"),
        "wav2vec2-large-960h": ("Wav2Vec2 Large", "o", "#55A868"),
        "parakeet-ctc-1.1b": ("Parakeet CTC 1.1B", "o", "#C44E52"),
        "whisper-small-en-lora": ("LoRA v1 (raw)", "D", "#7BAFD4"),
        "whisper-small-en-lora-v2": ("LoRA v2 (norm)", "s", "#9B59B6"),
        "whisper-small-en-lora-v2-eos": ("LoRA v2+EOS", "^", "#1ABC9C"),
        "whisper-small-en-lora-gep": ("LoRA GEP-mask", "P", "#E67E22"),
        "whisper-small-en-full-gep": ("Full FT GEP-wt", "*", "#2C3E50"),
    }

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for key, (label, marker, color) in all_models.items():
        if key not in test_evals:
            continue
        wer = test_evals[key]["wer"]["overall_wer"] * 100
        gep = test_evals[key]["gep"]["summary"]["_overall"]["preservation_rate"] * 100
        ax.scatter(wer, gep, s=200, color=color, zorder=5,
                   edgecolors="black", linewidths=0.8, marker=marker)
        xyoff = (8, -8) if "LoRA" not in label and "Full" not in label else (8, 6)
        ax.annotate(label, (wer, gep), textcoords="offset points",
                    xytext=xyoff, fontsize=8.5)

    ax.set_xlabel("WER (%)")
    ax.set_ylabel("GEP - Preservation (%)")
    ax.set_title("WER vs GEP Trade-off: All Experiments")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)
    fig.savefig(FIGURES_DIR / "fig18_tradeoff_all.png")
    plt.close(fig)


def fig19_full_gep_training_curve():
    """Training curve for the full fine-tuning GEP-weighted experiment."""
    log_path = RESULTS_DIR.parent / "checkpoints" / "whisper-small-full-gep" / "train_log.json"
    if not log_path.exists():
        return
    with open(log_path) as f:
        raw = f.read()
    import re
    data = json.loads(raw)
    log = data["log"]

    epochs = [e["epoch"] for e in log]
    wers = [e["val_wer"] * 100 for e in log]
    losses = [e["loss"] for e in log if e["loss"] is not None]
    loss_epochs = [e["epoch"] for e in log if e["loss"] is not None]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, wers, "o-", color="#2C3E50", label="Val WER", linewidth=2, markersize=6)
    ax.axhline(y=wers[0], color="#2C3E50", linestyle="--", alpha=0.4, label=f"Baseline ({wers[0]:.1f}%)")
    ax_twin = ax.twinx()
    ax_twin.plot(loss_epochs, losses, "s--", color="#E67E22", label="Train Loss", linewidth=1.5, markersize=5, alpha=0.7)
    ax_twin.set_ylabel("Loss (weighted)", color="#E67E22")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val WER (%)")
    ax.set_title("Full Fine-Tuning: Decoder with GEP Weighting\n(frozen encoder, LR=2e-7, error_weight=2.0)")
    ax.legend(loc="upper left", fontsize=8)
    ax_twin.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    fig.savefig(FIGURES_DIR / "fig19_full_gep_curve.png")
    plt.close(fig)


if __name__ == "__main__":
    eval_evals = {k: load_eval(k, "eval") for k in MODELS}
    evals = eval_evals

    # Test-subset evaluations (all experiments)
    test_model_keys = (
        list(MODELS.keys())
        + ["whisper-small-en-lora", "whisper-small-en-lora-v2",
           "whisper-small-en-lora-v2-eos", "whisper-small-en-lora-gep",
           "whisper-small-en-full-gep"]
    )
    test_evals = {}
    for k in test_model_keys:
        p = RESULTS_DIR / k / "eval-test-eval.json"
        if p.exists():
            test_evals[k] = json.load(open(p, encoding="utf-8"))

    print("Generating figures from eval set (3209 utterances)...")
    fig1_wer_comparison(evals)
    print("  fig1_wer_comparison.png")
    fig2_wer_per_utterance(evals)
    print("  fig2_wer_distribution.png")
    fig3_epr_by_mark(evals)
    print("  fig3_epr_by_mark.png")
    fig4_epr_overall(evals)
    print("  fig4_epr_overall.png")
    fig5_gep_overall(evals)
    print("  fig5_gep_overall.png")
    fig6_gep_by_type(evals)
    print("  fig6_gep_heatmap.png")
    fig6b_correction_rate_by_type(evals)
    print("  fig6b_correction_rate_heatmap.png")
    fig7_gep_correction_rate(evals)
    print("  fig7_gep_correction_rate.png")
    fig8_wer_errors_breakdown(evals)
    print("  fig8_wer_breakdown.png")
    fig9_tradeoff_scatter(evals)
    print("  fig9_tradeoff.png")
    fig10_eval_stability(eval_evals)
    print("  fig10_dev_vs_eval.png")
    fig11_gep_top_types(evals)
    print("  fig11_gep_top_types.png")
    fig12_gep_correction_by_category(evals)
    print("  fig12_gep_correction_category.png")

    print("\nGenerating fine-tuning figures...")
    fig13_finetune_training_curves()
    print("  fig13_finetune_curves.png")
    if "whisper-small-en-lora" in test_evals:
        fig14_finetune_comparison(test_evals)
        print("  fig14_finetune_comparison.png")
        fig15_tradeoff_with_ft(test_evals)
        print("  fig15_tradeoff_with_ft.png")

    print("\nGenerating advanced experiment figures...")
    fig16_all_experiments_comparison(test_evals)
    print("  fig16_all_experiments.png")
    fig17_deletion_analysis(test_evals)
    print("  fig17_deletion_analysis.png")
    fig18_tradeoff_all_experiments(test_evals)
    print("  fig18_tradeoff_all.png")
    fig19_full_gep_training_curve()
    print("  fig19_full_gep_curve.png")

    print("Done! All figures saved to", FIGURES_DIR)
