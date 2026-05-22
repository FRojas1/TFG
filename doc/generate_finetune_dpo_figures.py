"""Figures for the DPO fine-tuning chapter of the TFG document.

Generates 8 figures, all under ``doc/figures/`` with the ``fig_ft_`` prefix:

    fig_ft_pipeline.png              - data + training pipeline diagram
    fig_ft_pair_example.png          - example chosen/rejected pair with diff
    fig_ft_wer_gep.png               - WER/GEP/Corr bar comparison base vs DPO
    fig_ft_tradeoff.png              - WER vs GEP scatter with DPO point added
    fig_ft_gep_by_category.png       - GEP delta (DPO - base) by ERRANT category
    fig_ft_correction_by_category.png- correction-rate delta by ERRANT category
    fig_ft_examples.png              - 3 qualitative utterances
    fig_ft_training_curves.png       - epoch / loss / val WER / val GEP curves

Each generator is its own function; ``main`` calls all of them. Falls back
gracefully if a results file is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Libertinus Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

ROOT = Path(__file__).parent.parent
FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR = ROOT / "results"
CHECKPOINTS_DIR = ROOT / "checkpoints"

BASE_KEY = "whisper-small-en"
DPO_KEY = "whisper-small-dpo-3epochs"

COLOR_BASE = "#2563EB"          # blue, matches generate_figures.py Whisper Small
COLOR_DPO = "#7C3AED"           # violet, distinct from existing baselines
COLOR_CORRECT = "#3498db"
COLOR_PRESERVE = "#2ecc71"
COLOR_MUTATE = "#e74c3c"


def _load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_base_eval():
    return _load_json(RESULTS_DIR / BASE_KEY / "eval-eval.json")


def _load_dpo_eval():
    return _load_json(RESULTS_DIR / DPO_KEY / "eval-fair-eval.json")


# ----------------------------------------------------------------------
# Figure 1 - DPO data and training pipeline
# ----------------------------------------------------------------------

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    # Box-row x-centres run 2.2 .. 17.8 (mean = 10). Keep the axes midpoint at
    # 10 too so the title (anchored to the axes centre) lines up with the box
    # row after `savefig bbox=tight` trims the empty margins.
    ax.set_xlim(-1, 21)
    # Pad below 0 so the rounded bottom row (centred at y=1.0, height 2.0) and
    # its outer stroke aren't clipped by the axes before `savefig bbox=tight`
    # trims the canvas.
    ax.set_ylim(-0.3, 7.5)
    ax.axis("off")

    box_w, box_h = 3.2, 2.0
    y_top = 5.5
    y_bot = 1.0

    stages_top = [
        ("train.json\n3,883 utts\n(Whisper Small\nhypotheses)",      2.2, "#dbeafe", "#1e40af"),
        ("ERRANT filter\nutts where base\ncorrected a\nwhitelisted error",
                                                                     7.4, "#dbeafe", "#1e40af"),
        ("1,728 candidates",                                         12.6, "#fef3c7", "#92400e"),
        ("Gemini 3 Flash\nminimum-edit\nrestoration",                17.8, "#dbeafe", "#1e40af"),
    ]
    # Bottom row is laid out in REVERSE so the flow forms an S-shape and
    # the connector arrow drops straight down from the top-right rather
    # than crossing the whole figure.
    stages_bot = [
        ("whisper-small-\ndpo-3epochs",                              2.2,  "#fdf4ff", "#5b21b6"),
        ("LoRA-DPO  3 epochs\nr=16, all-linear, bf16\nlr 5e-6, beta 0.1",
                                                                     7.4,  "#ede9fe", "#5b21b6"),
        ("1,274 DPO pairs\nchosen / rejected",                       12.6, "#dcfce7", "#14532d"),
        ("ERRANT\nverification\n(targeted edit\nflipped back)",      17.8, "#dbeafe", "#1e40af"),
    ]

    def draw_row(stages, y):
        for label, x_center, face, edge in stages:
            rect = patches.FancyBboxPatch(
                (x_center - box_w / 2, y - box_h / 2),
                box_w, box_h,
                boxstyle="round,pad=0.05,rounding_size=0.22",
                linewidth=1.4, edgecolor=edge, facecolor=face,
            )
            ax.add_patch(rect)
            ax.text(x_center, y, label,
                    ha="center", va="center", fontsize=10, color=edge)

    draw_row(stages_top, y_top)
    draw_row(stages_bot, y_bot)

    arrow_kw = dict(arrowstyle="->", lw=1.7, color="#374151")

    for i in range(len(stages_top) - 1):
        x_from = stages_top[i][1] + box_w / 2
        x_to = stages_top[i + 1][1] - box_w / 2
        ax.annotate("", xy=(x_to, y_top), xytext=(x_from, y_top),
                    arrowprops=arrow_kw)
    # Bottom row arrows go right-to-left because the bottom row is reversed
    for i in range(len(stages_bot) - 1):
        x_from = stages_bot[i + 1][1] - box_w / 2
        x_to = stages_bot[i][1] + box_w / 2
        ax.annotate("", xy=(x_to, y_bot), xytext=(x_from, y_bot),
                    arrowprops=arrow_kw)
    # Vertical connector: top-right (Gemini) -> bottom-right (ERRANT verif)
    ax.annotate("",
                xy=(stages_bot[-1][1], y_bot + box_h / 2),
                xytext=(stages_top[-1][1], y_top - box_h / 2),
                arrowprops=arrow_kw)

    ax.set_title("DPO data construction and training pipeline", pad=12)
    fig.savefig(FIGURES_DIR / "fig_ft_pipeline.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 2 - example chosen/rejected pair
# ----------------------------------------------------------------------

def fig_pair_example():
    import textwrap

    fid = "SI1267-00785-P50021"
    pairs_path = ROOT / "info/sandi-corpus-2025/reference-materials/dpo-pool/finetune-train-dpo.json"
    pairs = _load_json(pairs_path)
    pair = pairs[fid]

    wrap_width = 95
    rejected_lines = textwrap.wrap(pair["rejected"], width=wrap_width)
    chosen_lines = textwrap.wrap(pair["chosen"], width=wrap_width)

    n_rej_lines = len(rejected_lines)
    n_cho_lines = len(chosen_lines)
    total_text_rows = n_rej_lines + n_cho_lines + 3 + 2  # +headers, +blank, +footer
    fig_h = 0.45 * total_text_rows + 0.8

    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Compute vertical anchor for each row.
    rows = []
    rows.append(("title", f"Utterance {fid}", None, None, None))
    rows.append(("header", "rejected  (raw Whisper Small, error auto-corrected)",
                 "#b91c1c", None, None))
    for line in rejected_lines:
        rows.append(("text", line, None, "role", "#fecaca"))
    rows.append(("blank", "", None, None, None))
    rows.append(("header", "chosen  (LLM-restored, error preserved as in reference)",
                 "#15803d", None, None))
    for line in chosen_lines:
        rows.append(("text", line, None, "roles", "#bbf7d0"))
    rows.append(("blank", "", None, None, None))
    rows.append(("footer",
                 'ERRANT edit:  R:NOUN:NUM   "role"  ->  "roles"   '
                 '(target restored exactly once)',
                 None, None, None))

    n_rows = len(rows)
    margin = 0.04
    span = 1.0 - 2 * margin
    step = span / max(1, n_rows - 1)
    for i, (kind, text, color, target, hl_color) in enumerate(rows):
        y = 1.0 - margin - i * step
        if kind == "title":
            ax.text(0.5, y, text, ha="center", va="center",
                    fontsize=11, fontweight="bold", color="#374151")
        elif kind == "header":
            ax.text(0.02, y, text, ha="left", va="center",
                    fontsize=10.5, fontweight="bold", color=color)
        elif kind == "blank":
            continue
        elif kind == "footer":
            ax.text(0.5, y, text, ha="center", va="center",
                    fontsize=10, style="italic", color="#374151")
        else:
            _draw_text_line(ax, 0.04, y, text, target, hl_color, fontsize=10.5)

    fig.savefig(FIGURES_DIR / "fig_ft_pair_example.png", bbox_inches="tight")
    plt.close(fig)


def _draw_text_line(ax, x, y, text, target_word, hl_color, fontsize=11):
    """Draw a single line of text, highlighting one occurrence of target_word
    by splitting into three text() calls so the highlight aligns with the word.
    Relies on matplotlib's text bbox to know widths -- requires a draw() pass.
    """
    import re

    if target_word is None:
        ax.text(x, y, text, ha="left", va="center",
                fontsize=fontsize, color="#1f2937")
        return

    pattern = re.compile(rf"\b{re.escape(target_word)}\b")
    m = pattern.search(text)
    if not m:
        ax.text(x, y, text, ha="left", va="center",
                fontsize=fontsize, color="#1f2937")
        return

    before = text[: m.start()]
    word = text[m.start(): m.end()]
    after = text[m.end():]

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_inv = ax.transData.inverted()
    ax_to_data = lambda px: ax_inv.transform(px)

    # Place "before" first
    t_before = ax.text(x, y, before, ha="left", va="center",
                       fontsize=fontsize, color="#1f2937")
    if before.strip():
        bbox_before = t_before.get_window_extent(renderer=renderer)
        # Use ax coords (axes fraction): since we've set xlim/ylim 0..1, data == axes
        end_before_data = ax_to_data((bbox_before.x1, bbox_before.y0))[0]
        word_start = end_before_data
    else:
        word_start = x

    t_word = ax.text(word_start, y, word, ha="left", va="center",
                     fontsize=fontsize, color="#1f2937",
                     bbox=dict(boxstyle="round,pad=0.18",
                               facecolor=hl_color, edgecolor="none"))
    bbox_word = t_word.get_window_extent(renderer=renderer)
    end_word_data = ax_to_data((bbox_word.x1, bbox_word.y0))[0]
    ax.text(end_word_data + 0.005, y, after, ha="left", va="center",
            fontsize=fontsize, color="#1f2937")


# ----------------------------------------------------------------------
# Figure 3 - WER / GEP / Correction% bar comparison
# ----------------------------------------------------------------------

def fig_wer_gep():
    base = _load_base_eval()
    dpo = _load_dpo_eval()

    metrics = [
        ("WER (%)",          base["wer"]["overall_wer"] * 100,
                             dpo["wer"]["overall_wer"] * 100,          False),
        ("GEP (%)",          base["gep"]["summary"]["_overall"]["preservation_rate"] * 100,
                             dpo["gep"]["summary"]["_overall"]["preservation_rate"] * 100, True),
        ("Correction (%)",   base["gep"]["summary"]["_overall"]["correction_rate"] * 100,
                             dpo["gep"]["summary"]["_overall"]["correction_rate"] * 100,   False),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(metrics))
    width = 0.36

    base_vals = [m[1] for m in metrics]
    dpo_vals = [m[2] for m in metrics]
    higher_better = [m[3] for m in metrics]

    b1 = ax.bar(x - width / 2, base_vals, width, color=COLOR_BASE,
                edgecolor="white", linewidth=0.8, label="Whisper Small (base)")
    b2 = ax.bar(x + width / 2, dpo_vals, width, color=COLOR_DPO,
                edgecolor="white", linewidth=0.8, label="DPO-3epochs")

    for bars, vals in [(b1, base_vals), (b2, dpo_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=10)

    for xi, (b, d, hb) in enumerate(zip(base_vals, dpo_vals, higher_better)):
        delta = d - b
        good = (delta > 0) if hb else (delta < 0)
        sign = "+" if delta >= 0 else ""
        col = "#15803d" if good else "#b91c1c"
        ax.text(xi, max(b, d) + 6, f"Δ {sign}{delta:.1f} pp",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=col)

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylabel("Value (%)")
    ax.set_title("Whisper Small vs DPO-3epochs on the eval partition (3,209 utt)")
    ax.set_ylim(0, max(max(base_vals), max(dpo_vals)) * 1.4)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(FIGURES_DIR / "fig_ft_wer_gep.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 4 - WER vs GEP trade-off scatter (baselines + DPO)
# ----------------------------------------------------------------------

def fig_tradeoff():
    baselines = {
        "Whisper Small":     ("whisper-small-en",       "#2563EB"),
        "Whisper Medium":    ("whisper-medium-en",      "#FFB71F"),
        "Wav2Vec2 Large":    ("wav2vec2-large-960h",    "#059669"),
        "Parakeet CTC 1.1B": ("parakeet-ctc-1.1b",      "#C82333"),
    }

    fig, ax = plt.subplots(figsize=(6.5, 5))

    base_wer = base_gep = None
    pts = []
    for label, (key, color) in baselines.items():
        d = _load_json(RESULTS_DIR / key / "eval-eval.json")
        wer = d["wer"]["overall_wer"] * 100
        gep = d["gep"]["summary"]["_overall"]["preservation_rate"] * 100
        ax.scatter(wer, gep, s=180, color=color, zorder=4,
                   edgecolors="black", linewidths=0.8)
        offset = (10, -4) if label != "Whisper Medium" else (10, 4)
        ax.annotate(label, (wer, gep), textcoords="offset points",
                    xytext=offset, fontsize=10)
        pts.append((wer, gep))
        if key == BASE_KEY:
            base_wer, base_gep = wer, gep

    dpo = _load_dpo_eval()
    dpo_wer = dpo["wer"]["overall_wer"] * 100
    dpo_gep = dpo["gep"]["summary"]["_overall"]["preservation_rate"] * 100
    ax.scatter(dpo_wer, dpo_gep, s=240, marker="*", color=COLOR_DPO,
               zorder=6, edgecolors="black", linewidths=0.8,
               label="DPO-3epochs (this work)")
    ax.annotate("DPO-3epochs", (dpo_wer, dpo_gep), textcoords="offset points",
                xytext=(10, -4), fontsize=10, fontweight="bold",
                color=COLOR_DPO)

    if base_wer is not None:
        ax.annotate(
            "",
            xy=(dpo_wer, dpo_gep),
            xytext=(base_wer, base_gep),
            arrowprops=dict(arrowstyle="->", color=COLOR_DPO, lw=1.6,
                            linestyle="--", alpha=0.8),
        )
        ax.text(base_wer - 0.6, base_gep - 1.2,
                "DPO move\n(better on both axes)",
                fontsize=9, color=COLOR_DPO, ha="right", va="top",
                style="italic")

    pts.append((dpo_wer, dpo_gep))
    wers = [p[0] for p in pts]
    geps = [p[1] for p in pts]

    ax.set_xlabel("WER (%) -- lower is better")
    ax.set_ylabel("GEP -- preservation (%)  -- higher is better")
    ax.set_title("Accuracy vs. grammatical preservation trade-off")
    ax.set_xlim(min(wers) - 3, max(wers) + 6)
    ax.set_ylim(min(geps) - 3, max(geps) + 4)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", framealpha=0.95)
    fig.savefig(FIGURES_DIR / "fig_ft_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 5 - GEP delta by ERRANT category
# ----------------------------------------------------------------------

def _category_deltas(metric="preservation_rate", min_n=80):
    base = _load_base_eval()["gep"]["summary"]
    dpo = _load_dpo_eval()["gep"]["summary"]
    rows = []
    for cat, b in base.items():
        if cat == "_overall":
            continue
        if cat not in dpo:
            continue
        d = dpo[cat]
        if b.get("total", 0) < min_n:
            continue
        rows.append({
            "cat": cat,
            "n": b["total"],
            "base": b[metric] * 100,
            "dpo": d[metric] * 100,
            "delta": (d[metric] - b[metric]) * 100,
        })
    return rows


def fig_gep_by_category():
    rows = _category_deltas(metric="preservation_rate", min_n=80)
    rows.sort(key=lambda r: r["delta"], reverse=True)
    rows = rows[:14]
    rows = list(reversed(rows))  # plot largest delta at the top

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    y = np.arange(len(rows))
    colors = [COLOR_PRESERVE if r["delta"] >= 0 else COLOR_MUTATE for r in rows]
    bars = ax.barh(y, [r["delta"] for r in rows], color=colors,
                   edgecolor="white", linewidth=0.7)
    for bar, r in zip(bars, rows):
        ax.text(bar.get_width() + (0.4 if r["delta"] >= 0 else -0.4),
                bar.get_y() + bar.get_height() / 2,
                f"{r['base']:.0f} -> {r['dpo']:.0f} (n={r['n']})",
                va="center",
                ha="left" if r["delta"] >= 0 else "right",
                fontsize=9, color="#374151")
    ax.set_yticks(y)
    ax.set_yticklabels([r["cat"] for r in rows], fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("DPO GEP - base GEP (percentage points)")
    ax.set_title("GEP improvement by ERRANT error type (top 14, n >= 80)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(FIGURES_DIR / "fig_ft_gep_by_category.png", bbox_inches="tight")
    plt.close(fig)


def fig_correction_by_category():
    rows = _category_deltas(metric="correction_rate", min_n=80)
    rows.sort(key=lambda r: r["delta"])  # most negative delta first (best correction-rate drop)
    rows = rows[:14]
    rows = list(reversed(rows))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    y = np.arange(len(rows))
    colors = [COLOR_PRESERVE if r["delta"] <= 0 else COLOR_MUTATE for r in rows]
    bars = ax.barh(y, [r["delta"] for r in rows], color=colors,
                   edgecolor="white", linewidth=0.7)
    for bar, r in zip(bars, rows):
        ax.text(bar.get_width() + (0.4 if r["delta"] <= 0 else -0.4),
                bar.get_y() + bar.get_height() / 2,
                f"{r['base']:.0f} -> {r['dpo']:.0f} (n={r['n']})",
                va="center",
                ha="left" if r["delta"] <= 0 else "right",
                fontsize=9, color="#374151")
    ax.set_yticks(y)
    ax.set_yticklabels([r["cat"] for r in rows], fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    deltas = [r["delta"] for r in rows]
    ax.set_xlim(min(deltas) - 1.5, 5)
    ax.set_xlabel("DPO correction% - base correction% (pp, lower is better)")
    ax.set_title("Correction-rate reduction by ERRANT error type (top 14, n >= 80)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(FIGURES_DIR / "fig_ft_correction_by_category.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 7 - qualitative examples
# ----------------------------------------------------------------------

QUAL_EXAMPLES = [
    {
        "fid": "SI114J-00762-P50019",
        "type": "R:VERB:SVA",
        "ref": "a team leader is a person who work hard",
        "base": "A team leader is a person who works hard, it should be kind, special, gorgeous and good working.",
        "dpo":  "A team leader is a person who work hard, it should be kind, special, gorgeous, good working.",
        "highlight_base": "works",
        "highlight_dpo":  "work",
    },
    {
        "fid": "SI127E-00368-P10008",
        "type": "R:VERB:TENSE",
        "ref": "maybe i will like to to improve my writing skills and speaking skills",
        "base": "Maybe I would like to improve my writing skills and speaking skills.",
        "dpo":  "Maybe I will like to improve my writing skills and speaking skills.",
        "highlight_base": "would",
        "highlight_dpo":  "will",
    },
    {
        "fid": "SI127E-00368-P10005",
        "type": "R:VERB:FORM",
        "ref": "i'm interested in in learn about culture music and shows and everything",
        "base": "I'm interested in learning about culture, music and shows and everything.",
        "dpo":  "I'm interested in learn about culture, music and shows and everything.",
        "highlight_base": "learning",
        "highlight_dpo":  "learn",
    },
]


def fig_examples():
    fig, axes = plt.subplots(len(QUAL_EXAMPLES), 1,
                             figsize=(11, 2.2 * len(QUAL_EXAMPLES) + 0.6))
    if len(QUAL_EXAMPLES) == 1:
        axes = [axes]
    for ax, ex in zip(axes, QUAL_EXAMPLES):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.005, 0.93, f"{ex['fid']}   ({ex['type']})",
                ha="left", va="top", fontsize=9.5, color="#6b7280")
        ax.text(0.005, 0.74,
                f"reference  (with error):  {ex['ref']}",
                ha="left", va="center", fontsize=10.5, color="#1f2937",
                fontstyle="italic")
        ax.text(0.005, 0.44, "base", ha="left", va="center",
                fontsize=10.5, fontweight="bold", color="#b91c1c")
        _draw_text_line(ax, 0.07, 0.44, ex["base"],
                        ex["highlight_base"], "#fecaca", fontsize=10.5)
        ax.text(0.005, 0.16, "DPO", ha="left", va="center",
                fontsize=10.5, fontweight="bold", color="#15803d")
        _draw_text_line(ax, 0.07, 0.16, ex["dpo"],
                        ex["highlight_dpo"], "#bbf7d0", fontsize=10.5)

    axes[0].set_title("Qualitative examples: utterances where DPO restored the learner's error",
                      pad=10, fontsize=12)
    fig.savefig(FIGURES_DIR / "fig_ft_examples.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 8 - training curves
# ----------------------------------------------------------------------

def _find_train_log():
    candidates = [
        CHECKPOINTS_DIR / "whisper-small-dpo-3epochs" / "train_log_clean.json",
        CHECKPOINTS_DIR / "whisper-small-dpo-3epochs-logsrun" / "train_log.json",
        CHECKPOINTS_DIR / "whisper-small-dpo-3epochs" / "train_log.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def fig_training_curves():
    log_path = _find_train_log()
    if log_path is None:
        print("  fig_ft_training_curves: no train log available yet, skipping")
        return
    data = _load_json(log_path)
    log = data["log"]
    epochs = [r["epoch"] for r in log]
    wer = [r.get("wer") for r in log]
    gep = [r.get("gep") for r in log]
    loss = [r.get("loss") for r in log]
    margin = [r.get("reward_margin") for r in log]

    fig, (ax_main, ax_aux) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_main.set_xlabel("Epoch")
    ax_main.set_ylabel("Validation WER (%)", color=COLOR_BASE)
    ax_main.plot(epochs, [w * 100 if w is not None else None for w in wer],
                 "o-", color=COLOR_BASE, label="val WER", linewidth=2)
    ax_main.tick_params(axis="y", labelcolor=COLOR_BASE)
    ax_main.grid(alpha=0.3)
    ax_main.spines["top"].set_visible(False)

    ax_gep = ax_main.twinx()
    ax_gep.set_ylabel("Validation GEP (%)", color=COLOR_PRESERVE)
    ax_gep.plot(epochs, [g * 100 if g is not None else None for g in gep],
                "s-", color=COLOR_PRESERVE, label="val GEP", linewidth=2)
    ax_gep.tick_params(axis="y", labelcolor=COLOR_PRESERVE)
    ax_gep.spines["top"].set_visible(False)
    ax_main.set_title("Validation metrics across epochs")
    ax_main.set_xticks(epochs)

    # second panel: loss + reward margin (skip epoch 0 which has no loss)
    train_epochs = [e for e, l in zip(epochs, loss) if l is not None]
    train_loss = [l for l in loss if l is not None]
    train_margin = [m for m, l in zip(margin, loss) if l is not None]

    ax_aux.set_xlabel("Epoch")
    ax_aux.set_ylabel("DPO loss", color="#374151")
    ax_aux.plot(train_epochs, train_loss, "o-", color="#374151",
                label="DPO loss", linewidth=2)
    ax_aux.tick_params(axis="y", labelcolor="#374151")
    ax_aux.grid(alpha=0.3)
    ax_aux.spines["top"].set_visible(False)

    ax_m = ax_aux.twinx()
    ax_m.set_ylabel("Reward margin", color=COLOR_DPO)
    ax_m.plot(train_epochs, train_margin, "^-", color=COLOR_DPO,
              label="reward margin", linewidth=2)
    ax_m.tick_params(axis="y", labelcolor=COLOR_DPO)
    ax_m.spines["top"].set_visible(False)
    ax_aux.set_title("DPO training signal")
    ax_aux.set_xticks(train_epochs)

    fig.suptitle("DPO fine-tuning: validation curves and training signal "
                 "(whisper-small-dpo-3epochs, val pool = 100)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_ft_training_curves.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    funcs = [
        fig_pipeline,
        fig_pair_example,
        fig_wer_gep,
        fig_tradeoff,
        fig_gep_by_category,
        fig_correction_by_category,
        fig_examples,
        fig_training_curves,
    ]
    for f in funcs:
        print(f"Generating {f.__name__} ...")
        f()
    print("\nDone. Figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
