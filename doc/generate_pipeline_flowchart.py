"""Generate the benchmark-methodology pipeline flowchart for the TFG document.

Single figure: eight stages of the baseline evaluation pipeline laid out as
an S-shape (top row left-to-right, vertical connector, bottom row right-to-
left). Visual language matches ``fig_ft_pipeline.png`` (chapter 8): pastel
rounded boxes with darker matching borders, colour-coded by stage type
(blue for data/process boxes, yellow for intermediate artefacts, purple for
the per-edit classification, green for the final metric output).

The fine-tuning branch is intentionally NOT shown here; chapter 8 ships its
own dedicated diagram (``fig_ft_pipeline.png``).
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Libertinus Serif"],
    "font.size": 11,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

FIGURES_DIR = Path(__file__).parent / "figures"

# Palette taken from generate_finetune_dpo_figures.fig_pipeline so the two
# pipeline figures read as a matched pair.
BLUE_FACE,   BLUE_EDGE   = "#dbeafe", "#1e40af"   # data / processing
YELLOW_FACE, YELLOW_EDGE = "#fef3c7", "#92400e"   # intermediate artefact
PURPLE_FACE, PURPLE_EDGE = "#ede9fe", "#5b21b6"   # per-edit classification
GREEN_FACE,  GREEN_EDGE  = "#dcfce7", "#14532d"   # final output


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

    # Top row, ordered left -> right (stages 1..4)
    stages_top = [
        ("S&I corpus\n3,209 eval utts\nTSV + STM\ntrans + GEC JSON",
         2.2,  BLUE_FACE, BLUE_EDGE),
        ("SANDiDataset\naudio @ 16 kHz\n+ verbatim ref\n+ per-word marks",
         7.4,  BLUE_FACE, BLUE_EDGE),
        ("ASR model\ntranscribe.py\nWhisper, Wav2Vec2,\nParakeet, ...",
         12.6, BLUE_FACE, BLUE_EDGE),
        ("results/*.json\n{file_id,\nhypothesis,\nreference}",
         17.8, YELLOW_FACE, YELLOW_EDGE),
    ]
    # Bottom row laid out REVERSED so the flow forms an S-shape (the vertical
    # connector drops straight down from the top-right rather than crossing
    # the whole figure). Reading order along the data flow is:
    # stages_bot[3] (normalisation) -> [2] (ERRANT) -> [1] (classify) -> [0] (metrics).
    stages_bot = [
        ("WER + EPR + GEP\nper-utterance\n+ aggregate",
         2.2,  GREEN_FACE, GREEN_EDGE),
        ("Per-edit classify\npreserved /\ncorrected /\nmutated",
         7.4,  PURPLE_FACE, PURPLE_EDGE),
        ("ERRANT alignment\nfluent vs. GEC\n-> 39 scored types\n10,397 edits",
         12.6, BLUE_FACE, BLUE_EDGE),
        ("Normalisation\nstrip (%hesitation%),\npartials, punct,\nfiller words",
         17.8, BLUE_FACE, BLUE_EDGE),
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

    # Top row: left -> right
    for i in range(len(stages_top) - 1):
        x_from = stages_top[i][1] + box_w / 2
        x_to = stages_top[i + 1][1] - box_w / 2
        ax.annotate("", xy=(x_to, y_top), xytext=(x_from, y_top),
                    arrowprops=arrow_kw)

    # Bottom row arrows go right-to-left because the array is reversed.
    # The data flow is stages_bot[3] -> [2] -> [1] -> [0], so we draw each
    # arrow with its head pointing at stages_bot[i] (the next-in-flow box).
    for i in range(len(stages_bot) - 1):
        x_from = stages_bot[i + 1][1] - box_w / 2
        x_to = stages_bot[i][1] + box_w / 2
        ax.annotate("", xy=(x_to, y_bot), xytext=(x_from, y_bot),
                    arrowprops=arrow_kw)

    # Vertical connector: top-right (results.json) -> bottom-right (normalisation)
    ax.annotate("",
                xy=(stages_bot[-1][1], y_bot + box_h / 2),
                xytext=(stages_top[-1][1], y_top - box_h / 2),
                arrowprops=arrow_kw)

    ax.set_title("Benchmark evaluation pipeline", pad=12)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "fig_pipeline.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    fig_pipeline()


if __name__ == "__main__":
    main()
