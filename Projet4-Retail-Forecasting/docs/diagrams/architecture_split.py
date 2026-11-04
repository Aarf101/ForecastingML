"""Deux diagrammes séparés : entraînement et inférence (sans agent LLM).

    python3 docs/diagrams/architecture_split.py
    # -> architecture_training.png / .svg
    # -> architecture_inference.png / .svg

Même langage visuel que architecture_medium.py : fond violet, icônes AWS
officielles sur pastille blanche, libellés courts.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
ICONS = OUT / "icons"

F = "DejaVu Sans"
BG = "#1E1033"
CARD = "#341B52"
MAIN_TXT = "#FFFFFF"
SUB_TXT = "#C3AEDC"
ARROW = "#9B8AB8"
BAR_EDGE = "#4B3A6B"
BAR_FILL = "#281640"

ORANGE, GREEN, SM, VIOLET, PINK = (
    "#F0A73C", "#5FBF72", "#6FA8F5", "#A78BFA", "#F2719E",
)

BW, BH = 2.10, 0.76


def new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(0, 0, 1, 1)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def chip(ax, x, y, side, icon):
    """Pastille blanche + icône officielle."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y), side, side,
            boxstyle="round,pad=0.005,rounding_size=0.09",
            linewidth=0, facecolor="#F7F5FB", zorder=4,
        )
    )
    pad = 0.05
    ax.imshow(
        mpimg.imread(ICONS / f"{icon}.png"),
        extent=[x + pad, x + side - pad, y + pad, y + side - pad],
        zorder=5, interpolation="lanczos", aspect="auto",
    )


def node(ax, x, y, icon, main, sub="", edge=SM, dashed=False, w=BW, h=BH):
    """Carte centrée en (x, y)."""
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.01,rounding_size=0.10",
            linewidth=1.5, edgecolor=edge, facecolor=CARD,
            linestyle=(0, (4, 2)) if dashed else "solid", zorder=3,
        )
    )
    side = 0.50
    left = x - w / 2 + 0.12
    chip(ax, left, y - side / 2, side, icon)
    ax.text(left + side + 0.12, y + (0.13 if sub else 0), main, ha="left",
            va="center", fontsize=9.6, weight="bold", family=F,
            color=MAIN_TXT, zorder=6)
    if sub:
        ax.text(left + side + 0.12, y - 0.15, sub, ha="left", va="center",
                fontsize=7.6, family=F, color=SUB_TXT, zorder=6)


def arrow(ax, p1, p2, style="-", both=False):
    ls = {"-": "solid", "--": (0, (4, 2)), ":": (0, (1.6, 1.6))}[style]
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle="<|-|>" if both else "-|>",
            mutation_scale=11, linewidth=1.4, color=ARROW,
            linestyle=ls, shrinkA=2, shrinkB=2, zorder=2,
        )
    )


def elbow(ax, p1, p2, style="-", mid=None):
    """Connecteur orthogonal (descente, course horizontale, descente)."""
    (x1, y1), (x2, y2) = p1, p2
    ym = mid if mid is not None else (y1 + y2) / 2
    ax.plot([x1, x1], [y1, ym], color=ARROW, linewidth=1.4, zorder=2,
            solid_capstyle="round")
    ax.plot([x1, x2], [ym, ym], color=ARROW, linewidth=1.4, zorder=2,
            solid_capstyle="round")
    arrow(ax, (x2, ym), (x2, y2), style=style)


def bar(ax, w, y, h, icons, text):
    """Bandeau MLOps en bas."""
    ax.add_patch(
        FancyBboxPatch(
            (0.45, y), w - 0.90, h,
            boxstyle="round,pad=0.01,rounding_size=0.12",
            linewidth=1.2, edgecolor=BAR_EDGE, facecolor=BAR_FILL, zorder=3,
        )
    )
    ax.text(0.80, y + h / 2, "MLOps", ha="left", va="center",
            fontsize=8.6, weight="bold", family=F, color=SUB_TXT, zorder=6)
    for i, icon in enumerate(icons):
        side = 0.34
        chip(ax, 1.55 + i * 0.50, y + (h - side) / 2, side, icon)
    ax.text(1.55 + len(icons) * 0.50 + 0.25, y + h / 2, text, ha="left",
            va="center", fontsize=8.4, family=F, color=SUB_TXT, zorder=6)


def draw_training() -> Path:
    """Chaîne d'entraînement : données -> features -> modèle -> registre."""
    w, h = 15.2, 4.35
    fig, ax = new_fig(w, h)
    xs = [1.50 + i * 2.44 for i in range(6)]
    y1, y2 = 3.55, 2.12

    row_a = [
        ("source", "Données", "POS · M5"),
        ("msk", "Kafka", "MSK"),
        ("firehose", "Firehose", "60 s"),
        ("s3", "S3 raw", "JSON.gz"),
        ("glue", "Glue ETL", "clean · dedupe"),
        ("s3", "S3 curated", "Parquet"),
    ]
    for x, (icon, main, sub) in zip(xs, row_a):
        edge = GREEN if icon in {"s3", "glue"} else ORANGE
        node(ax, x, y1, icon, main, sub, edge)

    row_b = [
        ("sagemaker", "Processing", "features"),
        ("sagemaker", "Training", "+ HPO"),
        ("sagemaker", "Evaluation", "WAPE"),
        ("sagemaker", "Registry", "modèle"),
    ]
    for x, (icon, main, sub) in zip(xs, row_b):
        node(ax, x, y2, icon, main, sub, SM)

    for i in range(5):
        arrow(ax, (xs[i] + BW / 2, y1), (xs[i + 1] - BW / 2, y1))
    for i in range(3):
        arrow(ax, (xs[i] + BW / 2, y2), (xs[i + 1] - BW / 2, y2))

    elbow(ax, (xs[5], y1 - BH / 2), (xs[0], y2 + BH / 2), mid=2.84)

    bar(ax, w, 0.42, 0.52, ("eventbridge", "stepfunctions", "cloudwatch"),
        "EventBridge (réentraînement)  →  Step Functions  ·  CloudWatch")

    path = OUT / "architecture_training.png"
    fig.savefig(path, dpi=190, facecolor=BG)
    fig.savefig(OUT / "architecture_training.svg", facecolor=BG)
    plt.close(fig)
    return path


def draw_inference() -> Path:
    """Chaîne d'inférence : modèle approuvé + features -> prévisions -> dashboard."""
    w, h = 10.6, 4.0
    fig, ax = new_fig(w, h)
    xs = [1.63 + i * 2.45 for i in range(4)]
    y1, y2 = 2.95, 1.85

    node(ax, xs[0], y1, "sagemaker", "Registry", "modèle approuvé", SM)
    node(ax, xs[1], y1, "sagemaker", "Endpoint", "temps réel", VIOLET)
    node(ax, xs[3], y1, "cloudwatch", "Model Monitor", "", SM, dashed=True)

    node(ax, xs[0], y2, "s3", "S3 curated", "features", GREEN)
    node(ax, xs[1], y2, "sagemaker", "Batch", "28 jours", VIOLET)
    node(ax, xs[2], y2, "s3", "S3 forecast", "P10 · P50 · P90", VIOLET)
    node(ax, xs[3], y2, "dashboard", "Dashboard", "Streamlit", PINK)

    arrow(ax, (xs[0] + BW / 2, y1), (xs[1] - BW / 2, y1))
    arrow(ax, (xs[0] + BW / 2, y1 - 0.16), (xs[1] - BW / 2, y2 + 0.16))
    arrow(ax, (xs[0] + BW / 2, y2), (xs[1] - BW / 2, y2))
    arrow(ax, (xs[1] + BW / 2, y2), (xs[2] - BW / 2, y2))
    arrow(ax, (xs[2] + BW / 2, y2), (xs[3] - BW / 2, y2))
    arrow(ax, (xs[1] + BW / 2, y1), (xs[3] - BW / 2, y1), style=":")

    bar(ax, w, 0.45, 0.52, ("eventbridge", "cloudwatch"),
        "EventBridge (batch quotidien)  ·  CloudWatch")

    path = OUT / "architecture_inference.png"
    fig.savefig(path, dpi=190, facecolor=BG)
    fig.savefig(OUT / "architecture_inference.svg", facecolor=BG)
    plt.close(fig)
    return path


def main() -> int:
    for path in (draw_training(), draw_inference()):
        print(f"[diagram] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

