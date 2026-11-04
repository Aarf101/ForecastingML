"""Diagramme « intermédiaire » : 16 étapes, icônes officielles AWS, fond violet.

    python3 docs/diagrams/architecture_medium.py
    # -> architecture_medium.png / .svg

Entre la version simple (7 blocs) et la version détaillée (8 étapes + notes).
Icônes : jeu officiel AWS (awslabs/aws-icons-for-plantuml), copiées dans icons/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
ICONS = OUT / "icons"

W, H = 15.2, 6.1
BW, BH = 2.10, 0.76
PX, PY = 2.44, 0.92
XS = [0.45 + BW / 2 + i * PX for i in range(6)]
YS = [5.45 - i * PY for i in range(5)]

F = "DejaVu Sans"
BG = "#1E1033"
CARD = "#341B52"
MAIN_TXT = "#FFFFFF"
SUB_TXT = "#C3AEDC"
ARROW = "#9B8AB8"

BLUE, ORANGE, GREEN, SM, VIOLET, PINK = (
    "#7EA6E0", "#F0A73C", "#5FBF72", "#6FA8F5", "#A78BFA", "#F2719E",
)

fig, ax = plt.subplots(figsize=(W, H))
fig.subplots_adjust(0, 0, 1, 1)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def node(col, row, icon, main, sub="", edge=BLUE, dashed=False):
    """Carte : pastille blanche + icône officielle + libellé court."""
    x, y = XS[col], YS[row]
    ax.add_patch(
        FancyBboxPatch(
            (x - BW / 2, y - BH / 2), BW, BH,
            boxstyle="round,pad=0.01,rounding_size=0.10",
            linewidth=1.5, edgecolor=edge, facecolor=CARD,
            linestyle=(0, (4, 2)) if dashed else "solid", zorder=3,
        )
    )
    side = 0.50
    cx, cy = x - BW / 2 + 0.12, y - side / 2
    ax.add_patch(
        FancyBboxPatch(
            (cx, cy), side, side,
            boxstyle="round,pad=0.005,rounding_size=0.09",
            linewidth=0, facecolor="#F7F5FB", zorder=4,
        )
    )
    img = mpimg.imread(ICONS / f"{icon}.png")
    pad = 0.055
    ax.imshow(
        img,
        extent=[cx + pad, cx + side - pad, cy + pad, cy + side - pad],
        zorder=5, interpolation="lanczos", aspect="auto",
    )
    ax.text(cx + side + 0.12, y + (0.13 if sub else 0), main, ha="left",
            va="center", fontsize=9.6, weight="bold", family=F,
            color=MAIN_TXT, zorder=6)
    if sub:
        ax.text(cx + side + 0.12, y - 0.15, sub, ha="left", va="center",
                fontsize=7.6, family=F, color=SUB_TXT, zorder=6)


def arrow(p1, p2, style="-", both=False):
    ls = {"-": "solid", "--": (0, (4, 2)), ":": (0, (1.6, 1.6))}[style]
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle="<|-|>" if both else "-|>",
            mutation_scale=11, linewidth=1.4, color=ARROW,
            linestyle=ls, shrinkA=2, shrinkB=2, zorder=2,
        )
    )


def v(col, row_from, row_to, style="-"):
    """Flèche verticale entre deux lignes d'une même colonne."""
    x = XS[col]
    arrow((x, YS[row_from] - BH / 2), (x, YS[row_to] + BH / 2), style)


def h(col_from, col_to, row, style="-"):
    arrow((XS[col_from] + BW / 2, YS[row]), (XS[col_to] - BW / 2, YS[row]), style)


def _ensure_source_icon() -> Path:
    """Petite icône « fichier de ventes » (aucune icône AWS pour une source)."""
    path = ICONS / "source.png"
    if path.exists():
        return path
    fig_i, ax_i = plt.subplots(figsize=(0.64, 0.64), dpi=200)
    fig_i.subplots_adjust(0, 0, 1, 1)
    ax_i.set_xlim(0, 1)
    ax_i.set_ylim(0, 1)
    ax_i.axis("off")
    ax_i.add_patch(
        FancyBboxPatch((0.14, 0.10), 0.72, 0.80,
                       boxstyle="round,pad=0.01,rounding_size=0.10",
                       linewidth=0, facecolor="#8E7CC3")
    )
    for i, y in enumerate((0.66, 0.50, 0.34)):
        ax_i.add_patch(
            FancyBboxPatch((0.26, y), 0.48 - i * 0.10, 0.07,
                           boxstyle="round,pad=0.005,rounding_size=0.03",
                           linewidth=0, facecolor="#FFFFFF")
        )
    fig_i.savefig(path, transparent=True)
    plt.close(fig_i)
    return path


_ensure_source_icon()

# --- colonne 1 : source -----------------------------------------------------
node(0, 0, "source", "Ventes", "POS · CSV", ORANGE)
node(0, 1, "source", "Historique", "M5 · Rossmann", ORANGE)

# --- colonne 2 : ingestion --------------------------------------------------
node(1, 0, "msk", "Kafka", "MSK", ORANGE)
node(1, 1, "firehose", "Firehose", "60 s", ORANGE)

# --- colonne 3 : lac de données --------------------------------------------
node(2, 0, "s3", "S3 raw", "JSON.gz", GREEN)
node(2, 1, "glue", "Glue ETL", "clean · dedupe", GREEN)
node(2, 2, "s3", "S3 curated", "Parquet", GREEN)
node(2, 3, "athena", "Athena", "KPI", GREEN, dashed=True)

# --- colonne 4 : SageMaker --------------------------------------------------
node(3, 1, "sagemaker", "Processing", "features", SM)
node(3, 2, "sagemaker", "Training", "+ HPO", SM)
node(3, 3, "sagemaker", "Registry", "modèle", SM)

# --- colonne 5 : serving ----------------------------------------------------
node(4, 1, "cloudwatch", "Model Monitor", "", SM, dashed=True)
node(4, 2, "sagemaker", "Endpoint", "serverless", VIOLET)
node(4, 3, "sagemaker", "Batch", "28 jours", VIOLET)
node(4, 4, "s3", "S3 forecast", "P10 · P50 · P90", VIOLET)

# --- colonne 6 : agent + dashboard -----------------------------------------
node(5, 2, "bedrock", "Agent LLM", "LangGraph · RAG", VIOLET)
node(5, 4, "dashboard", "Dashboard", "Streamlit", PINK)

# --- flèches ----------------------------------------------------------------
h(0, 1, 0)                       # ventes -> kafka
arrow((XS[0] + BW / 2, YS[1] + 0.10), (XS[1] - BW / 2, YS[0] - 0.10))  # backfill -> kafka
v(1, 0, 1)                       # kafka -> firehose
arrow((XS[1] + BW / 2, YS[1] + 0.06), (XS[2] - BW / 2, YS[0] - 0.06))
v(2, 0, 1)                       # s3 raw -> glue
v(2, 1, 2)                       # glue -> curated
v(2, 2, 3, style="--")           # curated -> athena (branche)
arrow((XS[2] + BW / 2, YS[2] + 0.06), (XS[3] - BW / 2, YS[1] - 0.06))
v(3, 1, 2)                       # processing -> training
v(3, 2, 3)                       # training -> registry
arrow((XS[3] + BW / 2, YS[3] + 0.06), (XS[4] - BW / 2, YS[2] - 0.06))
h(3, 4, 3)                       # registry -> batch
v(4, 2, 1, style=":")            # endpoint -> model monitor
v(4, 3, 4)                       # batch -> s3 forecast
h(4, 5, 2, style="--")           # endpoint -> agent
arrow((XS[4] + BW / 2, YS[4]), (XS[5] - BW / 2, YS[4]))  # forecast -> dashboard
arrow((XS[5], YS[2] + BH / 2), (XS[5], YS[4] - BH / 2))  # agent -> dashboard

# --- bandeau MLOps ----------------------------------------------------------
bar_y, bar_h = 0.52, 0.56
ax.add_patch(
    FancyBboxPatch(
        (0.45, bar_y), W - 0.90, bar_h,
        boxstyle="round,pad=0.01,rounding_size=0.12",
        linewidth=1.2, edgecolor="#4B3A6B", facecolor="#281640", zorder=3,
    )
)
ax.text(0.80, bar_y + bar_h / 2, "MLOps", ha="left", va="center",
        fontsize=8.6, weight="bold", family=F, color=SUB_TXT, zorder=6)
for i, icon in enumerate(("eventbridge", "stepfunctions", "cloudwatch")):
    side = 0.34
    cx = 1.55 + i * 0.50
    top = bar_y + (bar_h - side) / 2
    ax.add_patch(
        FancyBboxPatch((cx, top), side, side,
                       boxstyle="round,pad=0.004,rounding_size=0.07",
                       linewidth=0, facecolor="#F7F5FB", zorder=4)
    )
    ax.imshow(
        mpimg.imread(ICONS / f"{icon}.png"),
        extent=[cx + 0.04, cx + side - 0.04, top + 0.04, top + side - 0.04],
        zorder=5, interpolation="lanczos", aspect="auto",
    )
ax.text(3.20, bar_y + bar_h / 2,
        "EventBridge (quotidien)  →  Step Functions  ·  CloudWatch",
        ha="left", va="center", fontsize=8.4, family=F, color=SUB_TXT, zorder=6)

fig.savefig(OUT / "architecture_medium.png", dpi=190, facecolor=BG)
fig.savefig(OUT / "architecture_medium.svg", facecolor=BG)
print("rendered")

