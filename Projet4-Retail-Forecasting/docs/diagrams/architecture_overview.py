"""Diagramme « vue d'ensemble » : les deux pipelines qui partagent les features.

    python3 docs/diagrams/architecture_overview.py
    # -> architecture_overview.png / .svg

Une ligne = entraînement (mensuel), une ligne = inférence (quotidien). La colonne
« Features » est volontairement unique et verticale : c'est le même code dans les
deux cas, et c'est ce qui garantit que le modèle reçoit à l'inférence les mêmes
variables qu'à l'entraînement.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent

W, H = 15.4, 5.4
BW, BH = 2.15, 0.95
Y_TOP, Y_BOT = 4.15, 1.35
XS = [1.35 + i * 2.55 for i in range(6)]

BG = "#1E1033"
CARD = "#341B52"
TXT = "#FFFFFF"
SUB = "#C3AEDC"
ARROW = "#9B8AB8"
BLUE = "#6FA8F5"
ORANGE = "#F0A73C"
GREEN = "#5FBF72"
VIOLET = "#A78BFA"
PINK = "#F2719E"

F = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(W, H))
fig.subplots_adjust(0, 0, 1, 1)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x, y, w, h, main, sub="", edge=BLUE, dashed=False, fs=10.5, subfs=8.0,
        dy_main=None, dy_sub=-0.19):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.01,rounding_size=0.10",
            linewidth=1.6, edgecolor=edge, facecolor=CARD,
            linestyle=(0, (4, 2)) if dashed else "solid", zorder=3,
        )
    )
    dy_main = (0.13 if sub else 0.0) if dy_main is None else dy_main
    ax.text(x, y + dy_main, main, ha="center", va="center", fontsize=fs,
            weight="bold", family=F, color=TXT, zorder=4)
    if sub:
        ax.text(x, y + dy_sub, sub, ha="center", va="center", fontsize=subfs,
                family=F, color=SUB, zorder=4, linespacing=1.35)


def arrow(p1, p2, style="-", dashed=False):
    ls = {"-": "solid", "--": (0, (4, 2)), ":": (0, (1.6, 1.6))}[style]
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=12, linewidth=1.5,
            color=ARROW, linestyle=ls, shrinkA=2, shrinkB=2, zorder=2,
        )
    )


def elbow(p1, p2, mid_y):
    ax.plot([p1[0], p1[0]], [p1[1], mid_y], color=ARROW, linewidth=1.5, zorder=2)
    ax.plot([p1[0], p2[0]], [mid_y, mid_y], color=ARROW, linewidth=1.5, zorder=2)
    arrow((p2[0], mid_y), p2)


# --- titres de ligne --------------------------------------------------------
ax.text(0.30, 4.95, "① ENTRAÎNEMENT — une fois par mois", fontsize=10.5,
        weight="bold", family=F, color=BLUE, va="center")
ax.text(0.30, 2.15, "② INFÉRENCE — chaque jour", fontsize=10.5,
        weight="bold", family=F, color=VIOLET, va="center")

# --- ligne entraînement -----------------------------------------------------
box(XS[0], Y_TOP, BW, BH, "Historique 2 ans", "876 000 ventes", ORANGE)
box(XS[1], Y_TOP, BW, BH, "ETL", "dédup · typage", GREEN)
box(XS[3], Y_TOP, BW, BH, "Training LightGBM", "1 modèle par horizon", BLUE)
box(XS[4], Y_TOP, BW, BH, "Modèle approuvé", "WAPE 0,43", BLUE)

# --- colonne partagée : features -------------------------------------------
box(XS[2], (Y_TOP + Y_BOT) / 2, BW, Y_TOP - Y_BOT + BH,
    "Features", "lags 1→35\nmoyennes glissantes\ncalendrier · promo\n\nmême code\ndes deux côtés",
    GREEN, fs=11.5, subfs=8.0, dy_main=1.35, dy_sub=-0.40)

# --- ligne inférence --------------------------------------------------------
box(XS[0], Y_BOT, BW, BH, "Ventes récentes", "derniers jours", ORANGE)
box(XS[3], Y_BOT, BW, BH, "Batch Transform", "4 horizons", VIOLET)
box(XS[4], Y_BOT, BW, BH, "Prévisions S3", "4 800 lignes", VIOLET)
box(XS[5], Y_BOT, BW, BH, "Dashboard", "Streamlit", PINK)

# --- flèches ----------------------------------------------------------------
arrow((XS[0] + BW / 2, Y_TOP), (XS[1] - BW / 2, Y_TOP))
arrow((XS[1] + BW / 2, Y_TOP), (XS[2] - BW / 2, Y_TOP))
arrow((XS[2] + BW / 2, Y_TOP), (XS[3] - BW / 2, Y_TOP))
arrow((XS[3] + BW / 2, Y_TOP), (XS[4] - BW / 2, Y_TOP))

arrow((XS[0] + BW / 2, Y_BOT), (XS[2] - BW / 2, Y_BOT))
arrow((XS[2] + BW / 2, Y_BOT), (XS[3] - BW / 2, Y_BOT))
arrow((XS[3] + BW / 2, Y_BOT), (XS[4] - BW / 2, Y_BOT))
arrow((XS[4] + BW / 2, Y_BOT), (XS[5] - BW / 2, Y_BOT))

# le modèle entraîné alimente l'inférence
elbow((XS[4], Y_TOP - BH / 2), (XS[3], Y_BOT + BH / 2), mid_y=2.75)
ax.text((XS[3] + XS[4]) / 2, 2.75 + 0.17, "modèle chargé", fontsize=7.6,
        family=F, color=SUB, ha="center", style="italic")

fig.savefig(OUT / "architecture_overview.png", dpi=190, facecolor=BG)
fig.savefig(OUT / "architecture_overview.svg", facecolor=BG)
print("rendered")
