"""Version simplifiée du diagramme d'architecture (une seule ligne, 7 étapes).

    python3 docs/diagrams/architecture_simple.py
    # -> architecture_simple.png / .svg

Volontairement dépouillé : uniquement les étapes et leurs libellés courts,
sans légende, sans notes, sans découpage en services.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent

W, H = 13.6, 2.35
BW, BH, GAP = 1.60, 1.15, 0.32
PITCH = BW + GAP
XS = [0.24 + BW / 2 + i * PITCH for i in range(7)]
Y = 1.175

F = "DejaVu Sans"

# Thème sombre : fond violet profond, cartes violet foncé, bordures colorées.
BG = "#1E1033"
CARD = "#341B52"
MAIN_TXT = "#FFFFFF"
SUB_TXT = "#C3AEDC"
ARROW = "#9B8AB8"

STEPS = [
    ("Ventes", "POS · CSV", CARD, "#7EA6E0"),
    ("Kafka", "MSK", CARD, "#F0A73C"),
    ("S3 + ETL", "Glue", CARD, "#5FBF72"),
    ("SageMaker", "Training", CARD, "#6FA8F5"),
    ("Prévisions", "P10 · P50 · P90", CARD, "#B98BE6"),
    ("Agent LLM", "LangGraph", CARD, "#A78BFA"),
    ("Dashboard", "Streamlit", CARD, "#F2719E"),
]

fig, ax = plt.subplots(figsize=(W, H))
fig.subplots_adjust(0, 0, 1, 1)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

for x, (main, sub, fill, edge) in zip(XS, STEPS):
    ax.add_patch(
        FancyBboxPatch(
            (x - BW / 2, Y - BH / 2),
            BW,
            BH,
            boxstyle="round,pad=0.012,rounding_size=0.12",
            linewidth=1.8,
            edgecolor=edge,
            facecolor=fill,
            zorder=3,
        )
    )
    ax.text(x, Y + 0.16, main, ha="center", va="center", fontsize=13.5,
            weight="bold", family=F, color=MAIN_TXT, zorder=4)
    ax.text(x, Y - 0.24, sub, ha="center", va="center", fontsize=10,
            family=F, color=SUB_TXT, zorder=4)

for left, right in zip(XS[:-1], XS[1:]):
    ax.add_patch(
        FancyArrowPatch(
            (left + BW / 2 + 0.02, Y),
            (right - BW / 2 - 0.02, Y),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.6,
            color=ARROW,
            zorder=2,
        )
    )

fig.savefig(OUT / "architecture_simple.png", dpi=190, facecolor=BG)
fig.savefig(OUT / "architecture_simple.svg", facecolor=BG)
print("rendered")
