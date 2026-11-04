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
STEPS = [
    ("Ventes", "POS · CSV", "#eef3f8", "#4a7fb5"),
    ("Kafka", "MSK", "#fff4e3", "#e08a00"),
    ("S3 + ETL", "Glue", "#e9f5ea", "#2e7d32"),
    ("SageMaker", "Training", "#e4f0fb", "#1565c0"),
    ("Prévisions", "P10 · P50 · P90", "#f4ecfa", "#7b1fa2"),
    ("Agent LLM", "LangGraph", "#efe9f8", "#5e35b1"),
    ("Dashboard", "Streamlit", "#fdeaf1", "#ad1457"),
]

fig, ax = plt.subplots(figsize=(W, H))
fig.subplots_adjust(0, 0, 1, 1)
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
            linewidth=1.6,
            edgecolor=edge,
            facecolor=fill,
            zorder=3,
        )
    )
    ax.text(x, Y + 0.16, main, ha="center", va="center", fontsize=13.5,
            weight="bold", family=F, color="#111111", zorder=4)
    ax.text(x, Y - 0.24, sub, ha="center", va="center", fontsize=10,
            family=F, color="#5a6b7b", zorder=4)

for left, right in zip(XS[:-1], XS[1:]):
    ax.add_patch(
        FancyArrowPatch(
            (left + BW / 2 + 0.02, Y),
            (right - BW / 2 - 0.02, Y),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.6,
            color="#5f5f5f",
            zorder=2,
        )
    )

fig.savefig(OUT / "architecture_simple.png", dpi=190, facecolor="white")
fig.savefig(OUT / "architecture_simple.svg", facecolor="white")
print("rendered")
