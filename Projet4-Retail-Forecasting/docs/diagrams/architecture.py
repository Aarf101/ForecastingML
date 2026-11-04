"""Projet 4 — Retail ML Forecasting : diagramme d'architecture AWS.

Diagramme généré par code (reproductible, versionnable) pour le rendu de
soutenance : 7 étapes numérotées, de l'ingestion Kafka/MSK au dashboard.

Usage:
    python3 docs/diagrams/architecture.py
    # -> docs/diagrams/architecture.png  (slides / vidéo)
    # -> docs/diagrams/architecture.svg  (rapport, vectoriel)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent

W, H = 13.0, 6.40
fig, ax = plt.subplots(figsize=(W, H))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
F = 'DejaVu Sans'

T = {
    'src':  ('#eef3f8', '#4a7fb5'), 'ing':  ('#fff4e3', '#e08a00'),
    'lake': ('#e9f5ea', '#2e7d32'), 'ml':   ('#e4f0fb', '#1565c0'),
    'srv':  ('#f4ecfa', '#7b1fa2'), 'agt':  ('#efe9f8', '#5e35b1'),
    'bi':   ('#fdeaf1', '#ad1457'), 'ops':  ('#efefef', '#5f5f5f'),
}
BW, BH = 1.55, 0.58
COL = [0.95, 2.80, 4.65, 6.50, 8.35, 10.20, 12.05]
r1, r2, r3, r4, r5 = 5.08, 4.22, 3.36, 2.50, 1.64


def box(x, y, w, h, text, kind='src', dashed=False, fs=7.6, weight='normal', lh=1.38):
    fc, ec = T[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle='round,pad=0.012,rounding_size=0.10',
                                linewidth=1.3, edgecolor=ec,
                                facecolor=('white' if dashed else fc),
                                linestyle=(0, (4, 2)) if dashed else 'solid', zorder=3))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, family=F,
            color='#151515', linespacing=lh, weight=weight, zorder=4)


def head(x, y, text, color):
    ax.text(x, y, text, ha='center', va='center', fontsize=8.6, family=F,
            weight='bold', color=color, zorder=4)
    ax.add_line(Line2D([x - 0.70, x + 0.70], [y - 0.14, y - 0.14], color=color, lw=1.6, zorder=2))


def arr(p1, p2, style='-', lw=1.15, label=None, lpos=0.5, ldx=0.0, ldy=0.11,
        fs=6.6, both=False, color='#5f5f5f'):
    ls = {'-': 'solid', '--': (0, (4, 2)), ':': (0, (1.6, 1.6))}[style]
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='<|-|>' if both else '-|>',
                                 mutation_scale=8.5, linewidth=lw, color=color,
                                 linestyle=ls, shrinkA=1.5, shrinkB=1.5, zorder=2))
    if label:
        ax.text(p1[0] + (p2[0] - p1[0]) * lpos + ldx,
                p1[1] + (p2[1] - p1[1]) * lpos + ldy, label,
                ha='center', va='center', fontsize=fs, family=F, style='italic',
                color='#3d3d3d', zorder=6,
                bbox=dict(boxstyle='round,pad=0.10', fc='white', ec='none', alpha=0.96))


# ---------------- title ----------------
ax.text(0.18, 6.20, 'Projet 4 — Retail ML Forecasting : architecture AWS',
        fontsize=12.6, family=F, weight='bold', color='#0d2b45', va='center')
ax.text(0.18, 5.96,
        'Prédiction de la demande (M5 / Rossmann)   ·   SageMaker + Kafka/MSK + agent LLM   ·   binôme, soutenance 10 min',
        fontsize=7.9, family=F, color='#5a6b7b', va='center')

# ---------------- stage headers ----------------
for x, t, k in [(COL[0], '① SOURCES', 'src'), (COL[1], '② INGESTION', 'ing'),
                (COL[2], '③ LAKE & ETL', 'lake'), (COL[3], '④ SAGEMAKER ML', 'ml'),
                (COL[4], '⑤ SERVING', 'srv'), (COL[5], '⑥ LLM & AGENT', 'agt'),
                (COL[6], '⑦ DASHBOARD', 'bi')]:
    head(x, 5.66, t, T[k][1])

# ---------------- boxes ----------------
box(COL[0], r1, BW, BH, 'Sales CSV\n(M5 / Rossmann)\nbackfill', 'src')
box(COL[0], r2, BW, BH, 'POS event\nsimulator\n(Python replay)', 'src')

box(COL[1], r1, BW, BH, 'Kafka topic\nsales.events\n(Redpanda ⟷ MSK)', 'ing')
box(COL[1], r2, BW, BH, 'Kinesis Data\nStreams + Firehose\n(60 s micro-batch)', 'ing', fs=7.2)

box(COL[2], r1, BW, BH, 'S3  raw/\nJSON.gz partitions', 'lake')
box(COL[2], r2, BW, BH, 'Glue ETL (PySpark)\nou Lambda\nclean · dedupe', 'lake', fs=7.2)
box(COL[2], r3, BW, BH, 'S3  curated/\nParquet (store, date)', 'lake')
box(COL[2], r4, BW, BH, 'Glue Crawler\n→ Athena (KPI)', 'lake')

box(COL[3], r2, BW, BH, 'Processing Job\nfeatures: lags,\ncalendar, promo', 'ml', fs=7.2)
box(COL[3], r3, BW, BH, 'Training Job\nLightGBM global\n+ HPO', 'ml', weight='bold')
box(COL[3], r4, BW, BH, 'Evaluation +\nConditionStep\nWAPE · MAPE · RMSE', 'ml', fs=7.0)
box(COL[3], r5, BW, BH, 'Model Registry\napprove → prod', 'ml')

box(COL[4], r2, BW, BH, 'Model Monitor\n+ CloudWatch', 'srv', dashed=True, fs=7.3)
box(COL[4], r3, BW, BH, 'SageMaker Endpoint\n(serverless)', 'srv', fs=7.3)
box(COL[4], r4, BW, BH, 'Batch Transform\n28-day forecast', 'srv', fs=7.3)
box(COL[4], r5, BW, BH, 'S3  forecast/\nP10 · P50 · P90', 'srv', fs=7.3)

box(COL[5], 3.61, BW, 1.34,
    'LangGraph agent\ntools: get_forecast,\nKPIs, stockout_risk(P90),\nanomalies\nRAG: promos, lead times\nLLM: Bedrock · JumpStart\n· Ollama (local)\nbrief hebdo → S3',
    'agt', fs=7.0, lh=1.42)

box(COL[6], 4.19, BW, 0.52, 'QuickSight\n(production option)', 'bi', dashed=True, fs=6.9)
box(COL[6], 3.34, BW, 1.02,
    'Streamlit dashboard\nKPIs · forecast vs actual\nstockout risks\n« Ask the analyst » chat\nrefresh → pipeline',
    'bi', fs=7.0, lh=1.42)

# ---------------- edges ----------------
arr((COL[0] + BW/2, r1), (COL[1] - BW/2, r1), label='replay', ldy=0.14)
arr((COL[0] + BW/2, r2 + 0.10), (COL[1] - BW/2, r1 - 0.12))
arr((COL[1], r1 - BH/2), (COL[1], r2 + BH/2), label='consume', ldx=0.36, ldy=0.0)
arr((COL[1] + BW/2, r2 + 0.12), (COL[2] - BW/2, r1 - 0.12), label='put', ldy=0.15)
arr((COL[2], r1 - BH/2), (COL[2], r2 + BH/2), label='S3 event', ldx=0.46, ldy=0.0, fs=6.3)
arr((COL[2], r2 - BH/2), (COL[2], r3 + BH/2), label='Parquet', ldx=0.41, ldy=0.0, fs=6.3)
arr((COL[2], r3 - BH/2), (COL[2], r4 + BH/2), style='--', label='crawl', ldx=0.35, ldy=0.0, fs=6.3)
arr((COL[2] + BW/2, r3 - 0.04), (COL[3] - BW/2, r2 + 0.04), label='train', ldy=0.15, fs=6.3)
arr((COL[3], r2 - BH/2), (COL[3], r3 + BH/2), label='features', ldx=0.45, ldy=0.0, fs=6.3)
arr((COL[3], r3 - BH/2), (COL[3], r4 + BH/2), label='model', ldx=0.39, ldy=0.0, fs=6.3)
arr((COL[3], r4 - BH/2), (COL[3], r5 + BH/2), label='WAPE < seuil', ldx=0.55, ldy=0.0, fs=6.3)
arr((COL[3] + BW/2, r5 + 0.16), (COL[4] - BW/2, r3 - 0.16), label='deploy', ldy=0.15, fs=6.3)
arr((COL[3] + BW/2, r5 + 0.02), (COL[4] - BW/2, r4 - 0.02), label='daily', ldy=-0.15, fs=6.3)
arr((COL[4], r3 + BH/2), (COL[4], r2 - BH/2), style=':', label='drift', ldx=0.34, ldy=0.0, fs=6.3)
arr((COL[4], r4 - BH/2), (COL[4], r5 + BH/2), label='P10/90', ldx=0.40, ldy=0.0, fs=6.3)
arr((COL[4] + BW/2, r5 + 0.12), (COL[5] - BW/2, 3.10), label='forecasts', ldy=-0.15, fs=6.3)
arr((COL[5] + BW/2, 3.72), (COL[6] - BW/2, 3.44), both=True, label='chat', ldy=0.13, fs=6.3)
arr((COL[4] + BW/2, r5 - 0.04), (COL[6] - BW/2 - 0.10, 2.90), style=':', label='S3 / Athena', ldy=-0.02, fs=6.3)

# ---------------- MLOps bar ----------------
box(6.50, 0.74, 12.64, 0.36,
    '⑧  MLOps transversal :   EventBridge (quotidien) → Step Functions / SageMaker Pipelines    ·    CloudWatch metrics & alarms    ·    versioning S3    ·    tags de coût    ·    teardown',
    'ops', fs=7.2)

# ---------------- legend ----------------
ly = 0.28
ax.add_line(Line2D([0.20, 0.58], [ly, ly], color='#5f5f5f', lw=1.2))
ax.text(0.64, ly, 'flux de données', fontsize=6.8, family=F, color='#3d3d3d', va='center')
ax.add_line(Line2D([2.05, 2.43], [ly, ly], color='#5f5f5f', lw=1.2, linestyle=(0, (4, 2))))
ax.text(2.49, ly, 'asynchrone / optionnel', fontsize=6.8, family=F, color='#3d3d3d', va='center')
ax.add_line(Line2D([4.55, 4.93], [ly, ly], color='#5f5f5f', lw=1.2, linestyle=(0, (1.6, 1.6))))
ax.text(4.99, ly, 'supervision', fontsize=6.8, family=F, color='#3d3d3d', va='center')

# ---------------- learner-lab note ----------------
ax.text(0.18, 3.52, 'Contraintes AWS Learner Lab (plan B intégré)', fontsize=7.4,
        family=F, weight='bold', color='#8a4b00', va='center')
for i, line in enumerate([
    '• MSK indisponible → Redpanda local, Kinesis en AWS',
    '• Bedrock souvent bloqué → JumpStart / Ollama',
    '• QuickSight hors lab → Streamlit (option QuickSight)',
    '• Rôle unique LabRole · endpoint serverless · teardown',
]):
    ax.text(0.18, 3.26 - i * 0.19, line, fontsize=6.8, family=F,
            color='#5a6b7b', va='center')

OUT = Path(__file__).resolve().parent
fig.savefig(OUT / 'architecture.png', dpi=190, facecolor='white')
fig.savefig(OUT / 'architecture.svg', facecolor='white')
print('rendered')
