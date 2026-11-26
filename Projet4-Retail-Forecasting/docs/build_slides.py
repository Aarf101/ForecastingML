"""Génère docs/slides.pptx (5 slides) — exposé idée projet, projet 4.

    python3 docs/build_slides.py     # ou : make slides

Le contenu éditorial de référence reste docs/slides.md ; ce script produit la
version présentable (16:9), avec le diagramme d'architecture intégré.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
IMG = HERE / "diagrams" / "architecture.png"

W, H, M = 13.333, 7.5, 0.62
FONT = "Calibri"

DARK = RGBColor(0x0D, 0x2B, 0x45)
INK = RGBColor(0x1F, 0x2A, 0x36)
GREY = RGBColor(0x5A, 0x6B, 0x7B)
MUTED = RGBColor(0x9F, 0xB4, 0xC9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BG = RGBColor(0xF7, 0xF9, 0xFB)
ORANGE = RGBColor(0xE0, 0x8A, 0x00)
BLUE = RGBColor(0x15, 0x65, 0xC0)
PURPLE = RGBColor(0x5E, 0x35, 0xB1)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
PINK = RGBColor(0xAD, 0x14, 0x57)


def textbox(slide, l, t, w, h, text, size=15, bold=False, color=INK,
            align=PP_ALIGN.LEFT, italic=False, anchor=MSO_ANCHOR.TOP, spacing=1.0):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = spacing
    run = p.add_run()
    run.text = text
    f = run.font
    f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, FONT
    f.color.rgb = color
    return tb


def rect(slide, l, t, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    sh = slide.shapes.add_shape(shape, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = 0.12
    return sh


def blank(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = BG
    return slide


def header(slide, title, kicker, idx):
    rect(slide, 0, 0, W, 1.04, DARK, shape=MSO_SHAPE.RECTANGLE)
    rect(slide, 0, 1.04, W, 0.05, ORANGE, shape=MSO_SHAPE.RECTANGLE)
    textbox(slide, M, 0.17, 10.5, 0.3, kicker.upper(), 10.5, True, MUTED)
    textbox(slide, M, 0.44, 11.3, 0.5, title, 24, True, WHITE)
    textbox(slide, W - 1.15, 0.46, 0.55, 0.35, f"{idx}/5", 11.5, False, MUTED, PP_ALIGN.RIGHT)


def footer(slide):
    textbox(slide, M, H - 0.40, 8.5, 0.28,
            "MIT 2026  ·  Projet 4  ·  Retail ML Forecasting", 9, False, GREY)


def rich(paragraph, text, size, color=INK):
    """Rend ``**gras**`` à l'intérieur d'une puce."""
    for i, part in enumerate(text.split("**")):
        if not part:
            continue
        run = paragraph.add_run()
        run.text = part
        run.font.size = Pt(size)
        run.font.name = FONT
        run.font.bold = i % 2 == 1
        run.font.color.rgb = color


def bullets(slide, l, t, w, h, heading, items, accent, size=14.5):
    textbox(slide, l, t, w, 0.3, heading, 14.5, True, accent)
    rect(slide, l, t + 0.36, 0.85, 0.04, accent, shape=MSO_SHAPE.RECTANGLE)
    tb = slide.shapes.add_textbox(Inches(l), Inches(t + 0.52), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(6.5)
        p.line_spacing = 1.05
        rich(p, "•  " + item, size)
    return tb


def kpi_strip(slide, top, entries):
    """Bandeau de 3 indicateurs : (valeur, libellé, couleur)."""
    bw, gap = 3.83, 0.27
    for i, (value, label, color) in enumerate(entries):
        left = M + i * (bw + gap)
        rect(slide, left, top, bw, 0.92, WHITE)
        rect(slide, left, top, 0.075, 0.92, color, shape=MSO_SHAPE.RECTANGLE)
        textbox(slide, left + 0.30, top + 0.14, bw - 0.45, 0.4, value, 20, True, color)
        textbox(slide, left + 0.30, top + 0.56, bw - 0.45, 0.28, label, 10.5, False, GREY)


def process_strip(slide, top, steps, accent):
    """Parcours utilisateur en N étapes : [(numéro, libellé)]."""
    n = len(steps)
    gap = 0.22
    bw = (W - 2 * M - gap * (n - 1)) / n
    for i, (num, label) in enumerate(steps):
        left = M + i * (bw + gap)
        rect(slide, left, top, bw, 0.98, WHITE)
        rect(slide, left, top, bw, 0.055, accent, shape=MSO_SHAPE.RECTANGLE)
        textbox(slide, left + 0.22, top + 0.15, 0.6, 0.3, num, 13, True, accent)
        textbox(slide, left + 0.22, top + 0.45, bw - 0.44, 0.45, label, 10.5, False, INK, spacing=1.0)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def slide_1(prs):
    s = blank(prs)
    header(s, "Retail ML Forecasting — anticiper la demande, magasin par magasin",
           "Projet 4 · soutenance finale", 1)

    bullets(s, M, 1.45, 6.25, 2.9, "Le besoin", [
        "Réassort aujourd'hui à l'expertise : aucune prévision chiffrée",
        "Rupture de stock → ventes perdues, clients mécontents",
        "Surstock → trésorerie immobilisée et démarques",
        "Objectif : prévoir J+1 à J+16 par (magasin, article), **avec une incertitude exploitable**",
    ], ORANGE)

    bullets(s, 7.15, 1.45, 5.5, 2.4, "Pour qui ?", [
        "Responsable approvisionnement → décisions de réassort",
        "Chef de rayon → priorisation des ruptures",
        "Direction → pilotage par KPI (rupture, surstock, WAPE)",
    ], BLUE)

    rect(s, 7.15, 3.35, 5.5, 1.22, RGBColor(0xE8, 0xF0, 0xFA))
    textbox(s, 7.38, 3.51, 5.05, 0.95,
            "L'application : une console de pilotage de la demande — dashboard "
            "+ brief LangChain/Ollama fondé sur les agrégats, "
            "sans agent autonome ni estimation de demande perdue.", 11.5, False, INK, spacing=1.1)

    kpi_strip(s, 4.90, [
        ("0,45", "seuil de promotion WAPE", ORANGE),
        ("16 jours", "horizons quotidiens", BLUE),
        ("1 200", "séries magasin × article", PURPLE),
    ])
    textbox(s, M, 6.10, W - 2 * M, 0.4,
            "Référence AWS vérifiée : WAPE 0,422609 contre 0,559139 pour la baseline "
            "saisonnière ; les données évaluées sont synthétiques.",
            13.5, True, DARK)
    footer(s)
    notes(s, "On part d'un constat terrain : le réassort se fait à l'expérience. "
             "L'objectif est d'en faire une décision chiffrée, avec une fourchette "
             "de risque et non un simple chiffre moyen.")


def slide_2(prs):
    s = blank(prs)
    fill = s.background.fill
    fill.solid()
    fill.fore_color.rgb = WHITE  # le diagramme a un fond blanc : on l'harmonise
    header(s, "Une architecture AWS en 8 étapes, du flux de caisse au dashboard",
           "Architecture retenue", 2)
    width = 11.95
    height = width * 1216 / 2470
    s.shapes.add_picture(str(IMG), Inches((W - width) / 2), Inches(1.18),
                         width=Inches(width), height=Inches(height))
    textbox(s, M, 1.18 + height + 0.04, W - 2 * M, 0.25,
            "Kafka EC2 → Firehose → S3 → Glue → SageMaker → prévisions → Streamlit. "
            "LangChain/Ollama rédige le brief optionnel ; les horaires récurrents sont désactivés.",
            8.5, False, GREY)
    notes(s, "Deux natures de flux : le batch quotidien (prévisions de J+1 à J+16) et le "
             "temps réel (requête de serving). Le diagramme est "
             "généré par code, donc versionné avec le projet.")


def slide_3(prs):
    s = blank(prs)
    header(s, "Des ventes synthétiques et un protocole temporel reproductible", "Les données", 3)

    bullets(s, M, 1.45, 5.95, 3.2, "Sources", [
        "**Données utilisées** : 876 000 ventes synthétiques, 20 magasins et 60 articles",
        "**Période** : du 01/01/2023 au 30/12/2024 ; prévisions relatives à cette origine",
        "**Catalogue actif** : 1 200 séries ; les anciens historiques sans lien restent dans S3",
    ], GREEN, size=15)

    bullets(s, 6.85, 1.45, 5.8, 3.6, "Signal et features", [
        "Cible : ventes à **J+h**, h = 1…16 (direct multi-horizon)",
        "Lags 1/7/14/21/28/35, moyennes et écarts-types glissants 7/14/28",
        "Calendrier, jours fériés, promo, prix, lead time magasin",
        "Découpage **temporel purgé** : validation 56 jours, test 28 jours",
        "Doublons d'ingestion dédupliqués sur (magasin, article, date)",
    ], PURPLE, size=15)

    kpi_strip(s, 4.55, [
        ("876 000", "lignes de ventes simulées", GREEN),
        ("11", "colonnes par événement", PURPLE),
        ("730 j", "d'historique (2 ans)", BLUE),
    ])

    rect(s, M, 5.72, W - 2 * M, 0.85, RGBColor(0xFD, 0xF0, 0xE6))
    textbox(s, M + 0.25, 5.86, W - 2 * M - 0.5, 0.6,
            "Les ventes sont censurées pendant les ruptures : ce modèle prédit les ventes "
            "observées, pas la demande perdue. Stock et indicateur de rupture sont exclus "
            "des features ; aucune vente à zéro n’est inventée pour les jours absents.",
            11, False, INK, spacing=1.05)
    footer(s)
    notes(s, "Le point à faire passer : on sait déjà quel piège va nous mordre — la "
             "censure par rupture de stock et les doublons at-least-once de Kafka. "
             "Les deux sont traités explicitement dans le code.")


def slide_4(prs):
    s = blank(prs)
    header(s, "Prévisions, incertitude et rapport textuel contrôlé",
           "Fonctionnalités et services", 4)

    bullets(s, M, 1.45, 5.95, 3.9, "Fonctionnalités", [
        "Prévisions J+1…J+16 : **moyenne et P10/P90**, 48 boosters LightGBM",
        "Tableau de bord : historique, prévisions, bande P10/P90 et filtres",
        "LangChain : agrégats → PromptTemplate → Ollama → validation du texte",
        "Brief publié dans S3 ; aucun chiffre absent du contexte ne doit être publié",
    ], PINK, size=14.5)

    bullets(s, 6.85, 1.45, 5.8, 4.2, "Services mobilisés", [
        "Stockage et ingestion : **S3**, **Glue**, **Kafka EC2 + Firehose**",
        "ML : **SageMaker** (Processing, Training, HPO, Pipelines, Registry, Endpoint)",
        "IA : **LangChain + Ollama**, rapport textuel ; pas de RAG ni agent autonome",
        "Restitution : **Streamlit sur notebook SageMaker**, KPI Athena",
        "Orchestration et MLOps : **EventBridge**, **Step Functions**, **CloudWatch**",
    ], BLUE, size=14)

    process_strip(s, 4.85, [
        ("1", "16 jours + moyenne + P10/P90"),
        ("2", "Filtres et courbe d’incertitude"),
        ("3", "Brief LangChain/Ollama"),
        ("4", "Publication versionnée dans S3"),
    ], PINK)
    footer(s)
    notes(s, "La valeur ajoutée n'est pas le modèle seul, c'est la boucle complète : la "
             "prévision alimente une décision et le brief résume les agrégats. C'est ce "
             "qui distingue le projet d'un notebook.")


def slide_5(prs):
    s = blank(prs)
    header(s, "Résultats, preuves AWS et limites",
           "Validation et démonstration", 5)

    bullets(s, M, 1.45, 5.95, 3.0, "Parcours vérifié dans le Learner Lab", [
        "**19 200 prévisions** : 1 200 séries × 16 jours ; manifeste actif versionné",
        "Ingestion Kafka/Firehose : **20 000 événements** validés précédemment",
        "Évaluation indépendante, gate WAPE et publication conditionnelle S3",
        "Tests unitaires sans services externes ; preuves cloud archivées",
    ], GREEN, size=15)

    bullets(s, 6.85, 1.45, 5.8, 2.6, "Limites à expliquer au jury", [
        "Validation sur données synthétiques ; M5/Rossmann non benchmarkés",
        "HPO exploratoire sur J+1 ; ne prouve pas l’optimum sur 16 horizons",
        "P10/P90 : 80 % nominal, couverture réelle mesurée sur le test",
        "Rapport LLM à relire ; aucune demande perdue reconstituée",
    ], ORANGE, size=15)

    kpi_strip(s, 4.75, [
        ("0.422609", "WAPE — référence AWS", GREEN),
        ("0.559139", "WAPE — baseline saisonnière", GREY),
        ("−24,4 %", "gain relatif vérifié", ORANGE),
    ])
    textbox(s, M, 6.02, W - 2 * M, 0.5,
            "Démonstration : filtres Streamlit, incertitude, version du modèle et prévisions actives. "
            "Les identifiants exacts et les résultats sont consignés dans le rapport final.",
            11.5, True, DARK)
    footer(s)
    notes(s, "Présenter les preuves du rapport final, puis reconnaître les limites métier. "
             "Le script de soutenance fournit un parcours de dix minutes avec vidéo.")


def main() -> int:
    if not IMG.exists():
        raise SystemExit(f"{IMG} manquant — lancer d'abord : make diagram")
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    for build in (slide_1, slide_2, slide_3, slide_4, slide_5):
        build(prs)
    out = HERE / "slides.pptx"
    prs.save(out)
    print(f"[slides] {len(prs.slides._sldIdLst)} slides -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


