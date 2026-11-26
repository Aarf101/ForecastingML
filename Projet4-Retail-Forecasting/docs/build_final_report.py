"""Construit le rapport final depuis les preuves AWS, sans appel à un service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, PageBreak, Image, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/evidence/completion_2026-11-26.json'
OUT = ROOT / 'output/pdf/rapport_final.pdf'


def sections(evidence):
    """Quatre points du rendu, protocole et limites vérifiables."""
    evaluation = evidence['evaluation']
    metrics = evaluation['regression_metrics']
    details = evaluation['independent_details']
    wape = metrics['test_wape']['value']
    baseline = metrics['baseline_wape']['value']
    rmsle = mean(v['rmsle'] for v in details.values())
    coverage = mean(v['coverage_80'] for v in details.values())
    hpo = evidence['hpo']
    return [
        ('1. Besoin et pipeline complet', [
            "L'application aide un responsable des achats à examiner les ventes attendues par magasin et article. Elle produit une prévision pour chacun des 16 jours suivant la dernière observation, puis une bande P10/P90. Elle ne passe aucune commande automatiquement.",
            "Le jeu effectivement évalué contient 876 000 ventes synthétiques : 20 magasins, 60 articles et 730 jours, du 1er janvier 2023 au 30 décembre 2024. Les dates cibles de la démonstration sont donc du 31 décembre 2024 au 15 janvier 2025. Elles ne représentent pas des ventes réelles de novembre 2026.",
            "Le transport temps réel est Kafka open-source sur EC2, relayé vers Firehose DirectPut puis S3. Cette branche a été validée précédemment avec 20 000 événements ; les preuves et leur date sont conservées dans le dépôt. Le test complet du 26 novembre part des objets raw déjà présents dans S3.",
            "Le workflow mensuel exécute Glue, puis un pilote Processing qui attend SageMaker Pipelines : préparation des features, entraînement, évaluation indépendante, gate et inscription au registre. Un Batch Transform candidat est validé avant publication du manifeste actif. Le workflow quotidien réexécute Glue et l'inférence sur le modèle actif, sans entraînement.",
        ]),
        ('2. Services AWS et substitutions du Learner Lab', [
            "S3 conserve raw, curated, features, modèles, évaluations, prévisions et rapports. Glue transforme les événements en Parquet dédupliqué. SageMaker Processing, Training, Pipelines, HPO, Model Registry, Batch Transform et l'endpoint d'inférence sont utilisés selon les preuves d'exécution jointes.",
            "Step Functions orchestre les jobs avec attente de leur état terminal. EventBridge définit le mensuel à 02:00 UTC le premier du mois et le quotidien à 04:00 UTC. Les deux horaires restent désactivés pour limiter la consommation du crédit. CloudWatch conserve les journaux ; Athena produit les KPI ; Streamlit est hébergé sur le notebook SageMaker.",
            "MSK refusé par les permissions du lab est remplacé par Kafka sur EC2. Streamlit remplace QuickSight. Ollama sert le LLM textuel optionnel et LangChain gère le prompt. Ces choix préservent les fonctions attendues, sans prétendre avoir exécuté les services remplacés.",
            "Les diagrammes du dépôt décrivent l'architecture de référence. Les options non démontrées, notamment Model Monitor ou un dashboard QuickSight natif, ne doivent pas être confondues avec les composants validés. Les identifiants de jobs, statuts et résultats des preuves JSON font foi.",
        ]),
        ('3. Modèle et choix des hyperparamètres', [
            "Le modèle global est LightGBM, partagé entre magasins et articles. La stratégie directe utilise un booster de moyenne par horizon et deux boosters quantiles (alpha 0,1 et 0,9), soit 48 boosters pour 16 jours. La moyenne n'est pas présentée comme un quantile P50.",
            "Le modèle de référence du pipeline utilise au plus 400 arbres, un learning rate de 0,05, 63 feuilles et un arrêt anticipé après 40 itérations sans amélioration sur la validation. Les mêmes paramètres de structure servent aux quantiles, avec objectif et métrique quantile. Ces choix privilégient un calcul tabulaire reproductible et un temps d'exécution borné.",
            f"Le HPO AWS terminé explore six essais planifiés sur J+1, en minimisant uniquement valid_wape. Résultat gagnant : {hpo['best_validation_wape']:.6f}. Paramètres : {json.dumps(hpo['best_parameters'], ensure_ascii=False)}. Les compteurs réels des essais sont conservés dans les preuves. Cette exploration sur un horizon ne démontre pas l'optimum commun des 16 horizons et ne remplace pas automatiquement le modèle de référence.",
            "Les features comprennent les retards, moyennes et écarts-types glissants calendaires, identifiants encodés avec un vocabulaire figé et calendrier de la cible. Stock et rupture sont exclus. Les jours absents restent inconnus : ils ne deviennent pas des ventes nulles. Le catalogue actif est déterminé à la dernière origine, avec contrôle des séries récemment absentes.",
            "Le split est temporel, avec validation de 56 jours et test de 28 jours. Les labels sont purgés aux frontières pour qu'une cible future de train n'entre pas dans la validation ou le test. Il n'y a pas de découpage aléatoire. L'évaluation est à origines glissantes : chaque origine utilise uniquement son historique disponible.",
        ]),
        ('4. Évaluation indépendante et incertitude', [
            f"L'étape Processing recharge les boosters et les features de son exécution et recalcule les métriques sur le test. WAPE moyen des 16 horizons : {wape:.6f}, baseline saisonnière : {baseline:.6f}, gain relatif : {(baseline-wape)/baseline:.2%}. RMSLE moyen : {rmsle:.6f}. La baseline utilise la dernière date connue du même jour de semaine que la cible.",
            f"Les P10/P90 proviennent d'objectifs de régression quantile, et non d'un pourcentage arbitraire autour de la moyenne. Les deux valeurs sont ordonnées en cas de croisement et écrêtées à zéro. Couverture empirique de l'intervalle sur le test : {coverage:.2%}, pour 80 % nominal. Pinball loss et largeur sont recalculées par horizon dans independent_details.",
            "La couverture nominale n'est pas une garantie individuelle, ni une calibration conforme. Il ne faut pas sommer les P90 quotidiens pour prétendre obtenir un P90 de demande cumulée. Une politique de stock nécessite aussi coûts, délais, niveaux de service et données de demande perdue.",
        ]),
        ('5. Publication et intégration du LLM', [
            "L'approbation automatique est réservée au pipeline avec évaluation indépendante complète et WAPE moyen inférieur ou égal à 0,45. Les parcours autonomes d'inscription conservent PendingManualApproval. Le registre seul ne sélectionne jamais le modèle du dashboard.",
            "Chaque fichier de prévisions est immuable sous forecast/runs. Le manifeste deployment/active.json lie modèle, artefact, métriques et clé du fichier validé. La publication utilise IfMatch ou IfNoneMatch ; une concurrence laisse l'ancien manifeste intact. previous conserve le dernier résultat, sans historique récursif infini. Les sorties incomplètes, non finies, négatives ou aux quantiles inversés sont refusées.",
            "Le LLM n'entraîne pas le modèle et ne calcule pas les prévisions. La chaîne LangChain est explicite : agrégations déterministes, PromptTemplate, backend Ollama qwen2.5:3b, puis validation. Le rapport contient trois phrases d'analyse et deux actions. Le décodage Ollama est contraint par un schéma JSON : les trois faits sont recopiés exactement et les deux actions choisies parmi des recommandations autorisées. Une validation indépendante du texte refuse toute reformulation factuelle, doublon d’action ou affirmation ajoutée avant publication.",
            "Le backend Ollama de la démonstration tourne localement ; les prévisions sont lues depuis le manifeste AWS et le rapport validé est enregistré localement puis dans S3. Il n'y a ni agent autonome, ni RAG, ni décision de réassort automatique. La validation formelle du texte ne garantit pas toutes ses interprétations métier : une relecture humaine reste nécessaire.",
        ]),
        ('6. Dashboard, démonstration et exploitation', [
            "Le dashboard affiche les prévisions actives, le modèle, l'origine, les 16 horizons, l'historique, les P10/P90 et les KPI. Des filtres magasin, article et horizon permettent d'examiner les séries. En mode AWS, aucun ancien fichier local ne remplace silencieusement un manifeste absent. Le mode s3mock utilise les résultats locaux.",
            "La vue Athena forecast_active sélectionne le fichier désigné par le manifeste. Un nouveau run n'exige donc aucune modification de catalogue. Le navigateur réel vérifie le chargement et les graphiques du Streamlit hébergé ; la vidéo de deux minutes enregistre cette interface et ses interactions.",
            "Pour reproduire : make test ; make aws-workflows-up ; make aws-pipeline-run ; make aws-inference-run ; make athena ; make aws-dashboard. Le brief utilise make brief avec le backend Ollama démarré. Les exigences de dépendances et les limites du rôle Learner Lab sont décrites dans les runbooks.",
            "Après la validation, le notebook a été arrêté, l'endpoint de test supprimé et aucun job du projet n'est resté actif. Les deux horaires sont désactivés. La suite locale compte 150 tests réussis ; la vidéo dure exactement 120 secondes. Le lac S3, les modèles et le disque du notebook sont conservés : leur stockage peut continuer à être facturé. La répétition orale doit être effectuée par les étudiants avec le script de dix minutes fourni.",
        ]),
    ], [('WAPE du modèle', f'{wape:.6f}'), ('Baseline saisonnière', f'{baseline:.6f}'),
        ('Gain relatif', f'{(baseline-wape)/baseline:.2%}'), ('RMSLE moyen', f'{rmsle:.6f}'),
        ('Couverture P10/P90 sur test', f'{coverage:.2%}'),
        ('Prévisions / horizons', '19 200 / 16'),
        ('Séries actives', '1 200')]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=EVIDENCE)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args(argv)
    evidence = json.loads(args.evidence.read_text())
    if evidence['monthly']['status'] != 'SUCCEEDED' or evidence['daily']['status'] != 'SUCCEEDED':
        raise ValueError('ne pas exporter un rapport final avant la réussite des workflows')
    if not evidence['endpoint'].get('valid_quantiles') or evidence['hpo']['status'] != 'Completed':
        raise ValueError('appel endpoint réel et HPO terminé requis')
    body, summary = sections(evidence)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    font = Path('/usr/share/fonts/truetype/dejavu')
    pdfmetrics.registerFont(TTFont('DejaVu', str(font/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DejaVu-Bold', str(font/'DejaVuSans-Bold.ttf')))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='RetailBody', fontName='DejaVu', fontSize=10,
                              leading=15, spaceAfter=10, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='RetailTitle', fontName='DejaVu-Bold', fontSize=30,
                              leading=38, spaceAfter=22, textColor=colors.HexColor('#15394f')))
    styles.add(ParagraphStyle(name='RetailHeading', fontName='DejaVu-Bold', fontSize=17,
                              leading=23, spaceBefore=12, spaceAfter=14, textColor=colors.HexColor('#15394f')))
    styles.add(ParagraphStyle(name='RetailSmall', fontName='DejaVu', fontSize=8,
                              leading=12, spaceAfter=8))
    story = [Spacer(1, 24*mm), Paragraph('Retail ML Forecasting', styles['RetailTitle']),
             Paragraph('Projet 4 - Rapport technique final', styles['RetailHeading']),
             Paragraph('Ventes observées de J+1 à J+16 sur AWS Academy Learner Lab', styles['RetailBody']),
             Paragraph('Validation du 26 novembre 2026 - données synthétiques', styles['RetailBody']),
             Spacer(1, 12*mm)]
    table = Table([[Paragraph(escape(a), styles['RetailBody']), Paragraph(escape(b), styles['RetailBody'])] for a,b in summary], colWidths=[103*mm,68*mm])
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#eff4f7')),
                               ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),12),
                               ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),2)]))
    story += [table, PageBreak()]
    for title, paragraphs in body:
        story.append(Paragraph(escape(title), styles['RetailHeading']))
        for text in paragraphs:story.append(Paragraph(escape(text), styles['RetailBody']))
    story += [PageBreak(), Paragraph('Annexe - Preuves et architecture', styles['RetailHeading'])]
    for key in ['monthly', 'daily', 'endpoint', 'hpo']:
        proof=evidence[key]
        name=proof.get('arn',proof.get('name',proof.get('endpoint','')))
        story.append(Paragraph(escape(f'{key}: {proof.get("status", "")} - {name}'), styles['RetailSmall']))
    story.append(Paragraph('Preuves détaillées : docs/evidence/completion_2026-11-26.json ; rapport indépendant, journaux de tests et vidéo associés.', styles['RetailBody']))
    for name in ['architecture_training.png','architecture_inference.png']:
        path=ROOT/'docs/diagrams'/name
        if path.exists():
            from PIL import Image as PILImage
            with PILImage.open(path) as img:w,h=img.size
            story += [Spacer(1,5*mm), Image(str(path),width=174*mm,height=174*mm*h/w)]
    story.append(Paragraph('Diagrammes de référence : les preuves JSON distinguent les services réellement exécutés des options de la cible.',styles['RetailSmall']))
    refs = [('LightGBM - paramètres et objectifs quantiles','https://lightgbm.readthedocs.io/en/stable/Parameters.html'),
            ('LangChain - modèles et génération structurée','https://docs.langchain.com/oss/python/langchain/models'),
            ('Ollama - sorties contraintes par schéma JSON','https://github.com/ollama/ollama/blob/main/docs/capabilities/structured-outputs.mdx'),
            ('AWS - SageMaker avec Step Functions','https://docs.aws.amazon.com/step-functions/latest/dg/connect-sagemaker.html'),
            ('AWS - écritures conditionnelles S3','https://docs.aws.amazon.com/S3/latest/userguide/conditional-writes.html')]
    story += [Paragraph('Références', styles['RetailHeading'])]
    for title,url in refs:story.append(Paragraph(f'<link href="{url}" color="#155c90">{escape(title)}</link>',styles['RetailSmall']))
    def page(canvas, doc):
        canvas.setFont('DejaVu',8);canvas.setFillColor(colors.HexColor('#526677'))
        canvas.drawString(18*mm,12*mm,'Projet 4 - Retail ML Forecasting | 26 novembre 2026')
        canvas.drawRightString(A4[0]-18*mm,12*mm,str(doc.page))
    doc=SimpleDocTemplate(str(args.out),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,
                          topMargin=18*mm,bottomMargin=22*mm,title='Projet 4 - Retail ML Forecasting',author='Projet 4')
    doc.build(story,onFirstPage=page,onLaterPages=page)
    markdown='# Rapport final - Retail ML Forecasting\n\nValidation du 26 novembre 2026.\n\n'
    for title,paragraphs in body:markdown+='## '+title+'\n\n'+'\n\n'.join(paragraphs)+'\n\n'
    (ROOT/'docs/rapport_final.md').write_text(markdown.rstrip()+'\n')
    print(args.out)


if __name__ == '__main__':
    main()
