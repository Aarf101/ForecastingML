"""Produit la synthèse et la figure de l'ablation L1/calendrier."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    """Régénère les documents depuis les mesures, sans réentraîner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="reports/median_calendar_benchmark.json")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    selected = report["selected"]
    baseline = report["test"]["l2_existing"]
    candidate = report["test"][selected]
    gain = 100 * (1 - candidate["mean_wape"] / baseline["mean_wape"])
    lines = ["# LightGBM : erreur absolue et calendrier cible", "",
             "Lors de cette expérience locale, aucun appel AWS ni changement du modèle alors actif v7. "
             "L'intégration ultérieure est documentée dans [le parcours de production](median_production.md).", "",
             "## Protocole fixé avant l'évaluation", "",
             f"Réplication synthétique indépendante : seed **{report['protocol']['seed']}**, "
             "20 magasins × 60 articles × 730 jours, même simulateur, bruit et ruptures que le jeu initial. "
             "Les niveaux latents des articles/magasins changent : tous les modèles, y compris le témoin L2, "
             "sont réentraînés sur cette réplication. Ce résultat ne remplace pas les scores v7/Chronos "
             "obtenus sur l'ancien jeu.", "",
             f"Validation à partir du **{report['valid_start']}** ; test réservé : cibles du "
             f"**{report['test_target_start']} au {report['test_target_end']}**, "
             "soit 28 jours communs aux 16 horizons. Les cibles sont purgées aux frontières. "
             "Les deux blocs chronologiques de validation sont conservés dans le JSON.", "",
             "Cinq variantes préfixées ; sélection par moyenne des WAPE de validation des 16 horizons. "
             "Le test n'est calculé que pour le témoin et la variante sélectionnée, après enregistrement "
             "du choix. Les itérations sont choisies par early stopping L1 sur la validation "
             "(400 arbres maximum, patience 40), mêmes paramètres structurels et seed 42.", "",
             "Calendrier supplémentaire : jours fériés du simulateur, week-end, début/fin de mois, "
             "sinus/cosinus annuels et semi-annuels de la date cible. Aucun prix, promotion, stock "
             "ou volume futur observé. Le calendrier métier doit être remplacé pour des données réelles.", "",
             "## Sélection sur la validation", "",
             "| Variante | WAPE validation | Bloc 1 | Bloc 2 |", "|---|---:|---:|---:|"]
    for name, score in report["validation"].items():
        rows = [r for r in report["runs"] if r["variant"] == name]
        blocks = np.mean([r["valid_blocks_wape"] for r in rows], axis=0)
        lines.append(f"| {name} | {score:.4%} | {blocks[0]:.4%} | {blocks[1]:.4%} |")
    lines += ["", f"Variante retenue avant ouverture du test : **{selected}**. "
              "Les scores L1 et quantile médian avec calendrier sont pratiquement identiques "
              "sur la validation ; leur faible écart ne démontre pas une supériorité de l'une sur l'autre.", "",
              "## Test réservé", "", "| Modèle | WAPE moyen | RMSLE moyen |",
              "|---|---:|---:|"]
    for name in dict.fromkeys(["l2_existing", selected]):
        score = report["test"][name]
        lines.append(f"| {name} | {score['mean_wape']:.4%} | {score['mean_rmsle']:.6f} |")
    lines += ["", f"Variation relative favorable du WAPE : **{gain:.2f} %**. "
              "Une WAPE plus basse est préférable. Il ne s'agit pas d'un taux d'exactitude.", "",
              f"Le candidat améliore {sum(c['wape'] < b['wape'] for c, b in zip(candidate['by_horizon'], baseline['by_horizon']))} "
              "horizons sur 16, sur 537 600 couples évalués. Le RMSLE est également indiqué "
              "pour vérifier que le gain n'est pas limité à une seule métrique.", "",
              "![WAPE par horizon](../reports/median_calendar_by_horizon.png)", "",
              "## Limites et reproduction", "",
              "Une seule réplication synthétique ; pas de preuve sur données réelles, pas de "
              "nouvelle exécution AWS, pas de quantiles P10/P90 pour ce candidat. "
              "Cet essai isolait les features sans modifier la production. "
              "L'intégration ultérieure des intervalles et du pipeline AWS est suivie dans "
              "[le parcours de production](median_production.md) ; ce rapport conserve les mesures locales.", "",
              "```bash", "make benchmark-median", ".venv/bin/python docs/build_median_report.py", "```", "",
              "Le cache local `data/benchmark/median-calendar/` contient protocole préalable, choix figé, "
              "80 modèles et prédictions. Les données/modèles ne sont pas versionnés. Les mesures "
              "complètes sont dans `reports/median_calendar_benchmark.json`."]
    diagnostic_path = Path("reports/median_project_diagnostic.json")
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text())
        if diagnostic["selection_report_sha256"] != hashlib.sha256(Path(args.report).read_bytes()).hexdigest():
            raise ValueError("diagnostic provenant d'une autre sélection")
        lines += ["", "## Diagnostic secondaire sur le jeu du projet", "",
                  "Après gel du candidat, réentraînement sur le jeu d'origine, avec les mêmes "
                  "frontières et paramètres que v7 (sauf objectif et calendrier). "
                  "Appariement exact des 537 600 couples évalués aux prédictions du véritable artefact v7. "
                  "Ce test a déjà été consulté : ces chiffres sont descriptifs et ne servent pas à choisir le modèle.", "",
                  "| Modèle | WAPE moyen | RMSLE moyen |", "|---|---:|---:|",
                  f"| v7 déployé | {diagnostic['v7_wape']:.4%} | {diagnostic['v7_rmsle']:.6f} |",
                  f"| {selected}, candidat local | {diagnostic['candidate_wape']:.4%} | {diagnostic['candidate_rmsle']:.6f} |",
                  "", "```bash", ".venv/bin/python -m retail_forecast.evaluate.median_project_check",
                  ".venv/bin/python docs/build_median_report.py", "```", "",
                  "Ce diagnostic exige `data/raw/sales.parquet` et l'artefact v7 exact "
                  "dans `data/benchmark/active-model.tar.gz`, absents de Git. Leurs SHA-256 "
                  "sont conservés dans `reports/median_project_diagnostic.json`. "
                  "Le benchmark principal génère lui-même sa réplication et n'exige pas ces fichiers."]
        chronos_path = Path("reports/chronos2_full_benchmark.json")
        if chronos_path.exists():
            chronos = json.loads(chronos_path.read_text())
            if all(diagnostic[key] == chronos[key] for key in ("dataset_sha256", "reference_artifact_sha256")):
                lines += ["", f"Le benchmark Chronos-2 existant, sur ces mêmes données et cette même référence v7, "
                          f"avait donné **{chronos['metrics']['chronos']['mean_wape']:.4%}** de WAPE. "
                          "Le candidat médian est donc aussi inférieur sur ce diagnostic descriptif. "
                          "Chronos n'a pas été évalué sur la nouvelle réplication indépendante."]
    Path("docs/median_calendar_benchmark.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for name in dict.fromkeys(["l2_existing", selected]):
        scores = report["test"][name]["by_horizon"]
        ax.plot([r["horizon"] for r in scores], [100*r["wape"] for r in scores], marker="o", label=name)
    ax.set(xlabel="Horizon (jours)", ylabel="WAPE (%)", title="Test réservé — réplication synthétique indépendante",
           xticks=range(1, 17))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig("reports/median_calendar_by_horizon.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
