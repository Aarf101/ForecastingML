"""Étape d'évaluation du SageMaker Pipeline (conteneur Processing).

Ce script tourne **dans un job Processing**, pas en local. Il ne réentraîne rien :
il lit l'artefact produit par l'étape d'entraînement (``model.tar.gz``), en
extrait les métriques mesurées sur la fenêtre de test **temporelle** (jamais vue
à l'entraînement) et publie un rapport conforme au schéma SageMaker
``regression_metrics`` — c'est ce fichier que la condition ``GateWAPE`` lit.

Volontairement sans dépendance (stdlib seulement) : le conteneur Processing
n'embarque pas LightGBM, et il n'en a pas besoin.

Montage SageMaker :

* entrée  : ``/opt/ml/processing/input/model/model.tar.gz``
* sortie  : ``/opt/ml/processing/output/evaluation.json``
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

INPUT = Path("/opt/ml/processing/input/model")
OUTPUT = Path("/opt/ml/processing/output")


def _read_json_from_tar(archive: Path, suffix: str) -> dict:
    """Extrait un membre JSON de l'archive sans la décompresser sur disque."""
    with tarfile.open(archive, mode="r:gz") as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith(suffix)), None)
        if member is None:
            raise FileNotFoundError(f"{suffix} absent de {archive.name}")
        return json.loads(tar.extractfile(member).read())


def _metric(value: float) -> dict:
    """Format attendu par SageMaker Model Quality (``value`` + écart-type)."""
    return {"value": round(float(value), 6), "standard_deviation": "NaN"}


def main() -> int:
    archive = next(INPUT.glob("*.tar.gz"), None)
    if archive is None:
        raise SystemExit(f"aucune archive model.tar.gz dans {INPUT}")

    summary = _read_json_from_tar(archive, "metrics.json")
    by_horizon = summary.get("by_horizon", {})
    wapes = [m["test_wape"] for m in by_horizon.values()] or [summary["test_wape"]]

    # Moyenne des horizons : c'est le chiffre qui décide de la mise en production,
    # pas le seul horizon 1 (un modèle peut être bon à J+1 et mauvais à J+28).
    mean_wape = sum(wapes) / len(wapes)
    best_h = min(by_horizon, key=lambda h: by_horizon[h]["test_wape"]) if by_horizon else "1"
    worst_h = max(by_horizon, key=lambda h: by_horizon[h]["test_wape"]) if by_horizon else "1"

    report = {
        "regression_metrics": {
            "test_wape": _metric(mean_wape),
            "baseline_wape": _metric(summary["baseline_wape"]),
            "improvement_vs_baseline": _metric(summary["improvement_vs_baseline"]),
        },
        "by_horizon": {
            h: {k: round(float(v), 6) for k, v in m.items()} for h, m in by_horizon.items()
        },
        "n_horizons": len(by_horizon) or 1,
        "best_horizon": best_h,
        "worst_horizon": worst_h,
        "training_job": summary.get("training_job", ""),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(
        f"[evaluate] WAPE moyen {mean_wape:.4f} "
        f"(meilleur h={best_h}, pire h={worst_h}, "
        f"baseline {summary['baseline_wape']:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
