"""Étape d'évaluation du SageMaker Pipeline (conteneur Processing).

Ce script tourne **dans un job Processing**, pas en local. Il fait deux choses,
dans cet ordre :

1. **Évaluation indépendante** — il recharge les boosters ``model_<h>.txt`` de
   l'artefact, prédit sur la fenêtre de test (28 derniers jours, jamais vue à
   l'entraînement) et recalcule le WAPE. C'est la seule façon de vraiment
   *évaluer* : recopier les métriques de l'entraîneur ne prouverait rien. Les
   features proviennent de l'étape ``PreprocessFeatures`` (même table que
   l'entraînement) ; sur un artefact plus ancien, il les recalcule depuis
   ``curated/``.
2. **Repli explicite** — si les dépendances (LightGBM, pyarrow) ou le paquet
   ``retail_forecast`` sont indisponibles, il republie les métriques de
   l'artefact en marquant ``"independent": false``. Jamais de chiffre inventé,
   jamais d'étape muette.

Montage SageMaker :

* entrée  : ``/opt/ml/processing/input/model/model.tar.gz``
* entrée  : ``/opt/ml/processing/input/curated/`` (features précalculées, ou
  Parquet de ``curated/sales/`` pour un artefact antérieur)
* entrée  : ``/opt/ml/processing/input/code/sourcedir.tar.gz`` (paquet du projet)
* sortie  : ``/opt/ml/processing/output/evaluation.json``
"""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

INPUT = Path("/opt/ml/processing/input/model")
CURATED = Path("/opt/ml/processing/input/curated")
CODE = Path("/opt/ml/processing/input/code")
OUTPUT = Path("/opt/ml/processing/output")

TEST_DAYS = 28
VALID_DAYS = 56

# Colonnes marqueurs de l'étape PreprocessFeatures (cf. preprocess_entry.py).
PREPROCESSED = ("lag_1", "roll_mean_7")


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


def _install(packages: list[str]) -> bool:
    """Installe les dépendances manquantes (le conteneur SKLearn n'a pas LightGBM)."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *packages],
            check=True,
        )
        print(f"[evaluate] dépendances installées : {' '.join(packages)}")
        return True
    except Exception as exc:  # noqa: BLE001 - on ne veut jamais planter ici
        print(f"[evaluate] installation impossible : {exc}")
        return False


def _extract(tar_path: Path, dest: Path) -> None:
    with tarfile.open(tar_path, mode="r:gz") as tar:
        tar.extractall(dest)


def _recompute(archive: Path) -> dict | None:
    """Recalcule le WAPE de test depuis ``curated/`` et l'artefact.

    Renvoie ``{horizon: wape}`` ou ``None`` si le recalcul n'est pas possible
    (pas de données montées, dépendance absente). Ne lève jamais : l'étape doit
    rester utilisable même dégradée.
    """
    try:
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
        import lightgbm as lgb  # noqa: F401
    except ImportError:
        if not _install(["lightgbm", "pyarrow"]):
            return None
        try:
            import lightgbm as lgb  # noqa: F401, F811
        except ImportError:
            return None

    fichiers = sorted(CURATED.rglob("*.parquet"))
    if not fichiers:
        print(f"[evaluate] aucun Parquet dans {CURATED} : recalcul impossible")
        return None

    # 1) paquet du projet (features + métriques) : fourni par le pipeline
    source = next(CODE.glob("*.tar.gz"), None)
    if source is None:
        print(f"[evaluate] paquet de code absent de {CODE} : recalcul impossible")
        return None
    tmp = Path(tempfile.mkdtemp(prefix="pkg-"))
    _extract(source, tmp)
    sys.path.insert(0, str(tmp))

    try:
        import numpy as np
        import pandas as pd
        import lightgbm as lgb

        from retail_forecast.evaluate.metrics import wape
        from retail_forecast.features.build import (
            TARGET,
            add_target,
            build_features,
            time_split,
        )

        model_dir = tmp / "model"
        model_dir.mkdir(exist_ok=True)
        _extract(archive, model_dir)
        metadata = json.loads((model_dir / "metadata.json").read_text())
        features = metadata["features"]

        frames = [pd.read_parquet(f) for f in fichiers]
        sales = pd.concat(frames, ignore_index=True)
        sales["date"] = pd.to_datetime(sales["date"])
        # Même détection que train_entry : dans le pipeline, l'étape
        # PreprocessFeatures a déjà écrit les lags — recalculer ici donnerait la
        # même table pour un coût inutile, et masquerait une désynchronisation.
        if all(c in sales.columns for c in PREPROCESSED):
            base = sales if TARGET in sales.columns else add_target(sales, horizon=1)
            print("[evaluate] features précalculées relues (pas de recalcul)")
        else:
            base = build_features(sales, horizon=1, dropna=False)
            print("[evaluate] features recalculées depuis curated/")

        resultat: dict[str, float] = {}
        for horizon in metadata["horizons"]:
            df = add_target(base.drop(columns=[TARGET]), horizon=horizon)
            df = df.dropna(subset=[*features, TARGET])
            _train, _valid, test = time_split(
                df, valid_days=VALID_DAYS, test_days=TEST_DAYS
            )
            booster = lgb.Booster(model_file=str(model_dir / f"model_{horizon}.txt"))
            pred = np.clip(booster.predict(test[features]), 0.0, None)
            y = test[TARGET].to_numpy(dtype=float)
            resultat[str(horizon)] = float(wape(y, pred))
            print(f"[evaluate] h={horizon:>2}j  WAPE recalculé {resultat[str(horizon)]:.4f}")
        return resultat
    except Exception as exc:  # noqa: BLE001 - dégradation explicite
        print(f"[evaluate] recalcul impossible ({type(exc).__name__}: {exc})")
        return None


def main() -> int:
    archive = next(INPUT.glob("*.tar.gz"), None)
    if archive is None:
        raise SystemExit(f"aucune archive model.tar.gz dans {INPUT}")

    summary = _read_json_from_tar(archive, "metrics.json")
    artefact = {
        h: float(m["test_wape"]) for h, m in summary.get("by_horizon", {}).items()
    } or {str(summary.get("horizon", 1)): float(summary["test_wape"])}

    recalcule = _recompute(archive)
    source = "recompute" if recalcule else "artifact"
    wapes = recalcule or artefact
    valeurs = list(wapes.values())
    mean_wape = sum(valeurs) / len(valeurs)

    best_h = min(wapes, key=lambda h: wapes[h])
    worst_h = max(wapes, key=lambda h: wapes[h])

    # Comparaison artefact / recalcul : c'est la preuve que l'évaluation est
    # indépendante (ou, à défaut, que le repli est assumé).
    if recalcule:
        ecarts = {
            h: round(recalcule[h] - artefact[h], 6)
            for h in recalcule
            if h in artefact
        }
        print(f"[evaluate] écart artefact/recalcul : {ecarts}")

    report = {
        "regression_metrics": {
            "test_wape": _metric(mean_wape),
            "baseline_wape": _metric(summary["baseline_wape"]),
            "improvement_vs_baseline": _metric(summary["improvement_vs_baseline"]),
        },
        "by_horizon": {
            h: {
                "wape": round(w, 6),
                "wape_artefact": round(artefact[h], 6) if h in artefact else None,
            }
            for h, w in wapes.items()
        },
        "n_horizons": len(wapes),
        "best_horizon": best_h,
        "worst_horizon": worst_h,
        "independent": bool(recalcule),
        "source": source,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(
        f"[evaluate] WAPE moyen {mean_wape:.4f} ({source}) "
        f"· meilleur h={best_h} · pire h={worst_h} "
        f"· baseline {summary['baseline_wape']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
