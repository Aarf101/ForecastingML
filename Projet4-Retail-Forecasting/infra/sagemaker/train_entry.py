"""Point d'entrée du Training Job SageMaker (conteneur SKLearn managé).

Ce script tourne **dans le conteneur** SageMaker, pas en local. Il reprend le
même pipeline que ``retail_forecast.train.train_lgbm`` — mêmes features, mêmes
métriques — pour que les chiffres du local et ceux du cloud soient comparables.

Un modèle **par horizon** est entraîné (stratégie directe multi-horizon : un
booster par h évite d'empiler les erreurs comme le ferait une prévision
récursive). Les features sont calculées une seule fois, seules les cibles
changent — c'est ce qui permet de tenir 4 horizons en quelques minutes.

Convention de montage SageMaker :

* entrée  : ``/opt/ml/input/data/training/`` (les Parquet du canal ``training``)
* modèle  : ``/opt/ml/model/`` (archivé en ``model.tar.gz`` vers S3)
* sorties : ``/opt/ml/output/data/`` (journaux de métriques)

Usage (appelé par SageMaker, pas à la main) ::

    python train_entry.py --horizons 1,7,14,28 --n-estimators 300
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

BASE = Path("/opt/ml")
CODE_DIR = Path(__file__).resolve().parent
# Le paquet retail_forecast est embarqué à côté de ce script dans sourcedir.tar.gz.
sys.path.insert(0, str(CODE_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from retail_forecast.evaluate.metrics import bias, rmse, wape  # noqa: E402
from retail_forecast.features.build import (  # noqa: E402
    TARGET,
    add_target,
    build_features,
    time_split,
    xy,
)
from retail_forecast.train.train_lgbm import (  # noqa: E402
    DEFAULT_PARAMS,
    _fit,
    seasonal_naive,
)

import lightgbm as lgb  # noqa: E402


def _read_channel(name: str) -> pd.DataFrame:
    """Lit tous les Parquet du canal ``name`` (ou ses CSV)."""
    root = BASE / "input" / "data" / name
    files = sorted(root.rglob("*.parquet")) or sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"aucun fichier .parquet/.csv dans {root}")
    frames = [
        pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f) for f in files
    ]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    print(f"[entry] canal {name} : {len(files)} fichier(s), {len(df):,} lignes")
    return df


def _dataset_for_horizon(base: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Recolle la cible ``t + horizon`` sur des features calculées une seule fois."""
    df = add_target(base.drop(columns=[TARGET]), horizon=horizon)
    feat_cols = [c for c in df.columns if c.startswith(("lag_", "roll_"))]
    return df.dropna(subset=[*feat_cols, TARGET]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Training Job LightGBM (SageMaker).")
    ap.add_argument("--horizons", default="1")
    ap.add_argument("--n-estimators", type=int, default=DEFAULT_PARAMS["n_estimators"])
    ap.add_argument("--learning-rate", type=float, default=DEFAULT_PARAMS["learning_rate"])
    ap.add_argument("--num-leaves", type=int, default=DEFAULT_PARAMS["num_leaves"])
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--valid-days", type=int, default=56)
    ap.add_argument("--test-days", type=int, default=28)
    args = ap.parse_args()
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    sales = _read_channel("training")
    # Une seule passe de features (le rolling coûte cher) ; la cible est recollée.
    base = build_features(sales, horizon=1, dropna=False)
    print(f"[entry] features calculées : {len(base):,} lignes, horizons {horizons}")

    params = {
        **DEFAULT_PARAMS,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
    }

    model_dir = BASE / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    by_horizon: dict[str, dict] = {}
    features: list[str] = []

    for horizon in horizons:
        data = _dataset_for_horizon(base, horizon)
        train, valid, test = time_split(
            data, valid_days=args.valid_days, test_days=args.test_days
        )
        X_tr, y_tr = xy(train)
        X_va, y_va = xy(valid)
        X_te, y_te = xy(test)
        features = list(X_tr.columns)

        model = _fit(
            lgb.LGBMRegressor(**params), X_tr, y_tr, X_va, y_va, args.early_stopping_rounds
        )
        pred = np.clip(model.predict(X_te), 0, None)
        base_pred = np.clip(seasonal_naive(test), 0, None)
        base_wape = wape(y_te, base_pred)

        metrics = {
            "horizon": horizon,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_features": len(features),
            "best_iteration": int(model.best_iteration_ or args.n_estimators),
            "test_wape": wape(y_te, pred),
            "test_rmse": rmse(y_te, pred),
            "test_bias": bias(y_te, pred),
            "baseline_wape": base_wape,
            "improvement_vs_baseline": (base_wape - wape(y_te, pred)) / base_wape,
        }
        by_horizon[str(horizon)] = metrics
        model.booster_.save_model(str(model_dir / f"model_{horizon}.txt"))
        print(
            f"[entry] h={horizon:>2}j  WAPE {metrics['test_wape']:.4f}"
            f" (baseline {base_wape:.4f}, gain {metrics['improvement_vs_baseline']:+.1%})"
        )

    # Le handler d'inférence doit voyager DANS l'artefact (Batch Transform).
    handler = CODE_DIR / "inference.py"
    if handler.exists():
        shutil.copy(handler, model_dir / "inference.py")
    # Dépendances d'inférence : le conteneur les installe au démarrage via
    # SAGEMAKER_REQUIREMENTS (LightGBM n'est pas dans l'image SKLearn).
    reqs = CODE_DIR / "requirements.txt"
    if reqs.exists():
        shutil.copy(reqs, model_dir / "requirements.txt")

    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "framework": "lightgbm",
                "horizons": horizons,
                "features": features,
                "params": {k: v for k, v in params.items() if k != "n_jobs"},
                "metrics": by_horizon,
            },
            indent=2,
            default=float,
        )
    )

    # Résumé à plat (horizon principal = le premier) pour le Model Registry.
    summary = {
        "horizons": horizons,
        "by_horizon": by_horizon,
        **by_horizon[str(horizons[0])],
    }
    (model_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    out_dir = BASE / "output" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))

    print(
        "[entry] resume "
        + json.dumps({h: round(m["test_wape"], 4) for h, m in by_horizon.items()})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
