"""Point d'entrée du Training Job SageMaker (conteneur SKLearn managé).

Ce script tourne **dans le conteneur** SageMaker, pas en local. Il reprend
exactement le même pipeline que ``retail_forecast.train.train_lgbm`` — mêmes
features, mêmes métriques — pour que les chiffres du local et ceux du cloud
soient comparables.

Convention de montage SageMaker :

* entrée  : ``/opt/ml/input/data/training/`` (les Parquet du canal ``training``)
* modèle  : ``/opt/ml/model/`` (archivé en ``model.tar.gz`` vers S3)
* sorties : ``/opt/ml/output/data/`` (journaux de métriques)

Usage (appelé par SageMaker, pas à la main) ::

    python train_entry.py --horizon 1 --n-estimators 600
"""

from __future__ import annotations

import argparse
import json
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
    """Lit tous les Parquet du canal ``name`` (ou son unique fichier)."""
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Training Job LightGBM (SageMaker).")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--n-estimators", type=int, default=DEFAULT_PARAMS["n_estimators"])
    ap.add_argument("--learning-rate", type=float, default=DEFAULT_PARAMS["learning_rate"])
    ap.add_argument("--num-leaves", type=int, default=DEFAULT_PARAMS["num_leaves"])
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--valid-days", type=int, default=56)
    ap.add_argument("--test-days", type=int, default=28)
    args = ap.parse_args()

    sales = _read_channel("training")
    feats = build_features(sales, horizon=args.horizon)
    train, valid, test = time_split(
        feats, valid_days=args.valid_days, test_days=args.test_days
    )
    X_tr, y_tr = xy(train)
    X_va, y_va = xy(valid)
    X_te, y_te = xy(test)
    print(
        f"[entry] horizon={args.horizon}j  train={len(train):,}"
        f"  valid={len(valid):,}  test={len(test):,}  features={X_tr.shape[1]}"
    )

    params = {
        **DEFAULT_PARAMS,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
    }
    model = _fit(
        lgb.LGBMRegressor(**params),
        X_tr,
        y_tr,
        X_va,
        y_va,
        args.early_stopping_rounds,
    )

    pred = np.clip(model.predict(X_te), 0, None)
    base = np.clip(seasonal_naive(test), 0, None)
    metrics = {
        "horizon": args.horizon,
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "n_test": int(len(test)),
        "n_features": int(X_tr.shape[1]),
        "best_iteration": int(model.best_iteration_ or args.n_estimators),
        "test_wape": wape(y_te, pred),
        "test_rmse": rmse(y_te, pred),
        "test_bias": bias(y_te, pred),
        "baseline_wape": wape(y_te, base),
        "baseline_rmse": rmse(y_te, base),
        "baseline_bias": bias(y_te, base),
    }
    metrics["improvement_vs_baseline"] = (
        metrics["baseline_wape"] - metrics["test_wape"]
    ) / metrics["baseline_wape"]

    # 1. modèle + métadonnées dans /opt/ml/model (-> model.tar.gz sur S3)
    model_dir = BASE / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_dir / "model.txt"))
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "framework": "lightgbm",
                "horizon": args.horizon,
                "features": list(X_tr.columns),
                "params": {k: v for k, v in params.items() if k != "n_jobs"},
                "metrics": metrics,
            },
            indent=2,
            default=float,
        )
    )
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    # 2. copie dans /opt/ml/output/data (-> output.tar.gz, lisible dans les logs)
    out_dir = BASE / "output" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    print("[entry] metriques " + json.dumps(metrics, default=float))
    print(f"[entry] WAPE {metrics['test_wape']:.4f} "
          f"(baseline {metrics['baseline_wape']:.4f}, "
          f"gain {metrics['improvement_vs_baseline']:+.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
