"""Entraînement local du modèle global LightGBM (miroir du Training Job SageMaker).

Ce script est la version « local » du même entraînement que celui exécuté dans
SageMaker : mêmes features (``retail_forecast.features.build``), mêmes
métriques (``retail_forecast.evaluate.metrics``), même cible. Il sert à
valider le pipeline avant de dépenser du budget Learner Lab.

Usage::

    python -m retail_forecast.train.train_lgbm --data data/raw/sales.parquet
    python -m retail_forecast.train.train_lgbm --sample 0.2      # smoke test
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from retail_forecast.evaluate.metrics import bias, rmse, wape
from retail_forecast.features.build import build_features, time_split, xy

DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "l1",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_estimators": 600,
    "n_jobs": -1,
    "verbose": -1,
}

# LightGBM >= 4.6 déprécie `eval_set` au profit de `eval_X` / `eval_y`.
_FIT_PARAMS = inspect.signature(lgb.LGBMRegressor.fit).parameters


def _fit(model: lgb.LGBMRegressor, X_tr, y_tr, X_va, y_va, rounds: int):
    """``fit`` compatible 4.3 (SageMaker) et 4.6+ (local), avec early stopping."""
    callbacks = [lgb.early_stopping(rounds, verbose=False)]
    if "eval_X" in _FIT_PARAMS:
        model.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va, callbacks=callbacks)
    else:  # pragma: no cover - containers SageMaker plus anciens
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=callbacks)
    return model


def load_sales(path: str | Path) -> pd.DataFrame:
    """Charge un CSV ou Parquet de ventes."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable — lancez d'abord :\n"
            f"  python -m retail_forecast.data.simulate --out {path}"
        )
    df = pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def sample_series(df: pd.DataFrame, frac: float, seed: int = 42) -> pd.DataFrame:
    """Sous-échantillonne des *séries* (pas des lignes) pour un run rapide."""
    if frac >= 1.0:
        return df
    series = df[["store_id", "item_id"]].drop_duplicates()
    keep = series.sample(frac=frac, random_state=seed)
    return df.merge(keep, on=["store_id", "item_id"], how="inner")


def seasonal_naive(df: pd.DataFrame) -> np.ndarray:
    """Baseline : ventes du même jour de la semaine précédente (lag 7)."""
    return df["lag_7"].to_numpy(dtype=float)


def train_local(
    sales: pd.DataFrame,
    horizon: int = 1,
    params: dict | None = None,
    early_stopping_rounds: int = 50,
) -> dict:
    """Entraîne le modèle global et compare à la baseline saisonnière."""
    feats = build_features(sales, horizon=horizon)
    train, valid, test = time_split(feats)

    X_tr, y_tr = xy(train)
    X_va, y_va = xy(valid)
    X_te, y_te = xy(test)

    p = {**DEFAULT_PARAMS, **(params or {})}
    model = _fit(lgb.LGBMRegressor(**p), X_tr, y_tr, X_va, y_va, early_stopping_rounds)

    pred = np.clip(model.predict(X_te), 0, None)
    base = np.clip(seasonal_naive(test), 0, None)

    model_wape, base_wape = wape(y_te, pred), wape(y_te, base)
    return {
        "horizon": horizon,
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "n_test": int(len(test)),
        "best_iteration": int(model.best_iteration_ or p["n_estimators"]),
        "test_wape": model_wape,
        "test_rmse": rmse(y_te, pred),
        "test_bias": bias(y_te, pred),
        "baseline_wape": base_wape,
        "baseline_rmse": rmse(y_te, base),
        "baseline_bias": bias(y_te, base),
        "improvement_vs_baseline": (base_wape - model_wape) / base_wape,
        "feature_importance": (
            pd.Series(model.feature_importances_, index=X_tr.columns)
            .sort_values(ascending=False)
            .head(10)
            .round(1)
            .to_dict()
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entraînement LightGBM local.")
    parser.add_argument("--data", default="data/raw/sales.parquet")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--sample", type=float, default=1.0, help="fraction de séries")
    parser.add_argument("--estimators", type=int, default=DEFAULT_PARAMS["n_estimators"])
    parser.add_argument("--out", default="reports/local_baseline.json")
    args = parser.parse_args(argv)

    sales = sample_series(load_sales(args.data), args.sample)
    res = train_local(
        sales, horizon=args.horizon, params={"n_estimators": args.estimators}
    )
    n_series = len(sales[["store_id", "item_id"]].drop_duplicates())

    print(f"\n[local] horizon={res['horizon']}j  series={n_series}"
          f"  train={res['n_train']:,} lignes")
    print(f"[local] WAPE  modèle  : {res['test_wape']:.4f}")
    print(f"[local] WAPE  baseline: {res['baseline_wape']:.4f}"
          f"  (gain {res['improvement_vs_baseline']:+.1%})")
    print(f"[local] RMSE  modèle  : {res['test_rmse']:.3f}"
          f"   baseline: {res['baseline_rmse']:.3f}")
    print(f"[local] biais modèle  : {res['test_bias']:+.1%}"
          f"   baseline: {res['baseline_bias']:+.1%}")
    print(f"[local] best_iter={res['best_iteration']}")
    print("[local] top features :")
    for name, score in res["feature_importance"].items():
        print(f"         {name:<16} {score:>8.1f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[local] rapport -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
