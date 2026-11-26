"""Ablation L2/L1/médiane sur une réplication synthétique indépendante.

La sélection utilise exclusivement la validation. Le test final commun aux
16 horizons n'est évalué qu'après gel du choix. Aucun accès AWS ni publication.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from retail_forecast.data.simulate import simulate_sales
from retail_forecast.evaluate.metrics import wape, rmsle
from retail_forecast.features.build import (
    KEYS, TARGET, add_calendar_features, add_lag_features, add_target, feature_columns,
)
from retail_forecast.features.target_calendar import add_target_calendar
from retail_forecast.train.train_lgbm import DEFAULT_PARAMS, _fit

VARIANTS = {
    "l2_existing": ("regression", False),
    "l1_existing": ("regression_l1", False),
    "l2_calendar": ("regression", True),
    "l1_calendar": ("regression_l1", True),
    "median_calendar": ("quantile", True),
}


def temporal_parts(frame, valid_start, test_start):
    """Frontières communes en dates cibles, avec purge à la validation."""
    train = frame[frame.target_date < valid_start]
    valid = frame[(frame.date >= valid_start) & (frame.target_date < test_start)]
    test = frame[frame.target_date >= test_start]
    if any(part.empty for part in (train, valid, test)):
        raise ValueError("historique insuffisant pour les frontières fixées")
    return train, valid, test


def select_candidate(results):
    """Choisit la variante sur la moyenne des 16 WAPE de validation."""
    for name in VARIANTS:
        rows = [row for row in results if row["variant"] == name]
        if (len(rows) != 16 or {row["horizon"] for row in rows} != set(range(1, 17))
                or not all(np.isfinite(row["valid_wape"]) for row in rows)):
            raise ValueError("validation incomplète ou non finie")
    return min(VARIANTS, key=lambda name: np.mean([
        row["valid_wape"] for row in results if row["variant"] == name
    ]))


def run(args):
    """Exécute les cinq variantes, puis ouvre une seule fois le test réservé."""
    out = Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "seed": args.seed, "stores": 20, "items": 60, "days": 730,
        "trees": args.trees, "threads": args.threads, "early_stopping": 40,
        "variants": VARIANTS, "validation_days": 56, "test_days": 28,
        "selection": "mean_horizon_validation_wape", "version": 1,
    }
    protocol_path = out / "protocol.json"
    serialized = json.dumps(protocol, indent=2)
    if protocol_path.exists() and protocol_path.read_text() != serialized:
        raise ValueError("cache incompatible : utiliser un autre répertoire")
    protocol_path.write_text(serialized)
    sales = simulate_sales(seed=args.seed)
    data_path = out / "sales.parquet"
    sales.to_parquet(data_path, index=False)
    test_start = sales.date.max() - pd.Timedelta(days=27)
    valid_start = test_start - pd.Timedelta(days=56)
    base = add_lag_features(add_calendar_features(sales.sort_values([*KEYS, "date"]).reset_index(drop=True)))
    results = []
    for horizon in range(1, 17):
        frame = add_target(base, horizon)
        required = [c for c in frame if c.startswith(("lag_", "roll_"))]
        frame = frame.dropna(subset=[*required, TARGET, "seasonal_naive"])
        enriched = add_target_calendar(frame)
        for name, (objective, calendar) in VARIANTS.items():
            stem = out / f"{name}_h{horizon:02d}"
            checkpoint = stem.with_suffix(".json")
            if checkpoint.exists():
                results.append(json.loads(checkpoint.read_text()))
                continue
            started = time.monotonic()
            train, valid, test = temporal_parts(enriched if calendar else frame, valid_start, test_start)
            columns = feature_columns(train)
            params = {**DEFAULT_PARAMS, "objective": objective, "metric": "l1",
                      "n_estimators": args.trees, "n_jobs": args.threads,
                      "random_state": 42, "deterministic": True, "force_col_wise": True}
            if objective == "quantile":
                params["alpha"] = 0.5
            model = _fit(lgb.LGBMRegressor(**params), train[columns], train.y,
                         valid[columns], valid.y, 40)
            prediction = np.maximum(0, model.predict(valid[columns]))
            middle = valid_start + pd.Timedelta(days=28)
            blocks = valid.date < middle
            row = {"variant": name, "horizon": horizon,
                   "valid_wape": wape(valid.y, prediction),
                   "valid_blocks_wape": [wape(valid.y[mask], prediction[mask])
                                         for mask in (blocks, ~blocks)],
                   "n_train": len(train), "n_valid": len(valid), "n_test": len(test),
                   "best_iteration": model.best_iteration_,
                   "features": columns, "seconds": time.monotonic() - started}
            model.booster_.save_model(str(stem.with_suffix(".txt")))
            panel = test[[*KEYS, "date", "target_date", "y", "seasonal_naive"]].copy()
            panel["prediction"] = np.maximum(0, model.predict(test[columns]))
            panel.to_parquet(stem.with_suffix(".parquet"), index=False)
            checkpoint.write_text(json.dumps(row, indent=2))
            results.append(row)
            print(json.dumps({k: row[k] for k in ("variant", "horizon", "valid_wape", "seconds")}), flush=True)
        del frame, enriched
    selected = select_candidate(results)
    selection = {"selected": selected, "criterion": protocol["selection"],
                 "validation": {name: float(np.mean([r["valid_wape"] for r in results if r["variant"] == name]))
                                for name in VARIANTS}}
    (out / "selection.json").write_text(json.dumps(selection, indent=2))
    scores = {}
    for name in dict.fromkeys(["l2_existing", selected]):
        rows = []
        for horizon in range(1, 17):
            panel = pd.read_parquet(out / f"{name}_h{horizon:02d}.parquet")
            rows.append({"horizon": horizon, "wape": wape(panel.y, panel.prediction),
                         "rmsle": rmsle(panel.y, panel.prediction),
                         "seasonal_wape": wape(panel.y, panel.seasonal_naive), "rows": len(panel)})
        scores[name] = {"mean_wape": float(np.mean([r["wape"] for r in rows])),
                        "mean_rmsle": float(np.mean([r["rmsle"] for r in rows])), "by_horizon": rows}
    report = {"protocol": protocol, "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
              "lightgbm_version": lgb.__version__, "valid_start": str(valid_start.date()),
              "test_target_start": str(test_start.date()), "test_target_end": str(sales.date.max().date()),
              **selection, "test": scores, "runs": results,
              "limitations": ["independent synthetic replication, not the deployed v7 test",
                              "single seed; no AWS execution or promotion; point forecasts only"]}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps({"selected": selected, "test": scores}), flush=True)


def main():
    """Paramètres reproductibles du benchmark local complet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache", default="data/benchmark/median-calendar")
    parser.add_argument("--report", default="reports/median_calendar_benchmark.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
