"""Diagnostic secondaire du candidat figé sur l'ancien test déjà consulté.

Ce diagnostic ne sélectionne aucun modèle et ne constitue pas un test inédit.
Il compare le candidat au véritable artefact v7 sur les mêmes lignes.
"""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from retail_forecast.evaluate.benchmark_chronos import evaluation_panel
from retail_forecast.evaluate.benchmark_median import VARIANTS
from retail_forecast.evaluate.metrics import wape, rmsle
from retail_forecast.features.build import (
    KEYS, TARGET, add_calendar_features, add_lag_features, add_target, feature_columns, time_split,
)
from retail_forecast.features.target_calendar import add_target_calendar
from retail_forecast.train.train_lgbm import _fit


def main():
    """Entraîne uniquement la variante déjà retenue sur la réplication fraîche."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-report", default="reports/median_calendar_benchmark.json")
    parser.add_argument("--data", default="data/raw/sales.parquet")
    parser.add_argument("--archive", default="data/benchmark/active-model.tar.gz")
    parser.add_argument("--cache", default="data/benchmark/median-project")
    parser.add_argument("--report", default="reports/median_project_diagnostic.json")
    args = parser.parse_args()
    selection_bytes = Path(args.selection_report).read_bytes()
    selection = json.loads(selection_bytes)
    selected = selection["selected"]
    objective, calendar = VARIANTS[selected]
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive) as archive:
        metadata = json.load(archive.extractfile("metadata.json"))
    protocol = {"selected": selected, "selection_report_sha256": hashlib.sha256(selection_bytes).hexdigest(),
                "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                "reference_artifact_sha256": hashlib.sha256(Path(args.archive).read_bytes()).hexdigest(),
                "scope": "reused_test_descriptive_only_not_model_selection"}
    protocol_path = cache / "protocol.json"
    serialized = json.dumps(protocol, indent=2)
    if protocol_path.exists() and protocol_path.read_text() != serialized:
        raise ValueError("cache incompatible")
    protocol_path.write_text(serialized)
    sales = pd.read_parquet(args.data)
    reference_path = cache / "reference.parquet"
    if not reference_path.exists():
        evaluation_panel(sales, Path(args.archive)).to_parquet(reference_path, index=False)
    reference = pd.read_parquet(reference_path)
    base = add_lag_features(add_calendar_features(sales.sort_values([*KEYS, "date"]).reset_index(drop=True),
                                               levels=metadata["categorical_levels"]))
    params = {**metadata["params"], "objective": objective, "n_jobs": 4,
              "deterministic": True, "force_col_wise": True}
    if objective == "quantile":
        params["alpha"] = 0.5
    results = []
    for horizon in range(1, 17):
        checkpoint = cache / f"h{horizon:02d}.json"
        if checkpoint.exists():
            results.append(json.loads(checkpoint.read_text()))
            continue
        frame = add_target(base, horizon)
        frame = frame.dropna(subset=[*metadata["features"], TARGET, "seasonal_naive"])
        if calendar:
            frame = add_target_calendar(frame)
        train, valid, test = time_split(frame, **metadata["split"])
        columns = feature_columns(frame)
        model = _fit(lgb.LGBMRegressor(**params), train[columns], train.y, valid[columns], valid.y, 40)
        model.booster_.save_model(str(cache / f"model_{horizon}.txt"))
        panel = test[[*KEYS, "target_date", "y"]].copy()
        panel["candidate"] = np.maximum(0, model.predict(test[columns]))
        expected = reference[reference.horizon == horizon]
        paired = panel.merge(expected, on=[*KEYS, "target_date"], validate="one_to_one", suffixes=("", "_reference"))
        if len(paired) != len(panel) or len(paired) != len(expected) or not np.array_equal(paired.y, paired.y_reference):
            raise ValueError("lignes de référence non appariées")
        row = {"horizon": horizon, "rows": len(paired), "best_iteration": model.best_iteration_,
               "candidate_wape": wape(paired.y, paired.candidate),
               "v7_wape": wape(paired.y, paired.lightgbm),
               "candidate_rmsle": rmsle(paired.y, paired.candidate),
               "v7_rmsle": rmsle(paired.y, paired.lightgbm)}
        paired.to_parquet(cache / f"predictions_{horizon}.parquet", index=False)
        checkpoint.write_text(json.dumps(row, indent=2))
        results.append(row)
        print(json.dumps(row), flush=True)
    report = {**protocol, "params": params, "by_horizon": results,
              **{key: float(np.mean([r[key] for r in results]))
                 for key in ("candidate_wape", "v7_wape", "candidate_rmsle", "v7_rmsle")}}
    Path(args.report).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
