"""Régressions du contrat entraînement / inférence, sans aucun appel AWS."""

import io
import json
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest

from infra.sagemaker import batch_transform as batch, evaluate_entry, inference, train_entry
from retail_forecast.data.simulate import simulate_sales
from retail_forecast.features.build import build_features, categorical_levels, feature_columns


def test_inference_preserves_codes_and_uses_target_calendar():
    sales = simulate_sales(n_stores=2, n_items=2, days=60, seed=4)
    levels = categorical_levels(sales)
    features = feature_columns(build_features(sales))
    subset = sales[(sales.store_id == "S02") & (sales.item_id == "I002")]
    rows = batch.build_inference_rows(subset, features, [1, 7], levels=levels)
    assert rows.store_code.tolist() == [1, 1]
    assert rows.item_code.tolist() == [1, 1]
    assert rows.target_dow.tolist() == rows.target_date.dt.dayofweek.tolist()
    assert rows.target_dow.nunique() == 2
    assert rows.date.nunique() == 1


def test_collect_output_reads_only_current_job():
    sales = simulate_sales(n_stores=1, n_items=1, days=45, seed=4)
    rows = batch.build_inference_rows(sales, feature_columns(build_features(sales)), [1])

    class FakeS3:
        def get_object(self, **kwargs):
            assert kwargs["Key"] == "forecast/raw/current/features.jsonl.out"
            return {"Body": io.BytesIO(b"[3.5]\n")}

        def put_object(self, **kwargs):
            self.saved = pd.read_parquet(io.BytesIO(kwargs["Body"]))

    fake = FakeS3()
    result = batch.collect_output(fake, rows, 2, "current")
    assert result.forecast_units.tolist() == [3.5]
    assert fake.saved.model_version.tolist() == ["v2"]


@pytest.mark.parametrize("horizon", [1.5, -1, float("nan")])
def test_serving_rejects_invalid_horizons(horizon):
    with pytest.raises(ValueError, match="entiers positifs"):
        inference.predict_fn({"x": 1, "horizon": horizon}, {"features": ["x"], "models": {}})


@pytest.mark.parametrize("profile", ["median_calendar_v1", "legacy_l2"])
@pytest.mark.parametrize("preprocessed", [False, True])
def test_training_artifact_independent_evaluation_and_serving(tmp_path, monkeypatch, profile, preprocessed):
    """Un vrai petit modèle traverse le même contrat que les jobs cloud."""
    sales = simulate_sales(n_stores=1, n_items=2, days=180, seed=4)
    channel = tmp_path / "input/data/training"
    channel.mkdir(parents=True)
    training_data = build_features(sales, dropna=False) if preprocessed else sales
    training_data.to_parquet(channel / "sales.parquet", index=False)
    monkeypatch.setattr(train_entry, "BASE", tmp_path)
    from pathlib import Path
    monkeypatch.setattr(train_entry, "CODE_DIR", Path(train_entry.__file__).parent)
    # Limite les threads pour que le smoke test reste léger sur CI.
    monkeypatch.setitem(train_entry.DEFAULT_PARAMS, "n_jobs", 1)
    monkeypatch.setattr(sys, "argv", ["train_entry.py", "--horizons", ",".join(map(str, range(1, 17))),
                                    "--n-estimators", "5", "--valid-days", "35",
                                    "--test-days", "14", "--model-profile", profile])
    assert train_entry.main() == 0
    model_dir = tmp_path / "model"
    assert (model_dir / "code/inference.py").read_bytes() == (train_entry.CODE_DIR / "inference.py").read_bytes()
    assert (model_dir / "code/requirements.txt").exists()
    metadata = json.loads((model_dir / "metadata.json").read_text())
    assert metadata["feature_profile"] == profile
    assert metadata["point_statistic"] == ("median" if profile == "median_calendar_v1" else "mean")
    if profile == "median_calendar_v1":
        assert metadata["params"]["objective"] == "quantile"
        assert metadata["params"]["alpha"] == 0.5
        assert "target_is_holiday" in metadata["features"]
    else:
        assert metadata["params"]["objective"] == "regression"
        assert "target_is_holiday" not in metadata["features"]
        # Un ancien artefact ne déclare aucun profil : l'évaluateur reste compatible.
        legacy_metadata = {k: v for k, v in metadata.items() if k not in ("feature_profile", "point_statistic")}
        (model_dir / "metadata.json").write_text(json.dumps(legacy_metadata))
    assert metadata["split"] == {"valid_days": 35, "test_days": 14}
    rows = batch.build_inference_rows(
        sales, metadata["features"], list(range(1, 17)), levels=metadata["categorical_levels"]
    )
    predictions = inference.predict_fn(rows, inference.model_fn(str(model_dir)))
    assert len(predictions) == 32
    assert np.isfinite(predictions).all()
    assert min(predictions) >= 0
    intervals = inference.predict_fn(rows.assign(return_quantiles=True), inference.model_fn(str(model_dir)))
    assert all(0 <= item["p10"] <= item["p90"] for item in intervals)
    assert [item["forecast_units"] for item in intervals] == predictions

    archive = tmp_path / "model.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in model_dir.iterdir():
            tar.add(path, arcname=path.name)
    code = tmp_path / "code"
    code.mkdir()
    with tarfile.open(code / "source.tar.gz", "w:gz"):
        pass  # Le paquet est déjà importable dans l'environnement de test.
    monkeypatch.setattr(evaluate_entry, "CURATED", channel)
    monkeypatch.setattr(evaluate_entry, "CODE", code)
    actual = evaluate_entry._recompute(archive)
    assert actual is not None
    for horizon in map(str, range(1, 17)):
        assert actual[horizon] == pytest.approx(metadata["metrics"][horizon]["test_wape"])
        assert evaluate_entry.INDEPENDENT_DETAILS[horizon]["coverage_80"] == pytest.approx(metadata["metrics"][horizon]["test_coverage_80"])
        assert evaluate_entry.INDEPENDENT_DETAILS[horizon]["rmsle"] == pytest.approx(metadata["metrics"][horizon]["test_rmsle"])


def test_llm_prompt_does_not_invent_quality_or_volatility(monkeypatch):
    from retail_forecast.agent import brief

    monkeypatch.setattr(brief, "_load_metrics", lambda: {})
    forecast = pd.DataFrame({"horizon": [1, 7], "store_id": ["S01", "S01"],
                             "item_id": ["I001", "I001"], "forecast_units": [3., 4.]})
    prompt = brief.build_prompt(forecast)
    assert "qualité non évaluée" in prompt
    assert "WAPE 0" not in prompt
    assert "Séries les plus volatiles" not in prompt
    monkeypatch.setattr(brief, "_ollama", lambda prompt, model, base_url: (model, base_url))
    assert brief.generate("p", "ollama", model="custom", base_url="http://local") == (
        "custom", "http://local"
    )


def test_lambda_gate_checks_all_horizons(monkeypatch):
    import importlib
    from unittest.mock import Mock, patch

    with patch("boto3.client", return_value=Mock()):
        gate = importlib.import_module("infra.lambda.register_model")
    monkeypatch.setattr(gate, "_sm", Mock())
    gate._sm.create_model_package.return_value = {"ModelPackageArn": "arn:test"}
    monkeypatch.setattr(gate, "_metrics", lambda bucket, job: {
        "test_wape": 0.3, "mean_wape": 0.6, "baseline_wape": 0.7
    })
    result = gate.lambda_handler({"bucket": "b", "training_job": "j", "threshold": 0.45})
    assert result["approved"] is False
    assert result["wape"] == 0.6
    # Un seuil strict de zéro doit rester zéro.
    assert gate.lambda_handler({"bucket": "b", "training_job": "j", "threshold": 0})[
        "threshold"
    ] == 0


def test_dashboard_reads_glue_partitions(monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from dataclasses import replace
    from pathlib import Path

    from streamlit.testing.v1 import AppTest
    from retail_forecast import config
    from retail_forecast.ingest import storage

    sales = simulate_sales(n_stores=1, n_items=1, days=45, seed=4)
    forecast = pd.DataFrame({"store_id": ["S01"], "item_id": ["I001"],
                             "horizon": [1], "forecast_units": [3.], "model_version": ["v1"],
                             "model_job": ["j"], "origin_date": [sales.date.max()],
                             "target_date": [sales.date.max() + pd.Timedelta(days=1)]})

    class FakeS3:
        def get_object(self, **kwargs):
            key = kwargs["Key"]
            if key == "deployment/active.json":
                return {"Body": io.BytesIO(b'{"forecast_key":"forecast/runs/test/forecast.parquet"}'), "ETag": "e"}
            if key == "reports/cloud_metrics.json":
                return {"Body": io.BytesIO(b'{}')}
            frame = forecast if key.startswith("forecast/") else sales
            if not key.startswith("forecast/"):
                assert key == "curated/sales/part-000.parquet"
            buf = io.BytesIO()
            frame.to_parquet(buf, index=False)
            return {"Body": io.BytesIO(buf.getvalue())}

        def get_paginator(self, name):
            return self

        def paginate(self, **kwargs):
            return [{"Contents": [{"Key": "curated/sales/part-000.parquet"}]}]

    monkeypatch.setattr(config, "SETTINGS", replace(config.SETTINGS, bucket="test", s3_endpoint_url=""))
    monkeypatch.setattr(storage, "s3_client", lambda: FakeS3())
    app = AppTest.from_file(str(Path(__file__).parents[1] / "dashboards/streamlit/app.py"))
    app.run(timeout=15)
    assert not app.exception
    assert app.title[0].value.endswith("J+1")


def test_local_mock_mode_reads_local_forecast_without_cloud_manifest(tmp_path, monkeypatch):
    from dataclasses import replace
    from retail_forecast.agent import brief
    from retail_forecast.ingest import storage
    from unittest.mock import Mock

    folder = tmp_path / "data/forecast"
    folder.mkdir(parents=True)
    pd.DataFrame({"forecast_units": [3.]}).to_parquet(folder / "forecast.parquet")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "local_baseline.json").write_text('{"test_wape":0.4}')
    (reports / "cloud_metrics.json").write_text('{"test_wape":0.9}')
    monkeypatch.setattr(brief, "ROOT", tmp_path)
    monkeypatch.setattr(brief, "SETTINGS", replace(brief.SETTINGS, bucket="mock", s3_endpoint_url="http://localhost:9090"))
    remote = Mock(side_effect=AssertionError("local mode must not need a cloud manifest"))
    monkeypatch.setattr(storage, "s3_client", remote)
    assert brief._load_forecast().forecast_units.tolist() == [3.]
    assert brief._load_metrics()["test_wape"] == .4
    remote.assert_not_called()


def test_langchain_chain_validates_model_output(monkeypatch):
    pytest.importorskip("langchain_core")
    from retail_forecast.agent import brief
    monkeypatch.setattr(brief, "_load_metrics", lambda: {})
    frame = pd.DataFrame({"horizon": [1], "store_id": ["S01"], "item_id": ["I001"], "forecast_units": [3.]})
    monkeypatch.setattr(brief, "generate", lambda *a, **kw: json.dumps({
        "analyse": json.loads(brief.prompt_inputs(frame)["faits_json"]),
        "actions": json.loads(brief.prompt_inputs(frame)["actions_json"])[:2]}))
    assert "3 unités" in brief.generate_brief(frame, "ollama")
    with pytest.raises(ValueError):
        brief.validate_brief(json.dumps({"analyse": ["99999 unités", "a", "b"], "actions": ["a", "b"]}), "3 unités")


def test_sagemaker_boolean_coercion_is_accepted(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_entry.py", "--quantiles", "False"])
    def parsed(channel):
        raise RuntimeError("parsing successful")
    monkeypatch.setattr(train_entry, "_read_channel", parsed)
    with pytest.raises(RuntimeError, match="parsing successful"): train_entry.main()


@pytest.mark.parametrize('bounds', [(1., 5.), (-1., 5.), (5., 1.), (1., float('inf'))])
def test_quantile_batch_output_is_validated_before_publication(bounds):
    from unittest.mock import Mock
    sales = simulate_sales(n_stores=1, n_items=1, days=45, seed=4)
    rows = batch.build_inference_rows(sales, feature_columns(build_features(sales)), [1])
    rows['return_quantiles'] = True
    s3 = Mock()
    s3.get_object.return_value = {'Body': io.BytesIO(json.dumps([{
        'forecast_units': 3., 'p10': bounds[0], 'p90': bounds[1]}]).encode())}
    if bounds == (1., 5.):
        result = batch.collect_output(s3, rows, 7, 'quantile-job')
        assert result.p10.iloc[0] == 1. and result.p90.iloc[0] == 5.
        s3.put_object.assert_called_once()
    else:
        with pytest.raises(ValueError): batch.collect_output(s3, rows, 7, 'quantile-job')
        s3.put_object.assert_not_called()


def test_llm_rejects_unmeasured_stability():
    from retail_forecast.agent.brief import validate_brief
    with pytest.raises(ValueError, match='non mesurée'):
        validate_brief(json.dumps({'analyse': ['Ventes stables', 'a', 'b'], 'actions': ['a', 'b']}), '')


def test_brief_rejects_invented_claim_using_only_known_numbers(monkeypatch):
    """Des chiffres connus ne valident pas une relation métier inventée."""
    from retail_forecast.agent import brief
    monkeypatch.setattr(brief, '_load_metrics', lambda: {'test_wape': .4, 'baseline_wape': .5})
    frame = pd.DataFrame({'horizon': [1], 'store_id': ['S01'], 'item_id': ['I001'], 'forecast_units': [3.]})
    inputs = brief.prompt_inputs(frame)
    facts = json.loads(inputs['faits_json'])
    facts[0] = 'Le WAPE de la série S01/I001 est 40.0%.'
    with pytest.raises(ValueError, match='non autorisée'):
        brief.validate_brief(json.dumps({'analyse': facts, 'actions': json.loads(inputs['actions_json'])[:2]}), brief.build_prompt(frame))
