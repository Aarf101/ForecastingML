"""Tests du tuning SageMaker et de l'étape d'évaluation du pipeline.

Aucun appel AWS : on teste la *requête* (fonction pure) et le script d'évaluation
sur des fichiers temporaires.
"""

from __future__ import annotations

import io
import json
import os
import re
import tarfile

import pytest

from infra.sagemaker import evaluate_entry, tuning


def _fake_model_tar(tmp_path, summary: dict):
    """Fabrique un model.tar.gz minimal tel que le produit train_entry.py."""
    archive = tmp_path / "model.tar.gz"
    payload = json.dumps(summary).encode()
    with tarfile.open(archive, mode="w:gz") as tar:
        info = tarfile.TarInfo("metrics.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return archive


SUMMARY = {
    "horizons": [1, 28],
    "test_wape": 0.43,
    "baseline_wape": 0.57,
    "improvement_vs_baseline": 0.2456,
    "by_horizon": {
        "1": {"test_wape": 0.39, "baseline_wape": 0.56},
        "28": {"test_wape": 0.47, "baseline_wape": 0.58},
    },
}


def test_tuning_request_structure():
    """La requête cible le WAPE de validation en minimisation, pas le test."""
    req = tuning.tuning_request("bucket-test", "hpo-1", "s3://bucket-test/code/x.gz",
                                max_jobs=4, parallel=1, strategy="Random")
    config = req["HyperParameterTuningJobConfig"]
    assert config["HyperParameterTuningJobObjective"] == {
        "Type": "Minimize",
        "MetricName": "valid_wape",
    }
    # Garde-fou méthodologique : optimiser sur le test serait une fuite.
    assert config["HyperParameterTuningJobObjective"]["MetricName"] != "test_wape"
    assert config["ResourceLimits"] == {
        "MaxNumberOfTrainingJobs": 4,
        "MaxParallelTrainingJobs": 1,
    }
    noms = {p["Name"] for p in config["ParameterRanges"]["ContinuousParameterRanges"]}
    noms |= {p["Name"] for p in config["ParameterRanges"]["IntegerParameterRanges"]}
    assert noms == {"learning-rate", "num-leaves", "n-estimators"}


def test_tuning_request_reads_same_code_and_data_as_launch():
    """Même image, même archive de code, même préfixe de données que launch.py."""
    from infra.sagemaker.launch import IMAGE

    req = tuning.tuning_request("bucket-test", "hpo-1", "s3://bucket-test/code/x.gz",
                                max_jobs=2, parallel=1, strategy="Bayesian")
    job = req["TrainingJobDefinition"]
    assert job["AlgorithmSpecification"]["TrainingImage"] == IMAGE
    assert job["StaticHyperParameters"]["sagemaker_submit_directory"] == (
        "s3://bucket-test/code/x.gz"
    )
    assert job["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] == (
        "s3://bucket-test/curated/"
    )
    assert job["OutputDataConfig"]["S3OutputPath"] == "s3://bucket-test/tuning/"


def test_metric_regex_matches_entry_log_line():
    """Le regex du tuning doit lire la ligne émise par train_entry.py."""
    ligne = "[hpo] valid_wape=0.429813"
    match = re.search(tuning.METRIC_REGEX, ligne)
    assert match is not None
    assert abs(float(match.group(1)) - 0.429813) < 1e-9
    # La ligne de test ne doit pas être confondue avec l'objectif.
    assert re.search(tuning.METRIC_REGEX, "[hpo] test_wape=0.390000") is None


def test_evaluate_entry_writes_regression_metrics(tmp_path, monkeypatch):
    """L'étape d'évaluation agrège les horizons et publie le schéma attendu."""
    entree = tmp_path / "in"
    entree.mkdir()
    _fake_model_tar(entree, SUMMARY)
    sortie = tmp_path / "out"
    monkeypatch.setattr(evaluate_entry, "INPUT", entree)
    monkeypatch.setattr(evaluate_entry, "OUTPUT", sortie)
    monkeypatch.setattr(evaluate_entry, "_recompute", lambda archive: {"1": 0.39, "28": 0.47})

    assert evaluate_entry.main() == 0
    rapport = json.loads((sortie / "evaluation.json").read_text())

    # Moyenne des deux horizons = le chiffre que lit la condition GateWAPE.
    wape = rapport["regression_metrics"]["test_wape"]["value"]
    assert abs(wape - (0.39 + 0.47) / 2) < 1e-6
    assert rapport["best_horizon"] == "1"
    assert rapport["worst_horizon"] == "28"
    assert rapport["n_horizons"] == 2


def test_evaluate_entry_fails_without_archive(tmp_path, monkeypatch):
    """Sans model.tar.gz, l'étape doit échouer clairement (pas un JSON vide)."""
    entree = tmp_path / "vide"
    entree.mkdir()
    monkeypatch.setattr(evaluate_entry, "INPUT", entree)
    monkeypatch.setattr(evaluate_entry, "OUTPUT", tmp_path / "out")
    with pytest.raises(SystemExit):
        evaluate_entry.main()


def test_evaluate_entry_blocks_artifact_fallback(tmp_path, monkeypatch):
    """Sans données montées, aucune approbation automatique n'est possible."""
    entree = tmp_path / "in"
    entree.mkdir()
    _fake_model_tar(entree, SUMMARY)
    monkeypatch.setattr(evaluate_entry, "INPUT", entree)
    monkeypatch.setattr(evaluate_entry, "CURATED", tmp_path / "pas-de-donnees")
    monkeypatch.setattr(evaluate_entry, "CODE", tmp_path / "pas-de-code")
    monkeypatch.setattr(evaluate_entry, "OUTPUT", tmp_path / "out")

    with pytest.raises(SystemExit, match="approbation bloquée"):
        evaluate_entry.main()
    assert not (tmp_path / "out" / "evaluation.json").exists()


@pytest.mark.parametrize("values", [{"1": 0.39}, {"1": float("nan"), "28": 0.4}])
def test_evaluate_blocks_partial_or_invalid_metrics(tmp_path, monkeypatch, values):
    _fake_model_tar(tmp_path, SUMMARY)
    monkeypatch.setattr(evaluate_entry, "INPUT", tmp_path)
    monkeypatch.setattr(evaluate_entry, "_recompute", lambda archive: values)
    monkeypatch.setattr(evaluate_entry, "OUTPUT", tmp_path / "out")
    with pytest.raises(SystemExit, match="approbation bloquée"):
        evaluate_entry.main()


def test_gate_decision_uses_mean_wape_and_falls_back():
    """Les deux gates (Lambda et pipeline) comparent la même grandeur."""
    from infra.sagemaker.register import gate_decision

    # mean_wape prioritaire : 0,44 <= 0,45 alors que l'horizon 1 ferait pareil,
    # mais 0,50 > 0,45 doit bloquer malgré un h=1 flatteur.
    wape, approuve, statut = gate_decision(
        {"mean_wape": 0.50, "test_wape": 0.39}, 0.45
    )
    assert (wape, approuve, statut) == (0.50, False, "PendingManualApproval")

    # artefact ancien, sans mean_wape -> repli sur test_wape
    wape, approuve, statut = gate_decision({"test_wape": 0.43}, 0.45)
    assert (wape, approuve, statut) == (0.43, True, "Approved")


def test_boto_session_falls_back_when_profile_missing(monkeypatch):
    """Sur le notebook du lab il n'y a pas de profil AWS : repli sur le rôle IAM."""
    pytest.importorskip("sagemaker")
    from dataclasses import replace

    from infra.sagemaker import pipeline as pl

    monkeypatch.setenv("AWS_PROFILE", "profil-inexistant-lab")
    monkeypatch.setattr(pl, "SETTINGS", replace(pl.SETTINGS, profile="profil-inexistant-lab"))

    session = pl._boto_session()

    assert session.region_name == pl.SETTINGS.region
    # La variable doit disparaître, sinon botocore la relit et re-échoue.
    assert "AWS_PROFILE" not in os.environ


def test_pipeline_definition_graph(monkeypatch):
    """Le graphe du pipeline : features -> train -> eval -> condition -> registre.

    Les deux téléversements (code d'entraînement, script d'évaluation) sont
    simulés : la définition se construit donc sans AWS.
    """
    pytest.importorskip("sagemaker")
    from unittest import mock

    import sagemaker.processing as sp
    from sagemaker.estimator import EstimatorBase, UploadedCode

    from infra.sagemaker import pipeline as pl

    from dataclasses import replace
    monkeypatch.setattr(pl, "SETTINGS", replace(
        pl.SETTINGS, role_arn="arn:aws:iam::123456789012:role/test", bucket="b", profile=""
    ))

    fake_code = UploadedCode(
        s3_prefix="s3://b/code/sourcedir.tar.gz", script_name="train_entry.py"
    )
    with mock.patch(
        "infra.sagemaker.launch.upload_code",
        lambda client, bucket: "s3://b/code/sourcedir.tar.gz",
    ), mock.patch(
        "infra.sagemaker.pipeline.s3_client", lambda: None
    ), mock.patch.object(
        sp.ScriptProcessor, "_upload_code",
        lambda self, code, kms_key=None: "s3://b/code/evaluate_entry.py",
    ), mock.patch.object(
        EstimatorBase, "_stage_user_code_in_s3", lambda self, *a, **k: fake_code
    ):
        definition = json.loads(pl.build_pipeline(0.45).definition())

    assert [s["Name"] for s in definition["Steps"]] == [
        "PreprocessFeatures", "TrainLightGBM", "EvaluateModel", "GateWAPE",
    ]
    for step in definition["Steps"]:
        if step["Type"] == "Processing":
            paths = [i["S3Input"]["LocalPath"] for i in step["Arguments"]["ProcessingInputs"]]
            assert len(paths) == len(set(paths))
    gate = next(s for s in definition["Steps"] if s["Type"] == "Condition")
    assert [b["Name"] for b in gate["Arguments"]["IfSteps"]] == [
        "RegisterApproved-RegisterModel"
    ]
    assert [b["Name"] for b in gate["Arguments"]["ElseSteps"]] == ["WAPETropEleve"]
    assert [p["Name"] for p in definition["Parameters"]] == ["WapeThreshold"]
    metadata = gate["Arguments"]["IfSteps"][0]["Arguments"]["CustomerMetadataProperties"]
    for value in metadata.values():
        if isinstance(value, str):
            assert re.fullmatch(r"[A-Za-z0-9_.:/=+@ -]{1,256}", value)
    condition = json.dumps(gate["Arguments"]["Conditions"])
    assert "regression_metrics.test_wape.value" in condition

    # L'entraînement lit les features de l'étape précédente, pas curated/ brut :
    # c'est ce qui garantit une seule définition des features (pas de skew).
    train = next(s for s in definition["Steps"] if s["Name"] == "TrainLightGBM")
    assert train["DependsOn"] == ["PreprocessFeatures"]
    entree = json.dumps(train["Arguments"].get("InputDataConfig", []))
    # Le SDK sérialise la propriété soit en {"Get": …} soit en {"Std:Join": …}
    # selon qu'elle est seule ou concaténée : seul le renvoi à l'étape compte.
    assert "PreprocessFeatures" in entree and "S3Output.S3Uri" in entree
    assert "/curated/" not in entree

    # La sortie de l'étape est versionnée par exécution : deux exécutions
    # concurrentes n'écrasent pas leurs features respectives.
    pre = next(s for s in definition["Steps"] if s["Name"] == "PreprocessFeatures")
    sortie = json.dumps(
        pre["Arguments"]["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
    )
    assert "Std:Join" in sortie and "Execution.PipelineExecutionId" in sortie


def test_preprocess_entry_writes_feature_table(tmp_path, monkeypatch):
    """L'étape Preprocess écrit une table de features complète (lags inclus)."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    from infra.sagemaker import preprocess_entry as pp
    from retail_forecast.data.simulate import simulate_sales

    ventes = simulate_sales(n_stores=2, n_items=2, days=50, seed=11)
    entree = tmp_path / "curated"
    entree.mkdir()
    ventes.to_parquet(entree / "part-0.parquet", index=False)

    monkeypatch.setattr(pp, "INPUT", entree)
    monkeypatch.setattr(pp, "CODE", tmp_path / "code")
    monkeypatch.setattr(pp, "OUTPUT", tmp_path / "out")
    # Le paquet est importable dans l'environnement de test : pas besoin du tar.
    monkeypatch.setattr(pp, "_load_package", lambda: None)

    assert pp.main() == 0

    sortie = pd.read_parquet(tmp_path / "out" / "features" / "features.parquet")
    assert len(sortie) == len(ventes)          # dropna=False : aucune ligne perdue
    assert {"lag_1", "lag_7", "lag_35", "roll_mean_7", "roll_std_28"} <= set(sortie.columns)
    assert sortie["date"].dtype.kind == "M"

    rapport = json.loads(
        (tmp_path / "out" / "features" / "features.json").read_text()
    )
    assert rapport["rows"] == len(ventes)
    assert rapport["stores"] == 2
    assert "lag_35" in rapport["feature_columns"]
