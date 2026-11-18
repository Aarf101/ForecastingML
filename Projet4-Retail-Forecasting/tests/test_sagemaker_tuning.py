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
    """La requête cible le WAPE en minimisation avec les bonnes paires de paramètres."""
    req = tuning.tuning_request("bucket-test", "hpo-1", "s3://bucket-test/code/x.gz",
                                max_jobs=4, parallel=1, strategy="Random")
    config = req["HyperParameterTuningJobConfig"]
    assert config["HyperParameterTuningJobObjective"] == {
        "Type": "Minimize",
        "MetricName": "test_wape",
    }
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
    ligne = "[hpo] test_wape=0.429813"
    match = re.search(tuning.METRIC_REGEX, ligne)
    assert match is not None
    assert abs(float(match.group(1)) - 0.429813) < 1e-9


def test_evaluate_entry_writes_regression_metrics(tmp_path, monkeypatch):
    """L'étape d'évaluation agrège les horizons et publie le schéma attendu."""
    entree = tmp_path / "in"
    entree.mkdir()
    _fake_model_tar(entree, SUMMARY)
    sortie = tmp_path / "out"
    monkeypatch.setattr(evaluate_entry, "INPUT", entree)
    monkeypatch.setattr(evaluate_entry, "OUTPUT", sortie)

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


def test_pipeline_definition_graph():
    """Le graphe du pipeline : train -> eval -> condition -> registre / échec.

    Les deux téléversements (code d'entraînement, script d'évaluation) sont
    simulés : la définition se construit donc sans AWS.
    """
    pytest.importorskip("sagemaker")
    from unittest import mock

    import sagemaker.processing as sp
    from sagemaker.estimator import EstimatorBase, UploadedCode

    from infra.sagemaker import pipeline as pl

    fake_code = UploadedCode(
        s3_prefix="s3://b/code/sourcedir.tar.gz", script_name="train_entry.py"
    )
    with mock.patch.object(
        sp.ScriptProcessor, "_upload_code",
        lambda self, code, kms_key=None: "s3://b/code/evaluate_entry.py",
    ), mock.patch.object(
        EstimatorBase, "_stage_user_code_in_s3", lambda self, *a, **k: fake_code
    ):
        definition = json.loads(pl.build_pipeline(0.45).definition())

    assert [s["Name"] for s in definition["Steps"]] == [
        "TrainLightGBM", "EvaluateModel", "GateWAPE",
    ]
    gate = next(s for s in definition["Steps"] if s["Type"] == "Condition")
    assert [b["Name"] for b in gate["Arguments"]["IfSteps"]] == [
        "RegisterApproved-RegisterModel"
    ]
    assert [b["Name"] for b in gate["Arguments"]["ElseSteps"]] == ["WAPETropEleve"]
    assert [p["Name"] for p in definition["Parameters"]] == ["WapeThreshold"]
    condition = json.dumps(gate["Arguments"]["Conditions"])
    assert "regression_metrics.test_wape.value" in condition
