"""Cycle cloud : candidat évalué, inférence contrôlée, publication atomique."""

from __future__ import annotations

import argparse
import json
import time
import uuid

from botocore.exceptions import ClientError

from retail_forecast.config import SETTINGS
from retail_forecast.ingest.storage import s3_client

ACTIVE_KEY = "deployment/active.json"
HORIZONS = list(range(1, 17))


def read_active(s3, *, optional=False):
    """Ne masque ni les erreurs IAM ni les pannes réseau."""
    try:
        response = s3.get_object(Bucket=SETTINGS.bucket, Key=ACTIVE_KEY)
    except ClientError as exc:
        if optional and exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None, None
        raise
    return json.loads(response["Body"].read()), response["ETag"]


def publish(s3, candidate, previous, etag):
    """Compare puis échange le manifeste ; un concurrent ne peut être écrasé."""
    document = {**candidate, "previous": previous}
    condition = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
    s3.put_object(Bucket=SETTINGS.bucket, Key=ACTIVE_KEY,
                  Body=json.dumps(document).encode(), ContentType="application/json", **condition)


def run_forecast(sm, s3, package_arn, run_id):
    """Valide un lot complet avant toute modification du manifeste actif."""
    from infra.sagemaker import batch_transform as batch

    package = sm.describe_model_package(ModelPackageName=package_arn)
    if package["ModelApprovalStatus"] != "Approved":
        raise RuntimeError("candidat non approuvé")
    container = package["InferenceSpecification"]["Containers"][0]
    metadata = batch.model_metadata(s3, container["ModelDataUrl"])
    if sorted(metadata["horizons"]) != HORIZONS:
        raise ValueError("le modèle doit couvrir chaque jour de J+1 à J+16")
    if metadata.get("evaluation_protocol") != "rolling_origin_purged_v2":
        raise ValueError("ancien protocole : réentraînement requis")
    sales = batch.read_curated(s3)
    import pandas as pd
    sales["date"] = pd.to_datetime(sales["date"])
    rows = batch.build_inference_rows(sales, metadata["features"], HORIZONS,
                                      levels=metadata["categorical_levels"])
    current = sales[sales["date"] == sales["date"].max()]
    if rows[["store_id", "item_id"]].drop_duplicates().shape[0] != current[["store_id", "item_id"]].drop_duplicates().shape[0]:
        raise ValueError("couverture incomplète : séries absentes ou historique insuffisant")
    if metadata.get("quantiles"):
        rows["return_quantiles"] = True
    name = f"retail-transform-{run_id}"
    batch.ensure_model(sm, name, container["Image"], container["ModelDataUrl"])
    uri = batch.upload_rows(s3, rows, metadata["features"],
                            input_key=f"inference/input/{name}/features.jsonl")
    batch.run_transform(sm, name, name, uri)
    key = f"forecast/runs/{run_id}/forecast.parquet"
    batch.collect_output(s3, rows, package["ModelPackageVersion"], name, forecast_key=key)
    metrics = {key: sum(m[key] for m in metadata["metrics"].values()) / len(metadata["metrics"])
               for key in ["test_wape", "baseline_wape", "test_bias"]}
    for field in ["test_rmsle", "test_coverage_80", "test_pinball_p10", "test_pinball_p90", "test_interval_width"]:
        if all(field in m for m in metadata["metrics"].values()):
            metrics[field] = sum(m[field] for m in metadata["metrics"].values()) / len(metadata["metrics"])
    return {"metrics": metrics, "model_package_arn": package_arn, "forecast_key": key, "run_id": run_id,
            "model_data_url": container["ModelDataUrl"], "target": "observed_sales",
            "active_series": len(current[["store_id", "item_id"]].drop_duplicates()),
            "historical_series": len(sales[["store_id", "item_id"]].drop_duplicates())}


def run(mode, evaluated_execution=None):
    """Le mensuel entraîne ; le quotidien utilise exclusivement le modèle actif."""
    from infra.sagemaker.launch import _sm_client

    sm, s3 = _sm_client(), s3_client()
    previous, etag = read_active(s3, optional=mode == "training")
    run_id = uuid.uuid4().hex
    if mode == "training":
        from infra.sagemaker.pipeline import build_pipeline

        from infra.sagemaker.register import ensure_group
        ensure_group(sm)
        if evaluated_execution:
            from infra.sagemaker.recovery import recover_registration
            package_arn = recover_registration(sm, s3, evaluated_execution)
        else:
            pipeline = build_pipeline(SETTINGS.wape_threshold, ",".join(map(str, HORIZONS)))
            pipeline.upsert(role_arn=SETTINGS.role_arn)
            execution_arn = pipeline.start().arn
            deadline = time.monotonic() + 7200
            while True:
                status = sm.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
                if status["PipelineExecutionStatus"] == "Succeeded":
                    break
                if status["PipelineExecutionStatus"] in ("Failed", "Stopped", "Stopping"):
                    raise RuntimeError(f"pipeline : {status}")
                if time.monotonic() > deadline:
                    sm.stop_pipeline_execution(PipelineExecutionArn=execution_arn)
                    raise TimeoutError("pipeline interrompu après deux heures")
                time.sleep(20)
            packages = []
            gates = []
            for page in sm.get_paginator("list_pipeline_execution_steps").paginate(
                PipelineExecutionArn=execution_arn
            ):
                gates.extend(step for step in page["PipelineExecutionSteps"] if step["StepName"] == "GateWAPE")
                packages.extend(step["Metadata"]["RegisterModel"]["Arn"]
                                for step in page["PipelineExecutionSteps"]
                                if "RegisterModel" in step.get("Metadata", {}))
            if (len(gates) != 1 or gates[0]["StepStatus"] != "Succeeded"
                    or gates[0].get("Metadata", {}).get("Condition", {}).get("Outcome") != "True"):
                raise RuntimeError("aucune preuve de réussite du gate indépendant")
            if len(packages) != 1:
                raise RuntimeError("version de registre ambiguë pour cette exécution")
            package_arn = packages[0]
    else:
        package_arn = previous["model_package_arn"]
    candidate = run_forecast(sm, s3, package_arn, run_id)
    # Historique borné : conserve le dernier manifeste, sans récursion infinie.
    prior = {k: v for k, v in previous.items() if k != "previous"} if previous else None
    publish(s3, candidate, prior, etag)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["training", "inference"])
    parser.add_argument("--evaluated-execution", default=None)
    args = parser.parse_args(argv)
    if args.evaluated_execution and args.mode != "training":
        parser.error("la reprise concerne uniquement le workflow training")
    run(args.mode, args.evaluated_execution)


if __name__ == "__main__":
    main()
