"""HyperParameter Tuning Job SageMaker (recherche d'hyperparamètres LightGBM).

Répond au point « tuning » du sujet : au lieu de fixer ``n_estimators`` et
``learning_rate`` à la main, on laisse SageMaker explorer un espace de recherche
et retenir le jeu qui minimise le WAPE.

Comment ça marche :

* SageMaker lance N Training Jobs, chacun avec un jeu d'hyperparamètres différent
  (stratégie bayésienne : chaque essai tient compte des précédents) ;
* la métrique d'objectif est lue **dans les journaux CloudWatch** du conteneur,
  via le regex ``\\[hpo\\] test_wape=([0-9.]+)`` émis par ``train_entry.py`` ;
* à la fin, ``BestTrainingJob`` désigne le job gagnant — son artefact
  ``model.tar.gz`` est réutilisable tel quel par le Model Registry.

Toujours du boto3 pur, comme ``launch.py`` : pas de SDK ``sagemaker``.

Usage::

    python -m infra.sagemaker.tuning --dry-run        # affiche la requête
    python -m infra.sagemaker.tuning --max-jobs 6 --wait
    python -m infra.sagemaker.register --job <meilleur job>
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

from infra.sagemaker.launch import (
    IMAGE,
    INSTANCE,
    MAX_RUNTIME,
    PROGRAM,
    SOURCEDIR_KEY,
    _sm_client,
    upload_code,
)
from retail_forecast.config import SETTINGS
from retail_forecast.ingest.storage import s3_client

# Préfixe commun avec launch.py pour s'y retrouver dans la console.
JOB_PREFIX = "retail-lgbm-hpo"

# Métrique surveillée par le tuning (émise par train_entry.py, une seule fois).
# On optimise la **validation** : optimiser sur le test reviendrait à choisir les
# hyperparamètres sur les données censées juger le modèle (fuite indirecte).
OBJECTIVE = "valid_wape"
METRIC_REGEX = r"\[hpo\] valid_wape=([0-9\.]+)"

# Espace de recherche. Volontairement étroit : un espace trop large multiplie les
# essais donc la facture, et LightGBM converge vite sur ce volume de données.
SEARCH_SPACE = {
    "continuous": [
        {"Name": "learning-rate", "MinValue": "0.01", "MaxValue": "0.2",
         "ScalingType": "Logarithmic"},
    ],
    "integer": [
        {"Name": "num-leaves", "MinValue": "15", "MaxValue": "127",
         "ScalingType": "Linear"},
        {"Name": "n-estimators", "MinValue": "150", "MaxValue": "600",
         "ScalingType": "Linear"},
    ],
}

# Hyperparamètres figés pendant la recherche : un seul horizon pour que chaque
# essai reste court (~3 min), sinon 4 boosters par job quadruplent la facture.
STATIC = {
    "quantiles": "false",
    "horizons": "1",
    "early-stopping-rounds": "40",
    "valid-days": "56",
    "test-days": "28",
}


def tuning_request(bucket: str, name: str, code_uri: str, max_jobs: int,
                   parallel: int, strategy: str) -> dict:
    """Construit la requête ``CreateHyperParameterTuningJob``.

    Fonction pure : testable sans AWS (voir ``tests/test_tuning.py``).
    """
    return {
        "HyperParameterTuningJobName": name,
        "HyperParameterTuningJobConfig": {
            "Strategy": strategy,
            "HyperParameterTuningJobObjective": {
                "Type": "Minimize",
                "MetricName": OBJECTIVE,
            },
            "ResourceLimits": {
                "MaxNumberOfTrainingJobs": max_jobs,
                "MaxParallelTrainingJobs": parallel,
            },
            "ParameterRanges": {
                "ContinuousParameterRanges": SEARCH_SPACE["continuous"],
                "IntegerParameterRanges": SEARCH_SPACE["integer"],
            },
            # Le job s'arrête de lui-même si la métrique ne progresse plus :
            # évite de payer les derniers essais inutiles.
            "TrainingJobEarlyStoppingType": "Auto",
        },
        "TrainingJobDefinition": {
            "StaticHyperParameters": {
                # Convention « script mode » du conteneur SKLearn managé.
                "sagemaker_program": PROGRAM,
                "sagemaker_submit_directory": code_uri,
                "sagemaker_container_log_level": "20",
                "sagemaker_region": SETTINGS.region,
                **STATIC,
            },
            "AlgorithmSpecification": {
                "TrainingImage": IMAGE,
                "TrainingInputMode": "File",
                "MetricDefinitions": [
                    {"Name": OBJECTIVE, "Regex": METRIC_REGEX},
                ],
            },
            "RoleArn": SETTINGS.role_arn,
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"s3://{bucket}/curated/",
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "CompressionType": "None",
                    "ContentType": "application/x-parquet",
                }
            ],
            "OutputDataConfig": {"S3OutputPath": f"s3://{bucket}/tuning/"},
            "ResourceConfig": {
                "InstanceType": INSTANCE,
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": MAX_RUNTIME},
        },
        "Tags": [
            {"Key": "project", "Value": "retail-forecast"},
            {"Key": "phase", "Value": "tuning"},
        ],
    }


def wait_tuning(sm, name: str, poll: int = 30) -> dict:
    """Attend la fin du tuning en affichant l'avancement (essais terminés / total)."""
    started = time.time()
    while True:
        desc = sm.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=name
        )
        status = desc["HyperParameterTuningJobStatus"]
        counters = desc.get("TrainingJobStatusCounters", {})
        done = sum(
            counters.get(k, 0) for k in ("Completed", "Failed", "Stopped")
        )
        total = desc["HyperParameterTuningJobConfig"]["ResourceLimits"][
            "MaxNumberOfTrainingJobs"
        ]
        elapsed = (time.time() - started) / 60
        print(f"[tuning] {status} · {done}/{total} essais · {elapsed:.1f} min",
              flush=True)
        if status in ("Completed", "Failed", "Stopped"):
            return desc
        time.sleep(poll)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="HyperParameter Tuning Job LightGBM (SageMaker)."
    )
    parser.add_argument("--max-jobs", type=int, default=6,
                        help="nombre d'essais (1 essai ~3 min ~0,005 €)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="essais simultanés (1 = le moins cher)")
    parser.add_argument("--strategy", default="Bayesian",
                        choices=["Bayesian", "Random"])
    parser.add_argument("--wait", action="store_true", help="attendre la fin")
    parser.add_argument("--dry-run", action="store_true",
                        help="afficher la requête sans rien envoyer")
    parser.add_argument("--name", default="",
                        help="nom du tuning job (défaut horodaté)")
    parser.add_argument("--upload-only", action="store_true",
                        help="déposer seulement le code dans S3")
    args = parser.parse_args(argv)

    if not SETTINGS.bucket or not SETTINGS.role_arn:
        raise SystemExit(
            "S3_BUCKET et SAGEMAKER_ROLE_ARN doivent être renseignés (.env)"
        )

    name = args.name or f"{JOB_PREFIX}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    code_uri = f"s3://{SETTINGS.bucket}/{SOURCEDIR_KEY}"
    request = tuning_request(
        SETTINGS.bucket, name, code_uri, args.max_jobs, args.parallel, args.strategy
    )

    if args.dry_run:
        print(json.dumps(request, indent=2))
        return 0

    code_uri = upload_code(s3_client(), SETTINGS.bucket)
    request = tuning_request(SETTINGS.bucket, name, code_uri, args.max_jobs, args.parallel, args.strategy)
    if args.upload_only:
        print(f"[tuning] code déposé : {code_uri}")
        return 0

    sm = _sm_client()
    print(f"[tuning] {args.max_jobs} essais · {args.parallel} en parallèle · "
          f"objectif min {OBJECTIVE}")
    sm.create_hyper_parameter_tuning_job(**request)

    if not args.wait:
        print("[tuning] lancé — suivi : AWS_PROFILE=lab aws sagemaker "
              "describe-hyper-parameter-tuning-job "
              f"--hyper-parameter-tuning-job-name {name}")
        return 0

    desc = wait_tuning(sm, name)
    if desc["HyperParameterTuningJobStatus"] != "Completed":
        print(f"[tuning] échec : {desc.get('FailureReason', 'inconnue')}")
        return 1

    best = desc.get("BestTrainingJob", {})
    best_name = best.get("TrainingJobName", "")
    value = best.get("FinalHyperParameterTuningJobObjectiveMetric", {}).get("Value")
    print(f"[tuning] meilleur job : {best_name}")
    print(f"[tuning] {OBJECTIVE} = {value}")
    print("[tuning] hyperparamètres retenus :")
    for key, val in (best.get("TunedHyperParameters") or {}).items():
        print(f"  {key:<22} {val}")
    print(f"[tuning] artefact : s3://{SETTINGS.bucket}/tuning/{name}/"
          f"{best_name}/output/model.tar.gz")
    print(f"[tuning] suite : python -m infra.sagemaker.register --job {best_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
