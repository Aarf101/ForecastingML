"""Orchestration 100 % AWS : Step Functions + EventBridge autour du pipeline.

Le pipeline n'est plus enchaîné par des cibles ``make`` : une **state machine**
Step Functions exécute les étapes dans l'ordre, s'arrête sur erreur, et applique le
gate WAPE avant d'enregistrer le modèle. Une règle **EventBridge** peut la
déclencher automatiquement.

États : ``RunETL`` (Glue, .sync) → ``TrainModel`` (SageMaker, .sync) →
``GateWAPE`` (Lambda d'enregistrement) → ``ModeleApprouve`` (Choice) →
``Publie`` / ``Rejete``.

Usage :
    python -m infra.aws.orchestrate up          # Lambda + state machine + règle
    python -m infra.aws.orchestrate run         # une exécution, avec attente
    python -m infra.aws.orchestrate status      # état + dernières exécutions
    python -m infra.aws.orchestrate print       # ASL JSON (à coller dans la console)
    python -m infra.aws.orchestrate down        # supprime règle, machine et Lambda
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import boto3

from retail_forecast.config import SETTINGS

REGION = SETTINGS.region
FUNCTION = "retail-register-model"
MACHINE = "retail-forecast-pipeline"
RULE = "retail-forecast-daily"
GLUE_JOB = "retail-etl-raw-to-curated"
GROUP = "retail-forecast-lgbm"
IMAGE = (
    "683313688378.dkr.ecr.us-east-1.amazonaws.com/"
    "sagemaker-scikit-learn:1.4-2-py312-cpu-py3"
)
LAMBDA_FILE = Path(__file__).resolve().parents[1] / "lambda" / "register_model.py"


def _clients() -> dict:
    return {
        "sfn": boto3.client("stepfunctions", region_name=REGION),
        "glue": boto3.client("glue", region_name=REGION),
        "lambda": boto3.client("lambda", region_name=REGION),
        "events": boto3.client("events", region_name=REGION),
        "sts": boto3.client("sts", region_name=REGION),
    }


def _role_arn(c: dict) -> str:
    """Rôle d'exécution unique du lab : jamais de rôle créé (interdit ici)."""
    if SETTINGS.role_arn:
        return SETTINGS.role_arn
    return f"arn:aws:iam::{c['sts'].get_caller_identity()['Account']}:role/LabRole"


def _machine_arn(c: dict) -> str:
    """ARN complet de la state machine (``describe_state_machine`` l'exige)."""
    account = c["sts"].get_caller_identity()["Account"]
    return f"arn:aws:states:{REGION}:{account}:stateMachine:{MACHINE}"



def definition(role: str, lambda_arn: str) -> dict:
    """Définition Amazon States Language (ASL) du pipeline."""
    bucket = SETTINGS.bucket
    threshold = SETTINGS.wape_threshold
    train_name = "States.Format('retail-lgbm-{}', $.run_id)"

    return {
        "Comment": "Retail : Glue -> SageMaker -> gate WAPE -> Model Registry",
        "StartAt": "RunETL",
        "States": {
            "RunETL": {
                "Type": "Task",
                "Comment": "ETL managé : raw/ -> curated/ (Spark)",
                "Resource": "arn:aws:states:::glue:startJobRun.sync",
                "Parameters": {"JobName": GLUE_JOB, "Arguments": {"--BUCKET": bucket}},
                "Retry": [
                    {
                        "ErrorEquals": ["Glue.AWSGlueException", "States.TaskFailed"],
                        "IntervalSeconds": 30,
                        "MaxAttempts": 2,
                        "BackoffRate": 2.0,
                    }
                ],
                "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "Echec"}],
                "Next": "TrainModel",
            },
            "TrainModel": {
                "Type": "Task",
                "Comment": "Training Job LightGBM (ml.m5.large, 20 min max)",
                "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
                "Parameters": {
                    "TrainingJobName.$": train_name,
                    "AlgorithmSpecification": {
                        "TrainingImage": IMAGE,
                        "TrainingInputMode": "File",
                    },
                    "RoleArn": role,
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
                            "ContentType": "application/x-parquet",
                        }
                    ],
                    "OutputDataConfig": {"S3OutputPath": f"s3://{bucket}/models/"},
                    "ResourceConfig": {
                        "InstanceType": "ml.m5.large",
                        "InstanceCount": 1,
                        "VolumeSizeInGB": 30,
                    },
                    "StoppingCondition": {"MaxRuntimeInSeconds": 1200},
                    "HyperParameters": {
                        "sagemaker_program": "train_entry.py",
                        "sagemaker_submit_directory": f"s3://{bucket}/code/sourcedir.tar.gz",
                        "sagemaker_region": REGION,
                        "sagemaker_container_log_level": "20",
                        "horizons": "1,7,14,28",
                        "n-estimators": "300",
                        "early-stopping-rounds": "40",
                    },
                },
                "Retry": [
                    {
                        "ErrorEquals": ["SageMaker.AmazonSageMakerException"],
                        "IntervalSeconds": 30,
                        "MaxAttempts": 2,
                        "BackoffRate": 2.0,
                    }
                ],
                "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "Echec"}],
                "Next": "GateWAPE",
            },
            "GateWAPE": {
                "Type": "Task",
                "Comment": "Lit metrics.json puis enregistre (Approved ou bloqué)",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": lambda_arn,
                    "Payload": {
                        "bucket": bucket,
                        "training_job.$": train_name,
                        "threshold": threshold,
                        "group": GROUP,
                        "image": IMAGE,
                    },
                },
                "ResultSelector": {
                    "approved.$": "$.Payload.approved",
                    "status.$": "$.Payload.status",
                    "wape.$": "$.Payload.wape",
                    "baseline_wape.$": "$.Payload.baseline_wape",
                    "model_package_arn.$": "$.Payload.model_package_arn",
                },
                "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "Echec"}],
                "Next": "ModeleApprouve",
            },
            "ModeleApprouve": {
                "Type": "Choice",
                "Comment": "Le gate décide : WAPE <= seuil ?",
                "Choices": [
                    {"Variable": "$.approved", "BooleanEquals": True, "Next": "Publie"}
                ],
                "Default": "Rejete",
            },
            "Publie": {"Type": "Succeed"},
            "Rejete": {
                "Type": "Fail",
                "Error": "WapeAuDessusDuSeuil",
                "Cause": "Version enregistrée en PendingManualApproval : non servie.",
            },
            "Echec": {"Type": "Fail", "Error": "PipelineEnErreur"},
        },
    }


# --------------------------------------------------------------------------- #
# déploiement
# --------------------------------------------------------------------------- #
def deploy_lambda(c: dict, role: str) -> str:
    """Crée ou met à jour la Lambda du gate ; renvoie son ARN."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(LAMBDA_FILE, arcname="register_model.py")
    code = buf.getvalue()

    try:
        arn = c["lambda"].get_function(FunctionName=FUNCTION)["Configuration"][
            "FunctionArn"
        ]
        c["lambda"].update_function_code(FunctionName=FUNCTION, ZipFile=code)
        c["lambda"].get_waiter("function_updated_v2").wait(FunctionName=FUNCTION)
        c["lambda"].update_function_configuration(
            FunctionName=FUNCTION,
            Runtime="python3.12",
            Handler="register_model.lambda_handler",
            Role=role,
            Timeout=120,
            MemorySize=512,
            Environment={
                "Variables": {
                    "WAPE_THRESHOLD": f"{SETTINGS.wape_threshold:.2f}",
                }
            },
        )
        c["lambda"].get_waiter("function_updated_v2").wait(FunctionName=FUNCTION)
        print(f"[lambda] {FUNCTION} mise à jour")
        return arn
    except c["lambda"].exceptions.ResourceNotFoundException:
        arn = c["lambda"].create_function(
            FunctionName=FUNCTION,
            Runtime="python3.12",
            Role=role,
            Handler="register_model.lambda_handler",
            Code={"ZipFile": code},
            Timeout=120,
            MemorySize=512,
            Environment={
                "Variables": {"WAPE_THRESHOLD": f"{SETTINGS.wape_threshold:.2f}"}
            },
            Description="Gate WAPE + enregistrement au Model Registry",
        )["FunctionArn"]
        c["lambda"].get_waiter("function_active_v2").wait(FunctionName=FUNCTION)
        print(f"[lambda] {FUNCTION} créée")
        return arn


def deploy_machine(c: dict, role: str, lambda_arn: str) -> str:
    """Crée ou met à jour la state machine ; renvoie son ARN."""
    text = json.dumps(definition(role, lambda_arn))
    try:
        arn = _machine_arn(c)
        c["sfn"].describe_state_machine(stateMachineArn=arn)
        c["sfn"].update_state_machine(stateMachineArn=arn, definition=text, roleArn=role)
        print(f"[sfn] {MACHINE} mise à jour")
        return arn
    except c["sfn"].exceptions.StateMachineDoesNotExist:
        arn = c["sfn"].create_state_machine(
            name=MACHINE,
            definition=text,
            roleArn=role,
            type="STANDARD",
            tracingConfiguration={"enabled": False},
        )["stateMachineArn"]
        print(f"[sfn] {MACHINE} créée")
        return arn


def deploy_rule(c: dict, machine_arn: str, role: str, schedule: str) -> None:
    """Règle EventBridge : déclenche la machine à intervalle fixe (ex. rate(1 day))."""
    if not schedule:
        print("[events] pas de planification (--schedule vide)")
        return
    c["events"].put_rule(Name=RULE, ScheduleExpression=schedule, State="ENABLED")
    rule_arn = c["events"].describe_rule(Name=RULE)["Arn"]
    c["events"].put_targets(
        Rule=RULE,
        Targets=[
            {
                "Id": "retail-pipeline",
                "Arn": machine_arn,
                "RoleArn": role,
                "Input": json.dumps({"run_id": "scheduled"}),
            }
        ],
    )
    print(f"[events] règle {RULE} → {schedule}")


# --------------------------------------------------------------------------- #
# exécution / inspection / suppression
# --------------------------------------------------------------------------- #
def run(c: dict, machine_arn: str, wait: bool = True, run_id: str = "") -> dict:
    """Démarre une exécution ; attend la fin si demandé."""
    run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    name = f"run-{run_id}"[:80]
    resp = c["sfn"].start_execution(
        stateMachineArn=machine_arn,
        name=name,
        input=json.dumps({"run_id": run_id}),
    )
    arn = resp["executionArn"]
    print(f"[sfn] exécution {name} démarrée (run_id={run_id})")
    if not wait:
        return {"executionArn": arn, "status": "RUNNING"}

    for _ in range(180):  # ~30 min max (ETL 5 min + training 5 min + marge)
        desc = c["sfn"].describe_execution(executionArn=arn)
        status = desc["status"]
        if status != "RUNNING":
            print(f"[sfn] statut {status}")
            if desc.get("output"):
                print(json.dumps(json.loads(desc["output"]), indent=2, ensure_ascii=False))
            return {"executionArn": arn, "status": status, "output": desc.get("output")}
        time.sleep(10)
    return {"executionArn": arn, "status": "TIMEOUT"}


def status(c: dict) -> None:
    """État de la machine, de la règle et des trois dernières exécutions."""
    try:
        machine_arn = _machine_arn(c)
        machine = c["sfn"].describe_state_machine(stateMachineArn=machine_arn)
        print(f"sfn      : {MACHINE} {machine['status']}")
        for ex in c["sfn"].list_executions(stateMachineArn=machine_arn, maxResults=3)[
            "executions"
        ]:
            print(f"           {ex['name']:<32} {ex['status']}")
    except Exception as exc:  # noqa: BLE001
        print(f"sfn      : absent ({type(exc).__name__})")

    try:
        rule = c["events"].describe_rule(Name=RULE)
        print(f"events   : {RULE} {rule.get('State')} {rule.get('ScheduleExpression')}")
    except Exception as exc:  # noqa: BLE001
        print(f"events   : absent ({type(exc).__name__})")

    try:
        state = c["lambda"].get_function(FunctionName=FUNCTION)["Configuration"]["State"]
        print(f"lambda   : {FUNCTION} {state}")
    except Exception as exc:  # noqa: BLE001
        print(f"lambda   : absent ({type(exc).__name__})")


def down(c: dict) -> None:
    """Supprime la règle, puis la machine, puis la Lambda."""
    try:
        c["events"].remove_targets(Rule=RULE, Ids=["retail-pipeline"])
        c["events"].delete_rule(Name=RULE)
        print(f"[events] {RULE} supprimée")
    except c["events"].exceptions.ResourceNotFoundException:
        print(f"[events] {RULE} déjà absente")

    try:
        c["sfn"].delete_state_machine(stateMachineArn=_machine_arn(c))
        print(f"[sfn] {MACHINE} supprimée")
    except c["sfn"].exceptions.StateMachineDoesNotExist:
        print(f"[sfn] {MACHINE} déjà absente")

    try:
        c["lambda"].delete_function(FunctionName=FUNCTION)
        print(f"[lambda] {FUNCTION} supprimée")
    except c["lambda"].exceptions.ResourceNotFoundException:
        print(f"[lambda] {FUNCTION} déjà absente")


def deploy_glue_job(c: dict, role: str) -> None:
    """Dépose le script Spark dans S3 et crée (ou met à jour) le job Glue."""
    from retail_forecast.ingest.storage import s3_client

    spark_script = Path(__file__).resolve().parents[1] / "glue" / "etl_spark.py"
    key = "code/glue/etl_spark.py"
    s3_client().put_object(
        Bucket=SETTINGS.bucket,
        Key=key,
        Body=spark_script.read_bytes(),
        ContentType="text/x-python",
    )
    script_location = f"s3://{SETTINGS.bucket}/{key}"

    job = {
        "Name": GLUE_JOB,
        "Role": role,
        "Command": {
            "Name": "glueetl",
            "ScriptLocation": script_location,
            "PythonVersion": "3",
        },
        "DefaultArguments": {
            "--BUCKET": SETTINGS.bucket,
            "--job-language": "python",
            "--enable-metrics": "true",
            "--enable-continuous-cloudwatch-log": "true",
        },
        "GlueVersion": "4.0",
        "WorkerType": "G.1X",
        "NumberOfWorkers": 2,  # minimum : c'est ce qui fixe le prix (0,15 $/run)
        "Timeout": 15,
        "MaxRetries": 0,  # un retry doublerait la facture
        "ExecutionProperty": {"MaxConcurrentRuns": 1},
    }

    try:
        c["glue"].get_job(JobName=GLUE_JOB)
        # ``JobUpdate`` n'accepte pas ``Name`` (déjà dans JobName).
        update = {k: v for k, v in job.items() if k != "Name"}
        c["glue"].update_job(JobName=GLUE_JOB, JobUpdate=update)
        print(f"[glue] {GLUE_JOB} mis à jour ({script_location})")
    except c["glue"].exceptions.EntityNotFoundException:
        c["glue"].create_job(**job)
        print(f"[glue] {GLUE_JOB} créé ({script_location})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["up", "run", "status", "print", "down"], help="action"
    )
    parser.add_argument(
        "--schedule",
        default="rate(1 day)",
        help="expression EventBridge (vide pour ne pas planifier)",
    )
    parser.add_argument("--no-wait", action="store_true", help="ne pas attendre la fin")
    parser.add_argument("--run-id", default="", help="identifiant d'exécution")
    args = parser.parse_args(argv)

    c = _clients()
    if not SETTINGS.bucket:
        raise SystemExit("S3_BUCKET vide dans .env : lancer d'abord 'make aws-setup'")

    if args.action == "print":
        print(json.dumps(definition("ROLE_ARN", "LAMBDA_ARN"), indent=2, ensure_ascii=False))
        return 0

    role = _role_arn(c)
    if args.action == "up":
        print(f"[init] bucket={SETTINGS.bucket} region={REGION}")
        deploy_glue_job(c, role)
        lambda_arn = deploy_lambda(c, role)
        machine_arn = deploy_machine(c, role, lambda_arn)
        deploy_rule(c, machine_arn, role, args.schedule)
        print(f"[done] {machine_arn}")
    elif args.action == "run":
        machine_arn = _machine_arn(c)
        print(json.dumps(run(c, machine_arn, not args.no_wait, args.run_id), indent=2))
    elif args.action == "status":
        status(c)
    else:
        down(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

