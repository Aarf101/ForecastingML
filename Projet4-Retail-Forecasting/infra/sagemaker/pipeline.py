"""SageMaker Pipeline : features -> entraînement -> évaluation -> gate -> registre.

Répond au point « Mise en place de SageMaker Pipelines » du sujet. C'est la
version *managée par SageMaker* de ce que ``infra/aws/orchestrate.py`` fait avec
Step Functions : ici les étapes, les dépendances et les conditions sont décrites
en Python, et SageMaker les exécute lui-même.

Le graphe ::

    PreprocessFeatures  (Processing Job : features calculées UNE fois)
           │  features.parquet -> s3://<bucket>/features/<execution-id>/
           ▼
    TrainLightGBM  (Training Job, conteneur SKLearn managé)
           │  model.tar.gz
           ▼
    EvaluateModel  (Processing Job : recalcule le WAPE sur la fenêtre de test)
           │  evaluation.json  -> regression_metrics.test_wape.value
           ▼
    GateWAPE       (ConditionStep)
           ├── WAPE <= seuil  -> RegisterModel (version Approved)
           └── WAPE  > seuil  -> FailStep (échec explicite, rien n'est publié)

L'étape ``PreprocessFeatures`` n'est pas décorative : sans elle les features sont
recalculées par chaque consommateur (conteneur d'entraînement, client de Batch
Transform, échantillon de l'endpoint). Si l'un d'eux change un paramètre
(``LAGS``, ``ROLLS``), l'entraînement et l'inférence ne parlent plus le même
espace de features — c'est le *train/serve skew*. Ici la définition vit à un seul
endroit, et l'évaluation relit exactement la table de l'entraînement.

Le déploiement (Batch Transform vers ``forecast/``) reste
``infra/sagemaker/batch_transform.py`` : il n'a pas besoin d'être un step pour
tourner, et le garder hors pipeline évite de payer une instance à chaque run.

Seul module du dépôt qui utilise le SDK ``sagemaker`` (v2) : l'API des pipelines
n'existe pas en boto3 pur de façon maintenable. Les autres scripts restent en
boto3.

Usage::

    python -m infra.sagemaker.pipeline --dry-run        # définition JSON seule
    python -m infra.sagemaker.pipeline --upsert         # crée/met à jour
    python -m infra.sagemaker.pipeline --run --wait     # exécute et attend
    python -m infra.sagemaker.pipeline --status         # dernières exécutions
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from infra.sagemaker.launch import (
    IMAGE,
    INSTANCE,
    MAX_RUNTIME,
    _sm_client,
    build_tarball,
)
from retail_forecast.config import SETTINGS
from retail_forecast.ingest.storage import s3_client

PIPELINE_NAME = "retail-forecast-mlops"
MODEL_GROUP = "retail-forecast-lgbm"
EVAL_SCRIPT = Path(__file__).with_name("evaluate_entry.py")
PREPROC_SCRIPT = Path(__file__).with_name("preprocess_entry.py")


def _require_sdk():
    """Importe le SDK en un seul endroit, avec un message clair s'il manque."""
    try:
        import sagemaker  # noqa: F401
        from sagemaker.workflow.pipeline import Pipeline  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dépendance optionnelle
        raise SystemExit(
            "SDK sagemaker (v2) requis : pip install 'sagemaker>=2.200,<3'"
        ) from exc


def _boto_session():
    """Session boto3 : profil ``.env`` si présent, sinon chaîne par défaut.

    Sur le poste on passe par ``AWS_PROFILE=lab`` ; **sur l'instance SageMaker du
    Learner Lab il n'existe aucun profil** (les identifiants viennent du rôle
    IAM). Sans ce repli, ``pipeline.py`` lèverait ``ProfileNotFound`` dans le
    notebook, alors que tous les autres modules (boto3.client sans profil)
    fonctionnent.
    """
    import boto3
    from botocore.exceptions import ProfileNotFound

    if SETTINGS.profile:
        try:
            return boto3.Session(
                profile_name=SETTINGS.profile, region_name=SETTINGS.region
            )
        except ProfileNotFound:
            print(
                f"[pipeline] profil '{SETTINGS.profile}' absent -> chaîne par "
                "défaut (rôle IAM de l'instance)",
                flush=True,
            )
            # Sans ça, botocore relit AWS_PROFILE dans l'environnement et lève
            # de nouveau ProfileNotFound : la variable doit disparaître.
            os.environ.pop("AWS_PROFILE", None)
    return boto3.Session(region_name=SETTINGS.region)


def _session():
    """PipelineSession : les ``.fit()`` n'envoient rien, ils décrivent des steps."""
    from sagemaker.workflow.pipeline_context import PipelineSession

    boto_sess = _boto_session()
    # default_bucket = le bucket du projet : sinon le SDK tenterait de créer
    # ``sagemaker-<région>-<compte>``, ce que le lab n'autorise pas.
    return PipelineSession(boto_session=boto_sess, default_bucket=SETTINGS.bucket)


def _stage_source_dir() -> str:
    """Réextrait ``sourcedir.tar.gz`` dans un dossier temporaire pour le SDK.

    Le SDK veut un *répertoire* (``source_dir``) alors que ``launch.py`` produit
    déjà l'archive : on la réutilise pour que local et cloud entraînent
    exactement le même code.
    """
    import io
    import tarfile

    tmp = tempfile.mkdtemp(prefix="retail-src-")
    with tarfile.open(fileobj=io.BytesIO(build_tarball()), mode="r:gz") as tar:
        tar.extractall(tmp)
    return tmp


def build_pipeline(threshold: float, horizons: str = "1,7,14,28"):
    """Construit l'objet ``Pipeline`` (aucun appel AWS tant qu'on ne l'upsert pas)."""
    _require_sdk()
    from sagemaker.inputs import TrainingInput
    from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor

    from sagemaker.sklearn.estimator import SKLearn
    from sagemaker.workflow.condition_step import ConditionStep
    from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
    from sagemaker.workflow.execution_variables import ExecutionVariables
    from sagemaker.workflow.fail_step import FailStep
    from sagemaker.workflow.functions import Join, JsonGet
    from sagemaker.workflow.parameters import ParameterFloat
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.properties import PropertyFile
    from sagemaker.workflow.step_collections import RegisterModel
    from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep

    session = _session()
    bucket = SETTINGS.bucket
    role = SETTINGS.role_arn

    # Le paquet du projet est déposé à une clé connue : l'étape d'évaluation le
    # remonte pour réimporter ``retail_forecast`` (recalcul indépendant du WAPE).
    from infra.sagemaker.launch import upload_code

    code_uri = upload_code(s3_client(), bucket)

    seuil = ParameterFloat(name="WapeThreshold", default_value=threshold)

    # Le même conteneur (image SKLearn déjà validée dans ce compte) sert à toutes
    # les étapes Processing : pas de résolution d'image « processing » par version.
    processor = ScriptProcessor(
        command=["python3"],
        image_uri=IMAGE,
        instance_type=INSTANCE,
        instance_count=1,
        role=role,
        base_job_name="retail-proc",
        sagemaker_session=session,
    )

    # 0) Préparation des features — calculées UNE fois et écrites dans
    #    s3://<bucket>/features/. L'entraînement et l'évaluation relisent cette
    #    table (chaînage par propriété : chaque exécution consomme les features
    #    qu'elle vient de produire). C'est ce qui empêche le train/serve skew :
    #    une seule définition des features pour tout le pipeline.
    step_preprocess = ProcessingStep(
        name="PreprocessFeatures",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=f"s3://{bucket}/curated/sales/",
                destination="/opt/ml/processing/input/curated",
            ),
            ProcessingInput(
                source=code_uri,
                destination="/opt/ml/processing/input/code",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="features",
                source="/opt/ml/processing/output/features",
                # Un sous-dossier par exécution : deux exécutions concurrentes ne
                # s'écrasent pas, et le chaînage par propriété garantit que
                # l'entraînement lit la table de SON exécution.
                destination=Join(
                    on="/",
                    values=[
                        f"s3://{bucket}/features",
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                    ],
                ),
            )
        ],
        code=str(PREPROC_SCRIPT),
    )
    features_uri = (
        step_preprocess.properties.ProcessingOutputConfig.Outputs[
            "features"
        ].S3Output.S3Uri
    )

    # 1) Entraînement — même image, même code, mêmes hyperparamètres que launch.py
    estimator = SKLearn(
        entry_point="train_entry.py",
        source_dir=_stage_source_dir(),
        image_uri=IMAGE,
        instance_type=INSTANCE,
        instance_count=1,
        role=role,
        base_job_name="retail-lgbm-pipe",
        output_path=f"s3://{bucket}/models/",
        hyperparameters={
            "horizons": horizons,
            "n-estimators": "400",
            "early-stopping-rounds": "40",
        },
        max_run=MAX_RUNTIME,
        sagemaker_session=session,
    )
    step_train = TrainingStep(
        name="TrainLightGBM",
        estimator=estimator,
        inputs={
            "training": TrainingInput(
                s3_data=features_uri,
                content_type="application/x-parquet",
            )
        },
        # Cache : si le code, les hyperparamètres et les données n'ont pas changé,
        # SageMaker réutilise l'artefact précédent — pas de Training Job payé
        # pour rien lors d'une deuxième démonstration.
        cache_config=CacheConfig(enable_caching=True, expire_after="P14D"),
    )
    # Aucune donnée ne circule entre les deux steps (l'entrée est une propriété
    # de l'étape précédente, pas un artefact consommé directement) : la
    # dépendance doit être déclarée explicitement.
    step_train.add_depends_on([step_preprocess.name])

    # 2) Évaluation — Processing Job : recharge les boosters de l'artefact et
    #    RECALCULE le WAPE sur la fenêtre de test (features de l'étape 0).
    rapport = PropertyFile(name="EvaluationReport", output_name="evaluation",
                           path="evaluation.json")
    step_eval = ProcessingStep(
        name="EvaluateModel",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model",
            ),
            # Données de test : c'est ce qui permet de RECALCULER le WAPE au lieu
            # de recopier celui de l'entraîneur. Même table de features que
            # l'étape d'entraînement (aucune divergence possible).
            ProcessingInput(
                source=features_uri,
                destination="/opt/ml/processing/input/curated",
            ),
            # Paquet retail_forecast (features + métriques), même archive que
            # celle utilisée par l'entraînement.
            ProcessingInput(
                source=code_uri,
                destination="/opt/ml/processing/input/code",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output",
                destination=f"s3://{bucket}/evaluation/",
            )
        ],
        code=str(EVAL_SCRIPT),
        property_files=[rapport],
    )

    # 3a) Gate OK -> enregistrement au registre (même groupe que register.py).
    #     Métadonnées attachées à la version : le SDK v2 n'expose pas de champ
    #     « ModelQuality » dans ModelMetrics, on utilise donc les métadonnées
    #     client (même mécanisme que register.py) et on pointe le rapport
    #     d'évaluation produit par l'étape précédente.
    qualite = Join(
        on="/",
        values=[
            step_eval.properties.ProcessingOutputConfig.Outputs[
                "evaluation"
            ].S3Output.S3Uri,
            "evaluation.json",
        ],
    )
    step_register = RegisterModel(
        name="RegisterApproved",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=[INSTANCE],
        transform_instances=[INSTANCE],
        model_package_group_name=MODEL_GROUP,
        approval_status="Approved",
        customer_metadata_properties={
            "gate": f"mean_wape <= {threshold}",
            "horizons": horizons,
            "pipeline": PIPELINE_NAME,
            # Valeurs statiques imposées par le format du step ; le WAPE chiffré
            # est dans le rapport d'évaluation de l'exécution (sortie S3 ci-dessous).
            "evaluation_prefix": f"s3://{bucket}/evaluation/",
        },
    )

    # 3b) Gate KO -> échec explicite : rien n'est publié, l'exécution est rouge
    step_fail = FailStep(
        name="WAPETropEleve",
        error_message="WAPE moyen au-dessus du seuil : modele non publie",
    )

    step_gate = ConditionStep(
        name="GateWAPE",
        conditions=[
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=step_eval.name,
                    property_file=rapport,
                    json_path="regression_metrics.test_wape.value",
                ),
                right=seuil,
            )
        ],
        if_steps=[step_register],
        else_steps=[step_fail],
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[seuil],
        steps=[step_preprocess, step_train, step_eval, step_gate],
        sagemaker_session=session,
    )


def check_env() -> int:
    """Vérifie les pré-requis du Learner Lab avant de dépenser quoi que ce soit.

    Contrôle : identité, bucket (lecture + écriture), lecture des pipelines, et
    existence du groupe de registre. L'écriture ``CreatePipeline`` n'est pas
    testable sans créer un pipeline : elle est tentée à l'``--upsert``.
    """
    from botocore.exceptions import ClientError

    session = _boto_session()
    ok = True

    ident = session.client("sts").get_caller_identity()
    print(f"[check] compte {ident['Account']} · {ident['Arn']}")

    s3 = session.client("s3")
    try:
        s3.put_object(Bucket=SETTINGS.bucket, Key="code/.check", Body=b"ok")
        print(f"[check] bucket {SETTINGS.bucket} : écriture OK")
    except ClientError as exc:
        ok = False
        print(f"[check] bucket {SETTINGS.bucket} : ECHEC {exc.response['Error']['Code']}")

    sm = session.client("sagemaker")
    for label, appel in (
        ("sagemaker:ListPipelines", lambda: sm.list_pipelines(MaxResults=1)),
        ("sagemaker:ListModelPackages",
         lambda: sm.list_model_packages(
             ModelPackageGroupName=MODEL_GROUP, MaxResults=1)),
    ):
        try:
            appel()
            print(f"[check] {label} : lecture OK")
        except ClientError as exc:
            ok = False
            print(f"[check] {label} : ECHEC {exc.response['Error']['Code']}")

    nom_role = SETTINGS.role_arn.rsplit("/", 1)[-1]
    try:
        session.client("iam").get_role(RoleName=nom_role)
        print(f"[check] rôle {nom_role} : lisible")
    except ClientError as exc:
        ok = False
        print(f"[check] rôle {nom_role} : ECHEC {exc.response['Error']['Code']}")

    print("[check] " + ("prêt pour --upsert" if ok else "corriger les lignes ECHEC"))
    print("[check] note : sagemaker:CreatePipeline ne sera réellement testé "
          "qu'au premier --upsert")
    return 0 if ok else 1


def wait_execution(sm, arn: str, poll: int = 30) -> dict:
    """Suit l'exécution et affiche l'état des étapes (dédupliqué par nom)."""
    while True:
        desc = sm.describe_pipeline_execution(PipelineExecutionArn=arn)
        status = desc["PipelineExecutionStatus"]
        etapes: dict[str, str] = {}
        for step in desc.get("PipelineExecutionSteps", []):
            etapes[step["StepName"]] = step["StepStatus"]
        detail = " · ".join(f"{k}={v}" for k, v in etapes.items())
        print(f"[pipeline] {status} · {detail}", flush=True)
        if status in ("Succeeded", "Failed", "Stopped"):
            return desc
        time.sleep(poll)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SageMaker Pipeline : train -> eval -> gate WAPE -> registry."
    )
    parser.add_argument("--threshold", type=float, default=SETTINGS.wape_threshold,
                        help="seuil WAPE du gate (défaut .env)")
    parser.add_argument("--horizons", default="1,7,14,28")
    parser.add_argument("--dry-run", action="store_true",
                        help="affiche la définition JSON, n'envoie rien")
    parser.add_argument("--upsert", action="store_true",
                        help="crée ou met à jour le pipeline dans SageMaker")
    parser.add_argument("--run", action="store_true",
                        help="démarre une exécution (implique --upsert)")
    parser.add_argument("--wait", action="store_true",
                        help="attend la fin de l'exécution")
    parser.add_argument("--status", action="store_true",
                        help="affiche les dernières exécutions")
    parser.add_argument("--check", action="store_true",
                        help="vérifie les pré-requis du lab (bucket, IAM, registre)")
    args = parser.parse_args(argv)

    if not SETTINGS.bucket or not SETTINGS.role_arn:
        raise SystemExit(
            "S3_BUCKET et SAGEMAKER_ROLE_ARN doivent être renseignés (.env)"
        )

    if args.check:
        return check_env()

    if args.status:
        sm = _sm_client()
        try:
            executions = sm.list_pipeline_executions(
                PipelineName=PIPELINE_NAME, MaxResults=5, SortOrder="Descending"
            )["PipelineExecutionSummaries"]
        except sm.exceptions.ClientError:
            print(f"[pipeline] {PIPELINE_NAME} n'existe pas encore (--upsert)")
            return 1
        for ex in executions:
            print(f"  {ex['PipelineExecutionStatus']:<10} {ex['StartTime']}  "
                  f"{ex['PipelineExecutionDisplayName']}")
        return 0

    pipeline = build_pipeline(args.threshold, args.horizons)

    if args.dry_run:
        definition = json.loads(pipeline.definition())
        print(json.dumps(definition, indent=2)[:4000])
        print(f"\n[pipeline] {len(definition['Steps'])} étapes : "
              + ", ".join(s["Name"] for s in definition["Steps"]))
        return 0

    if not (args.upsert or args.run):
        print("[pipeline] rien à faire : préciser --upsert, --run ou --status")
        return 0

    from botocore.exceptions import ClientError

    try:
        response = pipeline.upsert(
            role_arn=SETTINGS.role_arn,
            description="Prevision de la demande J+1..J+28 (LightGBM, gate WAPE).",
        )
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        print(f"[pipeline] upsert refusé : {code}")
        if code in ("AccessDenied", "AccessDeniedException", "UnauthorizedOperation"):
            print("[pipeline] le compte n'autorise pas sagemaker:CreatePipeline.")
            print("[pipeline] repli : la partie ML reste couverte par Step Functions")
            print("           make aws-pipeline-run   (train -> gate -> registry)")
        return 1
    print(f"[pipeline] {PIPELINE_NAME} enregistré : {response['PipelineArn']}")

    if not args.run:
        print("[pipeline] suite : python -m infra.sagemaker.pipeline --run --wait")
        return 0

    execution = pipeline.start(parameters={"WapeThreshold": args.threshold})
    arn = execution.arn
    print(f"[pipeline] exécution {execution.execution_id} démarrée")
    print(f"[pipeline] console : SageMaker > Pipelines > {PIPELINE_NAME}")

    if not args.wait:
        return 0

    desc = wait_execution(_sm_client(), arn)
    statut = desc["PipelineExecutionStatus"]
    print(f"[pipeline] statut final : {statut}")
    if statut != "Succeeded":
        print(f"[pipeline] échec : {desc.get('FailureReason', 'voir la console')}")
        return 1
    print(f"[pipeline] artefact modèle : s3://{SETTINGS.bucket}/models/…/model.tar.gz")
    print(f"[pipeline] version au registre : groupe {MODEL_GROUP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
