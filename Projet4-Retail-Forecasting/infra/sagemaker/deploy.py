"""Endpoint d'inférence du modèle approuvé (serverless, coût quasi nul au repos).

Couvre le mot « **déploiement** » du sujet : le Batch Transform produit les
prévisions de masse, cet endpoint répond à une requête unitaire (une série, un
horizon) — utile pour vérifier un modèle ou alimenter une application.

Choix serverless plutôt qu'une instance permanente : `ml.m5.large` coûte ~0,11 €/h
même sans trafic, alors que le serverless facture à la milliseconde d'inférence et
redescend à zéro. Dans un Learner Lab dont la session dure 4 h, c'est la
différence entre « gratuit » et « 0,45 € par oubli ».

Le modèle servi est **celui du registre, statut `Approved`** — jamais un chemin
codé en dur : si la version approuvée change, `down` puis `up` suffit.

Usage::

    python -m infra.sagemaker.deploy up            # crée modèle + endpoint
    python -m infra.sagemaker.deploy invoke        # 4 prévisions de démonstration
    python -m infra.sagemaker.deploy status
    python -m infra.sagemaker.deploy down          # supprime tout (coût -> 0)
"""

from __future__ import annotations

import argparse
import json

from infra.sagemaker.batch_transform import (
    approved_model,
    build_inference_rows,
    ensure_model,
    model_metadata,
    read_curated,
)
from infra.sagemaker.pipeline import _boto_session
from retail_forecast.config import SETTINGS
from retail_forecast.ingest.storage import s3_client

ENDPOINT = "retail-demand-lgbm"
MODEL_NAME = "retail-demand-lgbm-model"
CONFIG_NAME = "retail-demand-lgbm-config"
MEMORY_MB = 2048  # 2 Go : suffisant (modèles LightGBM de quelques Mo)
MAX_CONCURRENCY = 2
HORIZONS = list(range(1, 17))


def _sm():
    return _boto_session().client("sagemaker")


def _runtime():
    return _boto_session().client("sagemaker-runtime")


def up(wait: bool = True, model_url: str | None = None) -> int:
    """Crée le modèle, la configuration serverless et l'endpoint."""
    sm = _sm()
    approved_url, image, version = approved_model(sm)
    if model_url and model_url != approved_url:
        raise ValueError("l'endpoint doit utiliser exactement l'artefact actif validé")
    model_url = approved_url
    print(f"[deploy] version active v{version} : {model_url}")

    ensure_model(sm, MODEL_NAME, image, model_url, serverless=True)

    try:
        sm.describe_endpoint_config(EndpointConfigName=CONFIG_NAME)
        print(f"[deploy] configuration existante : {CONFIG_NAME}")
    except sm.exceptions.ClientError:
        sm.create_endpoint_config(
            EndpointConfigName=CONFIG_NAME,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": MODEL_NAME,
                    "ServerlessConfig": {
                        "MemorySizeInMB": MEMORY_MB,
                        "MaxConcurrency": MAX_CONCURRENCY,
                    },
                }
            ],
        )
        print(f"[deploy] configuration serverless : {MEMORY_MB} Mo, "
              f"concurrence {MAX_CONCURRENCY}")

    try:
        desc = sm.describe_endpoint(EndpointName=ENDPOINT)
        print(f"[deploy] endpoint existant : {desc['EndpointStatus']}")
    except sm.exceptions.ClientError:
        sm.create_endpoint(EndpointName=ENDPOINT, EndpointConfigName=CONFIG_NAME)
        print(f"[deploy] endpoint demandé : {ENDPOINT} (1-5 min)")

    if not wait:
        return 0

    sm.get_waiter("endpoint_in_service").wait(
        EndpointName=ENDPOINT,
        WaiterConfig={"Delay": 15, "MaxAttempts": 40},
    )
    print(f"[deploy] InService · {ENDPOINT}")
    print("[deploy] suite : python -m infra.sagemaker.deploy invoke")
    return 0


def sample_rows():
    """4 lignes (une par horizon) pour la première série disponible.

    Les features sont recalculées à partir de ``curated/sales/`` : on envoie au
    modèle exactement les colonnes qu'il attend, dans le même ordre.
    """
    s3 = s3_client()
    sm = _sm()
    model_url, _image, _version = approved_model(sm)
    metadata = model_metadata(s3, model_url)
    features = metadata["features"]
    sales = read_curated(s3)
    rows = build_inference_rows(
        sales, features, metadata["horizons"], levels=metadata.get("categorical_levels")
    )
    premiere = rows.iloc[0]
    selection = rows[
        (rows["store_id"] == premiere["store_id"])
        & (rows["item_id"] == premiere["item_id"])
    ]
    return selection[["store_id", "item_id", "horizon", *features]]


def invoke() -> int:
    """Envoie une requête réelle à l'endpoint et affiche les prévisions."""
    rows = sample_rows()
    body = "\n".join(json.dumps(row, default=float) for row in rows.to_dict("records"))
    response = _runtime().invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Accept="application/json",
        Body=body,
    )
    predictions = json.loads(response["Body"].read())
    premiere = rows.iloc[0]
    print(f"[deploy] série {premiere['store_id']} / {premiere['item_id']}")
    for row, pred in zip(rows.to_dict("records"), predictions, strict=False):
        print(f"  J+{int(row['horizon']):<3} -> {float(pred):8.2f} unités")
    return 0


def status() -> int:
    sm = _sm()
    try:
        desc = sm.describe_endpoint(EndpointName=ENDPOINT)
    except sm.exceptions.ClientError:
        print(f"[deploy] endpoint absent : {ENDPOINT}")
        return 1
    print(f"[deploy] {desc['EndpointName']} · {desc['EndpointStatus']} · "
          f"config {desc['EndpointConfigName']}")
    return 0


def down() -> int:
    """Supprime endpoint, configuration et modèle (facturation à zéro)."""
    sm = _sm()
    try:
        sm.delete_endpoint(EndpointName=ENDPOINT)
    except sm.exceptions.ClientError as exc:
        if exc.response['Error']['Code'] != 'ValidationException':
            raise
    else:
        # L'endpoint doit être supprimé avant ses ressources dépendantes.
        sm.get_waiter('endpoint_deleted').wait(
            EndpointName=ENDPOINT, WaiterConfig={'Delay': 10, 'MaxAttempts': 60})
        print('[deploy] endpoint supprimé')
    for label, appel in (
        ("configuration", lambda: sm.delete_endpoint_config(EndpointConfigName=CONFIG_NAME)),
        ("modèle", lambda: sm.delete_model(ModelName=MODEL_NAME)),
    ):
        try:
            appel()
            print(f"[deploy] {label} supprimé")
        except sm.exceptions.ClientError as exc:
            if exc.response['Error']['Code'] != 'ValidationException':
                raise
            print(f"[deploy] {label} absent")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Endpoint serverless du modèle approuvé (LightGBM)."
    )
    parser.add_argument("action", nargs="?", default="status",
                        choices=["up", "status", "invoke", "down"])
    parser.add_argument("--no-wait", action="store_true",
                        help="rendre la main sans attendre InService")
    parser.add_argument("--model-url", default=None,
                        help="artefact S3 imposé (sinon : version approuvée du registre)")
    args = parser.parse_args(argv)

    if not SETTINGS.bucket:
        raise SystemExit("S3_BUCKET doit être renseigné (.env)")

    if args.action == "up":
        return up(wait=not args.no_wait, model_url=args.model_url)
    if args.action == "invoke":
        return invoke()
    if args.action == "down":
        return down()
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
