"""Endpoint LLM sur SageMaker (conteneur Hugging Face), à la place de Bedrock.

Bedrock n'est pas activé dans le compte Learner Lab (politique IAM sans
``bedrock:*``). Le sujet autorise explicitement « Hugging Face / OpenLLM sur
SageMaker » : on déploie donc un petit modèle instruct dans un conteneur HF
managé, en **serverless** (aucun coût quand il ne sert pas).

Usage::

    python -m infra.sagemaker.llm_endpoint --create --wait   # déploie (~8 min)
    python -m infra.sagemaker.llm_endpoint --invoke "résume: ..."
    python -m infra.sagemaker.llm_endpoint --delete          # supprime tout
"""

from __future__ import annotations

import argparse
import json
import time

from infra.sagemaker.launch import _sm_client
from retail_forecast.config import SETTINGS

# Conteneur d'inférence Hugging Face publié par AWS (us-east-1).
IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-inference:2.6-transformers4.51-cpu-py312-ubuntu22.04"
)
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME = "retail-llm-hf"
ENDPOINT_NAME = "retail-llm-hf"
CONFIG_NAME = "retail-llm-hf-config"
MEMORY_MB = 6144  # maximum du serverless ; le modèle pèse ~1 Go
MAX_NEW_TOKENS = 300


def _runtime():
    import boto3

    kwargs: dict = {"region_name": SETTINGS.region}
    if SETTINGS.s3_access_key:
        kwargs["aws_access_key_id"] = SETTINGS.s3_access_key
        kwargs["aws_secret_access_key"] = SETTINGS.s3_secret_key
    return boto3.client("sagemaker-runtime", **kwargs)


def create(sm, wait: bool) -> None:
    """Crée Model + EndpointConfig (serverless) + Endpoint."""
    sm.create_model(
        ModelName=MODEL_NAME,
        PrimaryContainer={
            "Image": IMAGE,
            "Environment": {
                "HF_MODEL_ID": MODEL_ID,
                "HF_TASK": "text-generation",
                "SM_NUM_GPUS": "0",
            },
        },
        ExecutionRoleArn=SETTINGS.role_arn,
    )
    print(f"[llm] model créé : {MODEL_NAME} ({MODEL_ID})")

    sm.create_endpoint_config(
        EndpointConfigName=CONFIG_NAME,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": MODEL_NAME,
                "ServerlessConfig": {"MemorySizeInMB": MEMORY_MB, "MaxConcurrency": 1},
            }
        ],
    )
    print(f"[llm] config créée : serverless {MEMORY_MB} Mo, concurrence 1")

    sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=CONFIG_NAME)
    print(f"[llm] endpoint {ENDPOINT_NAME} en cours de création (~5-10 min)")

    if not wait:
        return
    started = time.time()
    for _ in range(90):
        status = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)["EndpointStatus"]
        if status in {"InService", "Failed"}:
            break
        time.sleep(10)
    print(f"[llm] statut {status} en {(time.time() - started) / 60:.1f} min")
    if status != "InService":
        reason = sm.describe_endpoint(EndpointName=ENDPOINT_NAME).get(
            "FailureReason", "inconnue"
        )
        raise SystemExit(f"[llm] échec : {reason}")


def invoke(prompt: str) -> str:
    """Appelle l'endpoint et renvoie le texte généré."""
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False},
    }
    print("[llm] invocation (première requête = réveil du conteneur, 3-5 min)")
    started = time.time()
    response = _runtime().invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    body = json.loads(response["Body"].read())
    text = body[0]["generated_text"] if isinstance(body, list) else str(body)
    print(f"[llm] réponse en {time.time() - started:.1f} s")
    return text.strip()


def delete(sm) -> None:
    """Supprime endpoint, config et model (dans cet ordre)."""
    for fn, kwargs in (
        (sm.delete_endpoint, {"EndpointName": ENDPOINT_NAME}),
        (sm.delete_endpoint_config, {"EndpointConfigName": CONFIG_NAME}),
        (sm.delete_model, {"ModelName": MODEL_NAME}),
    ):
        try:
            fn(**kwargs)
            print(f"[llm] supprimé : {list(kwargs.values())[0]}")
        except sm.exceptions.ClientError as exc:
            print(f"[llm] déjà absent : {list(kwargs.values())[0]} ({exc.response['Error']['Code']})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Endpoint LLM HF sur SageMaker.")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--invoke", default="")
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args(argv)

    if not SETTINGS.role_arn:
        raise SystemExit("SAGEMAKER_ROLE_ARN manquant (.env)")

    sm = _sm_client()
    if args.create:
        create(sm, args.wait)
    if args.invoke:
        print(invoke(args.invoke))
    if args.delete:
        delete(sm)
    if not (args.create or args.invoke or args.delete):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
