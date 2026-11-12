"""Zéro -> ingestion testée dans AWS, en **une seule commande**.

Crée (ou réutilise) tout ce qu'il faut pour voir un événement traverser
Kafka -> Firehose -> S3, puis lance l'ETL Glue si demandé.

    1. session + bucket
    2. bucket S3 + dataset de référence dans raw/historique/
    3. delivery stream Firehose retail-sales-direct (DirectPut)
    4. instance EC2 + Kafka (KRaft)
    5. pip + dépôt dans /opt/retail sur l'instance
    6. attente du conteneur Kafka
    7. producteur 20 000 messages + pont vers Firehose
    8. attente de la livraison S3 (tampon Firehose 60 s)
    9. (option) job Glue -> curated/
   10. (option) arrêt de l'EC2 et suppression du stream

Usage :

    python -m infra.aws.ingest_all                  # tout, avec ETL
    python -m infra.aws.ingest_all --skip-glue      # ingestion seule
    python -m infra.aws.ingest_all --down           # nettoie à la fin
    python -m infra.aws.ingest_all --limit 5000

Chaque étape est idempotente : relancer ne recrée que ce qui manque.
"""

from __future__ import annotations

import argparse
import time

import boto3

from infra.aws import ec2_kafka, firehose_direct, orchestrate
from infra.aws.setup import ensure_bucket, ensure_dataset
from retail_forecast.config import SETTINGS

REGION = SETTINGS.region
PREFIX = "raw/sales/"


def _client(service: str):
    return boto3.client(service, region_name=REGION)


def step(num: int, total: int, label: str) -> None:
    print(f"\n=== {num}/{total} · {label} ===", flush=True)


def check_session() -> str:
    """Vérifie les credentials et le bucket ; renvoie le compte."""
    account = _client("sts").get_caller_identity()["Account"]
    print(
        f"[session] compte {account} · région {REGION} · bucket {SETTINGS.bucket or '—'}",
        flush=True,
    )
    if not SETTINGS.bucket:
        raise SystemExit("S3_BUCKET vide : renseigner .env ou exporter la variable")
    return account


def objects() -> dict[str, int]:
    """Objets présents sous raw/sales/ -> {clé: taille}."""
    s3 = _client("s3")
    out: dict[str, int] = {}
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=SETTINGS.bucket, Prefix=PREFIX
    ):
        for obj in page.get("Contents", []):
            out[obj["Key"]] = obj["Size"]
    return out


def wait_delivery(before: dict[str, int], timeout: int = 300) -> list[tuple[str, int]]:
    """Attend qu'un nouvel objet apparaisse sous raw/sales/."""
    print(f"[wait] tampon Firehose (≤ 60 s) puis livraison S3… délai max {timeout}s")
    deadline = time.monotonic() + timeout
    tick = 0
    while time.monotonic() < deadline:
        new = {k: v for k, v in objects().items() if k not in before}
        if new:
            for key, size in sorted(new.items()):
                print(f"[s3] {key}  ({size:,} octets)")
            return sorted(new.items())
        tick += 10
        print(f"[wait] rien encore… {tick}s (Firehose tamponne 60 s)", flush=True)
        time.sleep(10)
    raise RuntimeError(
        f"aucun objet livré sous {PREFIX} en {timeout}s — voir "
        "s3://…/raw/_errors/ et l'onglet Monitoring du stream"
    )


def run_glue(wait: bool = True) -> str:
    """Déploie (si besoin) puis exécute le job Glue, et attend son statut."""
    c = orchestrate._clients()
    role = orchestrate._role_arn(c)
    orchestrate.deploy_glue_job(c, role)

    glue = c["glue"]
    run_id = glue.start_job_run(
        JobName=orchestrate.GLUE_JOB, Arguments={"--BUCKET": SETTINGS.bucket}
    )["JobRunId"]
    print(f"[glue] run {run_id} démarré")
    if not wait:
        return run_id

    for _ in range(120):
        state = glue.get_job_run(JobName=orchestrate.GLUE_JOB, RunId=run_id)["JobRun"][
            "JobRunState"
        ]
        if state == "SUCCEEDED":
            print("[glue] SUCCEEDED")
            return run_id
        if state in ("FAILED", "STOPPED", "TIMEOUT"):
            raise RuntimeError(f"job Glue {state} — voir /aws-glue/jobs/logs-v2")
        time.sleep(10)
    raise RuntimeError("job Glue toujours en cours après 20 min")


def print_setup_summary(instance_id: str, private_ip: str, stream: str) -> None:
    """Affiche ce qui a été créé et les commandes/clics à faire ensuite."""
    print("\n" + "=" * 64)
    print("ENVIRONNEMENT PRÊT — aucune donnée n'a été envoyée")
    print("=" * 64)
    print(f"  bucket      : s3://{SETTINGS.bucket}/")
    print(f"  dataset     : s3://{SETTINGS.bucket}/raw/historique/sales.parquet")
    print(f"  Firehose    : {stream} (DirectPut) -> raw/sales/")
    print(f"  EC2 + Kafka : {instance_id} · {private_ip}:9092 · dépôt dans /opt/retail")
    print("\nTester sans Kafka, en clics (console) :")
    print(f"  1. Firehose -> {stream} -> bouton 'Test with demo data' -> Send")
    print("  2. attendre 60 s -> S3 -> raw/sales/ -> objet AAAA/MM/JJ/HH/*.gz")
    print("  3. Glue -> retail-etl-raw-to-curated -> Run")
    print("  4. S3 -> curated/sales/ (4 Parquet) · Athena -> retail_forecast")
    print("\nTester avec Kafka (terminal du navigateur) :")
    print("  EC2 -> Connect -> Session Manager -> Connect, puis :")
    print(f"  cd /opt/retail && export PYTHONPATH=/opt/retail/src \\\n"
          f"      S3_BUCKET={SETTINGS.bucket} AWS_REGION={REGION}")
    print(f"  python3 -m retail_forecast.ingest.producer --limit 20000 "
          f"--bootstrap {private_ip}:9092")
    print(f"  python3 -m retail_forecast.ingest.kafka_to_firehose --bootstrap "
          f"{private_ip}:9092 --stream {stream} --max-messages 20000")
    print("\nNettoyage : python -m infra.aws.ingest_all --down")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Crée tout l'environnement AWS d'ingestion et le teste de bout en bout."
    )
    parser.add_argument("--limit", type=int, default=20000, help="messages à rejouer")
    parser.add_argument("--stream", default=firehose_direct.STREAM)
    parser.add_argument("--skip-glue", action="store_true", help="s'arrêter après S3")
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="créer l'environnement et s'arrêter avant l'ingestion (pour tester via la console)",
    )
    parser.add_argument("--skip-dataset", action="store_true", help="ne pas vérifier le dataset")
    parser.add_argument("--regenerate-data", action="store_true")
    parser.add_argument("--down", action="store_true", help="supprimer EC2 + Firehose à la fin")
    parser.add_argument("--delivery-timeout", type=int, default=300)
    args = parser.parse_args(argv)

    total = 6 if args.setup_only else (7 if args.skip_glue else 8)

    step(1, total, "session et bucket")
    check_session()

    step(2, total, "bucket + dataset de référence")
    ensure_bucket()
    if args.skip_dataset:
        print("[data] vérification ignorée (--skip-dataset)")
    else:
        ensure_dataset(regenerate=args.regenerate_data)

    step(3, total, f"delivery stream Firehose {args.stream}")
    firehose_direct.ensure(name=args.stream)

    step(4, total, "instance EC2 + Kafka")
    info = ec2_kafka.up()
    instance_id, private_ip = info["instance_id"], info["private_ip"]

    step(5, total, "dépendances Python + dépôt sur l'instance")
    ec2_kafka.bootstrap(instance_id, SETTINGS.bucket)

    step(6, total, "attente du conteneur Kafka")
    ec2_kafka.wait_kafka(instance_id)

    if args.setup_only:
        print_setup_summary(instance_id, private_ip, args.stream)
        return 0

    step(7, total, f"ingestion : {args.limit:,} messages -> Firehose -> S3")
    before = objects()
    if before:
        print(f"[s3] {len(before)} objet(s) déjà sous {PREFIX} (référence)", flush=True)
    ec2_kafka.run_ingest(instance_id, private_ip, SETTINGS.bucket, args.limit, args.stream)
    delivered = wait_delivery(before, args.delivery_timeout)

    if not args.skip_glue:
        step(8, total, "ETL Glue -> curated/")
        run_glue()

    print("\n" + "=" * 60)
    print("INGESTION VALIDÉE DANS AWS")
    print("=" * 60)
    print(f"  Kafka      : {private_ip}:9092 · instance {instance_id}")
    print(f"  Firehose   : {args.stream} (DirectPut) -> s3://{SETTINGS.bucket}/{PREFIX}")
    for key, size in delivered:
        print(f"  objet livré: {key} ({size:,} o)")
    if not args.skip_glue:
        print(f"  ETL        : s3://{SETTINGS.bucket}/curated/sales/")

    if args.down:
        print("\n[nettoyage] …")
        firehose_direct.down(name=args.stream)
        ec2_kafka.down()
    else:
        print("  Nettoyage : python -m infra.aws.ingest_all --down")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
