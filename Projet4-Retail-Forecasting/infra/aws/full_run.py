"""Pipeline complet, exécutable **depuis l'intérieur d'AWS** (sans ``make``).

Ce module est l'équivalent Python de ``scripts/aws_full_run.sh`` : il ne dépend ni
de bash ni du Makefile, donc il tourne tel quel dans le terminal d'une instance
SageMaker, dans un conteneur, ou depuis un notebook — là où seul Python est
disponible.

Usage (dans le terminal Jupyter de l'instance) :
    python -m infra.aws.full_run                    # tout
    python -m infra.aws.full_run --skip-ingest      # réutilise raw/ existant
    python -m infra.aws.full_run --only orchestrate  # une seule étape
    python -m infra.aws.full_run --list             # liste les étapes

Chaque étape est un sous-processus (``python -m …``) : une erreur arrête tout et
renvoie le code de sortie de l'étape fautive.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import boto3

from retail_forecast.config import SETTINGS

REGION = SETTINGS.region


def _python(*args: str) -> int:
    """Exécute ``python -m <args>`` avec l'interpréteur courant."""
    cmd = [sys.executable, "-m", *args]
    print(f"  $ {' '.join(cmd[1:])}")
    return subprocess.call(cmd)


def _session_ok() -> None:
    sts = boto3.client("sts", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)
    try:
        who = sts.get_caller_identity()
        print(f"  compte {who['Account']} · bucket {SETTINGS.bucket}")
        s3.head_bucket(Bucket=SETTINGS.bucket)
        print("  bucket accessible")
    except Exception as exc:  # noqa: BLE001 - message d'aide plutôt qu'une trace
        detail = str(exc).splitlines()[0]
        raise SystemExit(
            f"[échec] credentials ou bucket indisponible : {detail}\n"
            "  - session expirée ? relancer le lab puis exporter les identifiants\n"
            "  - bucket absent ? lancer le script aws_setup (ou make aws-setup)\n"
            "  - dans AWS (instance/CloudShell), boto3 utilise le rôle : retirer\n"
            "    AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY de l'environnement"
        )


def _wait_for_batches(timeout: int = 240) -> int:
    """Attend que Firehose ait déposé au moins un objet dans ``raw/sales/``."""
    s3 = boto3.client("s3", region_name=REGION)
    deadline = time.monotonic() + timeout
    count = 0
    while time.monotonic() < deadline:
        resp = s3.list_objects_v2(Bucket=SETTINGS.bucket, Prefix="raw/sales/")
        count = resp.get("KeyCount", 0)
        print(f"  raw/sales/ : {count} objet(s)")
        if count:
            return count
        time.sleep(20)
    return count


def _report() -> None:
    """État final : ingestion + orchestration."""
    subprocess.call([sys.executable, "-m", "infra.aws.kinesis_ingest", "status"])
    subprocess.call([sys.executable, "-m", "infra.aws.orchestrate", "status"])


# nom -> (fonction, description)
STEPS = {
    "session": (_session_ok, "vérifie les credentials et le bucket"),
    "code": (lambda: _python("infra.sagemaker.launch", "--upload-only"),
             "dépose sourcedir.tar.gz dans S3"),
    "chunks": (lambda: _python("scripts.prepare_replay_chunks", "--rows", "20000", "--delete"),
               "parquet -> morceaux JSONL gzip dans replay/"),
    "ingest-up": (lambda: _python("infra.aws.kinesis_ingest", "up"),
                  "crée Kinesis + Firehose + Lambda producteur"),
    "ingest-run": (lambda: _python("infra.aws.kinesis_ingest", "run", "--max-seconds", "300"),
                   "rejoue les événements dans Kinesis"),
    "ingest-wait": (lambda: _wait_for_batches(), "attend le flush Firehose"),
    "orchestrate-up": (lambda: _python("infra.aws.orchestrate", "up"),
                       "job Glue + state machine + EventBridge + Lambda de gate"),
    "orchestrate-run": (lambda: _python("infra.aws.orchestrate", "run"),
                        "Glue -> Training -> gate WAPE -> Registry"),
    "transform": (lambda: _python("infra.sagemaker.batch_transform"),
                  "Batch Transform -> forecast/"),
    "athena": (lambda: _python("infra.athena.setup"), "tables externes + KPI"),
}

ORDER = list(STEPS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="liste les étapes")
    parser.add_argument("--only", default="", help="n'exécuter qu'une étape")
    parser.add_argument("--skip-ingest", action="store_true", help="sauter l'ingestion")
    parser.add_argument("--skip-orchestration", action="store_true", help="sauter Step Functions")
    parser.add_argument("--skip-transform", action="store_true")
    parser.add_argument("--skip-athena", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        for name in ORDER:
            print(f"  {name:<16} {STEPS[name][1]}")
        return 0

    if not SETTINGS.bucket:
        raise SystemExit("S3_BUCKET vide dans l'environnement/.env")

    if args.only:
        if args.only not in STEPS:
            raise SystemExit(f"étape inconnue : {args.only} (voir --list)")
        print(f"\n== {args.only} : {STEPS[args.only][1]} ==")
        return STEPS[args.only][0]() or 0

    skip: set[str] = {"session"}  # déjà fait juste en dessous
    if args.skip_ingest:
        skip |= {"chunks", "ingest-up", "ingest-run", "ingest-wait"}
    if args.skip_orchestration:
        skip |= {"orchestrate-up", "orchestrate-run"}
    if args.skip_transform:
        skip.add("transform")
    if args.skip_athena:
        skip.add("athena")

    print("\n== session : credentials et bucket ==")
    _session_ok()

    for name in ORDER:
        if name in skip:
            print(f"\n-- {name} : sauté")
            continue
        print(f"\n== {name} : {STEPS[name][1]} ==")
        code = STEPS[name][0]() or 0
        if code:
            print(f"[échec] étape '{name}' code {code} — arrêt")
            return code

    print("\n== état final ==")
    _report()
    print("\nTerminé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
