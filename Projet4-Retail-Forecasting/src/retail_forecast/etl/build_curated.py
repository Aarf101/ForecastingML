"""ETL ``raw/`` -> ``curated/`` (miroir local du job Glue du diagramme).

Le job réel du diagramme est un Glue ETL PySpark. Faute de budget Learner Lab,
la même logique tourne ici en pandas : lire le brut (Parquet historique + lots
gzip JSONL d'événements), dédupliquer sur ``(store_id, item_id, date)``, caster
les types et écrire du Parquet dans ``curated/``. Le contrat de sortie est
identique, donc la suite du pipeline (features, training) ne voit pas la
différence.

Usage::

    python -m retail_forecast.etl.build_curated
    python -m retail_forecast.etl.build_curated --dry-run
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path

import pandas as pd

from retail_forecast.config import RAW_DIR, SETTINGS
from retail_forecast.ingest.storage import s3_client

CURATED_KEY = "curated/sales/sales.parquet"

DTYPES = {
    "store_id": "string",
    "item_id": "string",
    "category": "string",
    "unit_price": "float32",
    "promo": "int8",
    "is_holiday": "int8",
    "lead_time_days": "int16",
    "units": "int32",
    "stock_on_hand": "int32",
    "stockout": "int8",
}


def _read_jsonl_gz(body: bytes) -> pd.DataFrame:
    """Décompresse un lot gzip JSONL (une vente par ligne)."""
    with gzip.open(io.BytesIO(body), "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return pd.DataFrame(rows)


def read_raw_s3(client, bucket: str, prefix: str = "raw") -> pd.DataFrame:
    """Lit tout ``raw/`` dans S3 : fichiers Parquet et lots ``*.json.gz``."""
    paginator = client.get_paginator("list_objects_v2")
    frames: list[pd.DataFrame] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
            if key.endswith((".parquet", ".pq")):
                frames.append(pd.read_parquet(io.BytesIO(body)))
            elif key.endswith(".json.gz"):
                frames.append(_read_jsonl_gz(body))
            else:
                print(f"[etl] ignoré (format inconnu) : {key}")
                continue
            print(f"[etl] lu {key} ({len(body):,} octets)")
    if not frames:
        raise FileNotFoundError(f"aucun objet exploitable sous s3://{bucket}/{prefix}/")
    return pd.concat(frames, ignore_index=True)


def read_raw_local(directory: Path = RAW_DIR) -> pd.DataFrame:
    """Même chose en local (utile pour les tests et le mode hors-ligne)."""
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"aucun Parquet dans {directory}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Déduplication, typage et tri — les trois règles du job Glue."""
    missing = {"date", "store_id", "item_id", "units"} - set(df.columns)
    if missing:
        raise ValueError(f"colonnes manquantes dans le brut : {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()

    # Le flux d'événements peut rejouer une vente : on garde la dernière reçue.
    if "event_time" in out.columns:
        out["event_time"] = pd.to_datetime(out["event_time"], errors="coerce")
        out = out.sort_values("event_time")
    before = len(out)
    out = out.drop_duplicates(subset=["store_id", "item_id", "date"], keep="last")
    duplicates = before - len(out)

    for col, dtype in DTYPES.items():
        if col in out.columns:
            out[col] = out[col].astype(dtype)
    out = out.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    print(
        f"[etl] {before:,} lignes brutes -> {len(out):,} lignes propres"
        f" ({duplicates:,} doublons supprimés)"
    )
    print(f"[etl] période : {out['date'].min().date()} -> {out['date'].max().date()}")
    print(f"[etl] séries   : {len(out[['store_id', 'item_id']].drop_duplicates()):,}")
    return out


def write_curated_s3(client, bucket: str, df: pd.DataFrame, key: str = CURATED_KEY) -> str:
    """Écrit le Parquet dans ``curated/`` et renvoie l'URI S3."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="snappy")
    body = buf.getvalue()
    client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/parquet")
    print(f"[etl] écrit s3://{bucket}/{key} ({len(body) / 1e6:.2f} Mo)")
    return f"s3://{bucket}/{key}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ETL raw/ -> curated/ (Parquet).")
    parser.add_argument("--prefix", default="raw", help="préfixe S3 à lire")
    parser.add_argument("--key", default=CURATED_KEY, help="clé S3 de sortie")
    parser.add_argument("--local", action="store_true", help="lire data/raw/ au lieu de S3")
    parser.add_argument("--dry-run", action="store_true", help="ne rien écrire")
    args = parser.parse_args(argv)

    if not SETTINGS.bucket:
        raise SystemExit("S3_BUCKET non configuré (voir .env)")

    client = s3_client()
    raw = read_raw_local() if args.local else read_raw_s3(client, SETTINGS.bucket, args.prefix)
    curated = clean(raw)

    if args.dry_run:
        print(f"[etl] dry-run : {len(curated):,} lignes prêtes, rien écrit")
        return 0

    write_curated_s3(client, SETTINGS.bucket, curated, args.key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
