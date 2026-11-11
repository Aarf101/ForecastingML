"""Handler Lambda — rejoue le dataset dans Kinesis (ingestion 100 % AWS).

Rôle : remplacer le ``producer.py`` local. Le Lambda lit des morceaux JSONL gzip
depuis S3 (``REPLAY_PREFIX``) et les publie dans un stream Kinesis par lots de
500 enregistrements (``PutRecords``). Une checkpoint dans S3 mémorise le
couple (chunk, offset) pour qu'une nouvelle invocation reprenne là où la
précédente s'est arrêtée : l'invocation peut donc être interrompue sans perte ni
double envoi.

Variables d'environnement attendues :
  BUCKET          bucket du lac de données
  STREAM_NAME     nom du stream Kinesis cible
  REPLAY_PREFIX   préfixe S3 des morceaux à rejouer (défaut ``replay/sales``)
  CHECKPOINT_KEY  objet S3 de checkpoint (défaut ``replay/_state/checkpoint.json``)

Aucune dépendance externe : gzip/json/boto3, tous fournis par le runtime Lambda.
"""

import gzip
import io
import json
import os
import time

import boto3

BATCH = 500  # PutRecords : 500 enregistrements max par appel

_s3 = boto3.client("s3")
_kinesis = boto3.client("kinesis")


def _load_state(bucket: str, key: str) -> dict:
    """État (chunk, offset) ; 0/0 si le checkpoint n'existe pas encore."""
    try:
        body = _s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    except _s3.exceptions.NoSuchKey:
        return {"chunk": 0, "offset": 0}


def _save_state(bucket: str, key: str, state: dict) -> None:
    _s3.put_object(
        Bucket=bucket, Key=key, Body=json.dumps(state).encode("utf-8")
    )


def _list_chunks(bucket: str, prefix: str) -> list[str]:
    """Clés des morceaux ``*.json.gz``, triées."""
    keys: list[str] = []
    paginator = _s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json.gz"):
                keys.append(obj["Key"])
    return sorted(keys)


def _decode(body: bytes) -> list[dict]:
    """Un lot gzip JSONL -> liste de dicts (une vente par ligne)."""
    with gzip.open(io.BytesIO(body), "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _put_records(stream: str, records: list[dict]) -> int:
    """Publie un lot en Kinesis ; réessaie une fois les enregistrements en échec.

    Chaque enregistrement est terminé par ``\\n`` : Firehose concatène les
    enregistrements **tels quels**, donc sans ce séparateur l'objet S3 contient
    ``{…}{…}`` sur une seule ligne et devient illisible en JSON Lines.
    """
    entries = [
        {
            "Data": (json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"),
            "PartitionKey": str(r.get("store_id", "s0")),
        }
        for r in records
    ]
    resp = _kinesis.put_records(StreamName=stream, Records=entries)
    failed = resp.get("FailedRecordCount", 0)
    if failed:
        # Les échecs sont signalés par un ErrorCode dans la réponse : on renvoie
        # une seule fois ces enregistrements, suffisant pour du throttling passager.
        retry = [
            e for e, out in zip(entries, resp["Records"]) if out.get("ErrorCode")
        ]
        if retry:
            _kinesis.put_records(StreamName=stream, Records=retry)
    return len(entries)


def lambda_handler(event, context=None):
    """Rejoue autant de morceaux que possible avant la fin du temps imparti."""
    bucket = os.environ["BUCKET"]
    stream = os.environ["STREAM_NAME"]
    prefix = os.environ.get("REPLAY_PREFIX", "replay/sales")
    ck_key = os.environ.get("CHECKPOINT_KEY", "replay/_state/checkpoint.json")

    budget = float(event.get("max_seconds", 600))
    deadline = time.monotonic() + budget

    keys = _list_chunks(bucket, prefix)
    state = _load_state(bucket, ck_key)
    sent = 0
    done_chunks = 0

    while state["chunk"] < len(keys):
        if time.monotonic() >= deadline:
            break
        key = keys[state["chunk"]]
        records = _decode(_s3.get_object(Bucket=bucket, Key=key)["Body"].read())

        offset = state["offset"]
        while offset < len(records) and time.monotonic() < deadline:
            batch = records[offset:offset + BATCH]
            sent += _put_records(stream, batch)
            offset += len(batch)
            _save_state(bucket, ck_key, {"chunk": state["chunk"], "offset": offset})

        if offset >= len(records):
            state = {"chunk": state["chunk"] + 1, "offset": 0}
            _save_state(bucket, ck_key, state)
            done_chunks += 1
        else:
            break

    finished = state["chunk"] >= len(keys)
    print(
        f"[replay] {sent:,} événements publiés · {done_chunks} morceau(x) terminé(s) · "
        f"chunk {state['chunk']}/{len(keys)} · finished={finished}"
    )
    return {
        "sent": sent,
        "chunks_completed": done_chunks,
        "next_chunk": state["chunk"],
        "total_chunks": len(keys),
        "finished": finished,
    }
