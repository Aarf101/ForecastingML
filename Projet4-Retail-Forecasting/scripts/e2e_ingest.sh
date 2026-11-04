#!/usr/bin/env bash
# Validation de bout en bout de la Phase 3, sans AWS :
#   Docker (Redpanda + s3mock) -> producer -> topic sales.events -> consumer
#   -> objets gzip dans le S3 local -> vérification du contenu.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
[ -f .env ] || cp .env.example .env

N=${N:-3000}          # messages à rejouer
BUCKET=${S3_BUCKET:-retail-forecast-local}

echo "== 1/5 démarrage des conteneurs =="
docker compose up -d

echo "== 2/5 attente de Kafka et du S3 local (max 60 s) =="
for _ in $(seq 1 30); do
  kafka_ok=0; s3_ok=0
  docker compose exec -T redpanda rpk cluster health 2>/dev/null | grep -q 'Healthy:.*true' && kafka_ok=1
  "$PY" - <<'PY' >/dev/null 2>&1 && s3_ok=1 || true
import urllib.request
urllib.request.urlopen("http://localhost:9090/", timeout=2)
PY
  if [ "$kafka_ok" = "1" ] && [ "$s3_ok" = "1" ]; then echo "   Kafka + S3 OK"; break; fi
  sleep 2
done

echo "== 3/5 production de $N messages =="
$PY -m retail_forecast.ingest.producer --limit "$N"

echo "== 4/5 consommation vers le S3 local =="
$PY -m retail_forecast.ingest.to_s3 --max-messages "$N"

echo "== 5/5 vérification des objets =="
$PY - <<PYEOF
from retail_forecast.config import SETTINGS
from retail_forecast.ingest.storage import s3_client

bucket = SETTINGS.bucket or "$BUCKET"
client = s3_client()
keys = []
paginator = client.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=bucket, Prefix="raw/sales/"):
    keys += [o["Key"] for o in page.get("Contents", [])]

total = sum(
    client.head_object(Bucket=bucket, Key=k)["ContentLength"] for k in keys
)
print(f"   bucket={bucket}  objets={len(keys)}  octets={total:,}")
for k in keys:
    print("   -", k)
assert keys, "aucun objet écrit : vérifier le S3 local / le consumer"
print("\\n[OK] Phase 3 validée : Kafka -> micro-lots gzip -> S3")
PYEOF

