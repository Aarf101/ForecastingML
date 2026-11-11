#!/usr/bin/env bash
# Pipeline complet, exécuté intégralement dans AWS.
#
# Enchaîne : S3 -> (Kinesis/Firehose/Lambda) -> Glue -> Training -> gate WAPE ->
# Registry -> Batch Transform -> Athena. S'arrête à la première erreur.
#
# Usage :   bash scripts/aws_full_run.sh [--skip-ingest] [--skip-orchestration]
#
# Prérequis : .env sur AWS (S3_ENDPOINT_URL et AWS_ACCESS_KEY_ID/SECRET vides),
#             session Learner Lab active, job Glue déjà créé dans la console.
set -euo pipefail

PY=".venv/bin/python"
[ -x "$PY" ] || PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY="python3"
[ -x "$PY" ] || PY="python"
BUCKET=$(grep -E '^S3_BUCKET=' .env | cut -d= -f2-)
SKIP_INGEST=0
SKIP_ORCH=0
for arg in "$@"; do
  case "$arg" in
    --skip-ingest) SKIP_INGEST=1 ;;
    --skip-orchestration) SKIP_ORCH=1 ;;
    *) echo "argument inconnu : $arg"; exit 2 ;;
  esac
done

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$1"; }
fail() { printf '\033[1;31m[échec] %s\033[0m\n' "$1"; exit 1; }

step "0. vérification de la session AWS"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) \
  || fail "session expirée : Start Lab puis recopier les 3 valeurs dans ~/.aws/credentials [lab]"
echo "compte $ACCOUNT · bucket $BUCKET"
aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null \
  || fail "bucket $BUCKET introuvable : lancer 'make aws-setup'"

step "1. dépôt du code d'entraînement dans S3"
$PY -m infra.sagemaker.launch --upload-only

if [ "$SKIP_INGEST" -eq 0 ]; then
  step "3. ingestion : morceaux JSONL -> S3"
  $PY scripts/prepare_replay_chunks.py --rows 20000 --delete

  step "4. ingestion : Kinesis + Firehose + Lambda"
  $PY -m infra.aws.kinesis_ingest up

  step "5. ingestion : rejeu des événements"
  $PY -m infra.aws.kinesis_ingest run --max-seconds 300

  step "6. attente du flush Firehose (60 s + marge)"
  for i in $(seq 1 12); do
    N=$(aws s3api list-objects-v2 --bucket "$BUCKET" --prefix raw/sales/ \
          --query 'KeyCount' --output text 2>/dev/null || echo 0)
    echo "  raw/sales/ : $N objet(s)"
    [ "$N" -gt 0 ] && break
    sleep 20
  done
  [ "$N" -gt 0 ] || fail "aucun objet livré par Firehose : voir les erreurs dans raw/_errors/"
fi

if [ "$SKIP_ORCH" -eq 0 ]; then
  step "7. orchestration : job Glue + Step Functions + EventBridge"
  $PY -m infra.aws.orchestrate up

  aws glue get-job --job-name retail-etl-raw-to-curated >/dev/null 2>&1 \
    || fail "job Glue absent après 'orchestrate up'"

  step "8. exécution Glue -> Training -> gate WAPE -> Registry"
  $PY -m infra.aws.orchestrate run
fi

step "9. Batch Transform -> forecast/"
$PY -m infra.sagemaker.batch_transform

step "10. Athena : tables + KPI"
$PY -m infra.athena.setup

step "état final"
$PY -m infra.aws.kinesis_ingest status || true
$PY -m infra.aws.orchestrate status || true
echo
echo "Terminé. Penser à : make aws-ingest-down && make aws-pipeline-down && make aws-teardown-go"
