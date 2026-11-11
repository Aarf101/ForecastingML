#!/usr/bin/env bash
# Dépose le code du projet dans S3 (une fois), pour pouvoir l'installer depuis
# l'intérieur du lab avec scripts/bootstrap.sh.
#
# Usage : bash scripts/upload_repo.sh
set -euo pipefail

BUCKET=$(grep -E '^S3_BUCKET=' .env | cut -d= -f2-)
[ -n "$BUCKET" ] || { echo "S3_BUCKET absent de .env"; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "== archive du code =="
tar czf "$TMP/repo.tar.gz" \
  --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='data/raw' --exclude='data/forecast' --exclude='.git' \
  src infra scripts dashboards pyproject.toml requirements.txt requirements-aws.txt Makefile .env.example

SIZE=$(du -h "$TMP/repo.tar.gz" | cut -f1)
echo "archive : $SIZE"

echo "== envoi vers s3://$BUCKET/code/repo.tar.gz =="
aws s3 cp "$TMP/repo.tar.gz" "s3://$BUCKET/code/repo.tar.gz"
echo "terminé — dans le lab : bash bootstrap.sh --run"
