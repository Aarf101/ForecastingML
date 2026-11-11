#!/usr/bin/env bash
# À exécuter DANS le lab (terminal Jupyter d'une instance SageMaker ou CloudShell).
#
# Récupère le code du projet depuis S3, l'installe, puis lance le pipeline.
# Aucun accès au poste de travail n'est nécessaire.
#
# Usage :
#   bash bootstrap.sh                 # installe seulement
#   bash bootstrap.sh --run           # installe puis lance le pipeline complet
#   bash bootstrap.sh --run --skip-ingest
#
# Prérequis : le bucket contient code/repo.tar.gz (déposé une fois avec
#             `make aws-upload-repo` depuis le poste de travail).
set -euo pipefail

BUCKET="${BUCKET:-retail-forecast-654654161593-ayoub}"
WORK="${WORK:-$HOME/retail}"
TARBALL="s3://$BUCKET/code/repo.tar.gz"

echo "== récupération du code depuis $TARBALL =="
mkdir -p "$WORK"
aws s3 cp "$TARBALL" /tmp/repo.tar.gz
tar xzf /tmp/repo.tar.gz -C "$WORK" --strip-components=1
cd "$WORK"
echo "code extrait dans $WORK"

echo "== installation des dépendances =="
python -m pip install -q --upgrade pip
python -m pip install -q -e . -r requirements.txt
echo "installation terminée"

if [ "${1:-}" = "--run" ]; then
  shift || true
  echo "== exécution du pipeline =="
  python -m infra.aws.full_run "$@"
else
  cat <<EOF

Prêt. Depuis ce terminal :

  cd $WORK
  python -m infra.aws.full_run --list        # voir les étapes
  python -m infra.aws.full_run               # pipeline complet
  python -m infra.aws.full_run --skip-ingest # sans rejouer Kinesis

EOF
fi
