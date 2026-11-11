#!/usr/bin/env bash
# Point d'entrée À EXÉCUTER DANS LE LAB (terminal Jupyter d'une instance
# SageMaker, ou CloudShell). Aucun accès au poste de travail n'est nécessaire.
#
#   bash bootstrap.sh                 # installe le code seulement
#   bash bootstrap.sh --setup         # installe puis crée TOUTE l'infrastructure
#   bash bootstrap.sh --setup --run   # ... puis exécute le pipeline complet
#
# Prérequis : le bucket S3 contient code/repo.tar.gz (déposé une fois, soit par
# `make aws-upload-repo` depuis le poste, soit en glissant l'archive dans S3
# depuis la console).
set -euo pipefail

BUCKET="${BUCKET:-retail-forecast-654654161593-ayoub}"
WORK="${WORK:-$HOME/retail}"
REGION="${REGION:-us-east-1}"
TARBALL="s3://$BUCKET/code/repo.tar.gz"

# Dans AWS, boto3 doit utiliser le rôle de l'instance : on retire d'éventuelles
# clés statiques de l'environnement, sinon elles prennent le dessus.
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN 2>/dev/null || true

export S3_BUCKET="$BUCKET"
export AWS_DEFAULT_REGION="$REGION"
export AWS_REGION="$REGION"

echo "== récupération du code depuis $TARBALL =="
mkdir -p "$WORK"
aws s3 cp "$TARBALL" /tmp/repo.tar.gz
tar xzf /tmp/repo.tar.gz -C "$WORK"
cd "$WORK"
echo "code extrait dans $WORK"

echo "== installation des dépendances =="
python -m pip install -q --upgrade pip
python -m pip install -q -e . -r requirements.txt
echo "installation terminée (S3_BUCKET=$S3_BUCKET, region=$AWS_REGION)"

for arg in "$@"; do
  case "$arg" in
    --setup)
      echo "== mise en place de l'infrastructure =="
      python -m infra.aws.setup
      ;;
    --run)
      echo "== exécution du pipeline =="
      python -m infra.aws.full_run
      ;;
    --inventory)
      python -m infra.aws.setup --inventory
      ;;
  esac
done

if [ "$#" -eq 0 ]; then
  cat <<EOF

Prêt. Depuis ce terminal :

  cd $WORK
  python -m infra.aws.setup --inventory    # que contient le compte ?
  python -m infra.aws.setup                # crée toute l'infrastructure
  python -m infra.aws.full_run --list      # étapes du pipeline
  python -m infra.aws.full_run             # pipeline complet

EOF
fi
