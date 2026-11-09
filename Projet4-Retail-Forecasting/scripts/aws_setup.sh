#!/usr/bin/env bash
# Première mise en route sur le compte AWS Learner Lab.
#
# À lancer APRÈS avoir collé ses identifiants de session dans ~/.aws/credentials
# sous la section [lab] (voir docs/learner_lab.md). Le script :
#   1. vérifie que les identifiants fonctionnent (et affiche le numéro de compte) ;
#   2. crée le bucket S3 du projet s'il n'existe pas ;
#   3. y dépose l'historique local (data/raw/sales.parquet) ;
#   4. affiche le bloc à coller dans .env pour basculer le projet sur AWS.
#
# Usage : bash scripts/aws_setup.sh [nom-du-binome]
set -euo pipefail

PROFILE="${AWS_PROFILE:-lab}"
SUFFIX="${1:-$(whoami)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data/raw/sales.parquet"
REGION="${AWS_REGION:-us-east-1}"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "1/4 · vérification des identifiants (profil $PROFILE)"
if ! ACCOUNT=$(AWS_PROFILE="$PROFILE" aws sts get-caller-identity --query Account --output text 2>/dev/null); then
  echo "[ERREUR] pas d'identifiants valides pour le profil '$PROFILE'." >&2
  echo "         Learner Lab -> AWS Details -> AWS CLI -> Show, puis coller les" >&2
  echo "         3 valeurs dans ~/.aws/credentials sous [lab]. Voir docs/learner_lab.md." >&2
  exit 1
fi
ARN=$(AWS_PROFILE="$PROFILE" aws sts get-caller-identity --query Arn --output text)
echo "  compte : $ACCOUNT"
echo "  rôle   : $ARN"
case "$ARN" in
  *LabRole*) echo "  [OK] LabRole détecté" ;;
  *) echo "  [ATTENTION] ce n'est pas LabRole : vérifier SAGEMAKER_ROLE_ARN" >&2 ;;
esac

BUCKET="retail-forecast-${ACCOUNT}-${SUFFIX}"

say "2/4 · bucket S3 : $BUCKET"
if AWS_PROFILE="$PROFILE" aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "  déjà présent"
else
  if [ "$REGION" = "us-east-1" ]; then
    AWS_PROFILE="$PROFILE" aws s3api create-bucket --bucket "$BUCKET" --region us-east-1 >/dev/null
  else
    AWS_PROFILE="$PROFILE" aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
      --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
  fi
  # Chiffrement côté serveur par défaut : exigé par la plupart des politiques.
  AWS_PROFILE="$PROFILE" aws s3api put-bucket-encryption --bucket "$BUCKET" \
    --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}' >/dev/null
  echo "  créé (chiffrement AES256 activé)"
fi

say "3/4 · dépôt de l'historique"
if [ -f "$DATA" ]; then
  AWS_PROFILE="$PROFILE" aws s3 cp "$DATA" "s3://$BUCKET/raw/historique/sales.parquet" --only-show-errors
  echo "  s3://$BUCKET/raw/historique/sales.parquet ($(du -h "$DATA" | cut -f1))"
else
  echo "  [ATTENTION] $DATA absent — lancer 'make data' d'abord" >&2
fi

say "4/4 · bloc à coller dans .env"
cat <<EOF
AWS_REGION=$REGION
S3_BUCKET=$BUCKET
S3_ENDPOINT_URL=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_PROFILE=$PROFILE
SAGEMAKER_ROLE_ARN=arn:aws:iam::${ACCOUNT}:role/LabRole
EOF

cat <<'EOF'

Rappel : un endpoint ou une app Studio allumés coûtent à l'heure.
Arrêt : AWS_PROFILE=lab aws sagemaker list-endpoints --query 'Endpoints[].EndpointName'
EOF
