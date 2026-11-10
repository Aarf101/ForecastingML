#!/usr/bin/env bash
# Arrête tout ce qui coûte de l'argent dans le compte Learner Lab.
#
# Ne touche PAS au lac de données (S3) ni au Model Registry : ce sont les preuves
# du rendu. Supprime l'endpoint LLM (serverless = 0 € au repos, mais autant
# nettoyer) et arrête l'instance notebook (≈0,05 €/h si elle reste allumée).
#
# Usage :
#   bash scripts/aws_teardown.sh --dry-run   # montre ce qui serait supprimé
#   bash scripts/aws_teardown.sh --yes       # agit
set -euo pipefail

PROFILE="${AWS_PROFILE:-lab}"
REGION="${AWS_REGION:-us-east-1}"
ENDPOINT="${LLM_ENDPOINT:-retail-llm-hf}"
NOTEBOOK="${NOTEBOOK_NAME:-retail-forecast-nb}"
MODE="${1:---dry-run}"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
aws_() { AWS_PROFILE="$PROFILE" aws "$@" --region "$REGION"; }

say "1/4 · état actuel"
aws_ sagemaker list-endpoints --query 'Endpoints[].[EndpointName,EndpointStatus]' --output table || true
aws_ sagemaker list-notebook-instances --query 'NotebookInstances[].[NotebookInstanceName,NotebookInstanceStatus]' --output table || true

if [ "$MODE" != "--yes" ]; then
  say "mode --dry-run : rien n'a été supprimé"
  echo "Relancer avec : bash scripts/aws_teardown.sh --yes"
  exit 0
fi

say "2/4 · endpoint LLM"
aws_ sagemaker delete-endpoint --endpoint-name "$ENDPOINT" 2>/dev/null \
  && echo "  endpoint supprimé" || echo "  endpoint déjà absent"
for cfg in "${ENDPOINT}-config-v2" "${ENDPOINT}-config"; do
  aws_ sagemaker delete-endpoint-config --endpoint-config-name "$cfg" 2>/dev/null \
    && echo "  config supprimée : $cfg" || true
done

say "3/4 · instance notebook"
aws_ sagemaker stop-notebook-instance --notebook-instance-name "$NOTEBOOK" 2>/dev/null \
  && echo "  notebook arrêté (le disque persiste, pas de facture compute)" \
  || echo "  notebook déjà arrêté ou absent"

say "4/4 · vérification"
aws_ sagemaker list-endpoints --query 'Endpoints[].EndpointName' --output text || true
aws_ sagemaker list-notebook-instances --query 'NotebookInstances[].[NotebookInstanceName,NotebookInstanceStatus]' --output table || true

cat <<'EOF'

Conservé volontairement :
  - le bucket S3 (raw/, curated/, models/, forecast/, reports/) : les preuves
  - le Model Registry (3 versions) et la base Athena
  - les objets Model des Batch Transform terminés (métadonnées, gratuites)

À faire aussi côté console Academy : bouton « End Lab » pour arrêter la session.
EOF
