#!/usr/bin/env bash
# Arrête TOUT ce qui facture dans le compte Learner Lab.
#
# Ce qui coûte de l'argent :
#   1. l'instance EC2 (Kafka)            ~0,042 $/h tant qu'elle tourne
#   2. la règle EventBridge              ~0,16 $ À CHAQUE déclenchement (rate(1 day))
#      -> la plus dangereuse : elle persiste entre les sessions et relance seule
#         la chaîne ETL + entraînement
#   3. Firehose                          ~0,03 $/Go ingéré (0 au repos)
#   4. endpoint LLM / notebook           (serverless = 0 au repos ; notebook = 0,05 $/h)
#
# Ne touche PAS au lac de données (S3), au Model Registry, au catalogue Athena ni
# aux jobs Glue : ce sont les preuves du rendu, et ils ne facturent pas au repos.
#
# Usage :
#   bash scripts/aws_teardown.sh --dry-run   # montre ce qui serait supprimé
#   bash scripts/aws_teardown.sh --yes       # agit
set -euo pipefail

PROFILE="${AWS_PROFILE:-lab}"
REGION="${AWS_REGION:-us-east-1}"
ENDPOINT="${LLM_ENDPOINT:-retail-llm-hf}"
NOTEBOOK="${NOTEBOOK_NAME:-retail-forecast-nb}"
STREAM="${STREAM_NAME:-retail-sales-direct}"
RULE="${RULE_NAME:-retail-forecast-daily}"
KAFKA_TAG="${KAFKA_TAG:-retail-kafka}"
SG="${KAFKA_SG:-retail-kafka-sg}"
MODE="${1:---dry-run}"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
aws_() { AWS_PROFILE="$PROFILE" aws "$@" --region "$REGION"; }

say "1/6 · état actuel"
echo "-- EC2 --"
aws_ ec2 describe-instances \
  --filters "Name=tag:Name,Values=$KAFKA_TAG" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].[InstanceId,State.Name,InstanceType]' \
  --output table || true
echo "-- Endpoint LLM --"
aws_ sagemaker list-endpoints --query 'Endpoints[].[EndpointName,EndpointStatus]' --output table || true
echo "-- Notebook --"
aws_ sagemaker list-notebook-instances \
  --query 'NotebookInstances[].[NotebookInstanceName,NotebookInstanceStatus]' --output table || true
echo "-- Firehose --"
aws_ firehose list-delivery-streams --query 'DeliveryStreamNames' --output text || true
echo "-- Règle EventBridge --"
aws_ events list-rules --query "Rules[?Name=='$RULE'].[Name,State,ScheduleExpression]" --output table || true

if [ "$MODE" != "--yes" ]; then
  say "mode --dry-run : rien n'a été supprimé"
  echo "Relancer avec : bash scripts/aws_teardown.sh --yes"
  exit 0
fi

say "2/6 · règle EventBridge $RULE (le poste le plus coûteux)"
TARGETS=$(aws_ events list-targets-by-rule --rule "$RULE" \
  --query 'Targets[].Id' --output text 2>/dev/null || true)
if [ -n "$TARGETS" ]; then
  # shellcheck disable=SC2086
  aws_ events remove-targets --rule "$RULE" --ids $TARGETS >/dev/null \
    && echo "  cibles retirées : $TARGETS" || true
fi
aws_ events delete-rule --name "$RULE" 2>/dev/null \
  && echo "  règle supprimée (plus aucun déclenchement planifié)" \
  || echo "  règle déjà absente"

say "3/6 · instance EC2 $KAFKA_TAG"
IDS=$(aws_ ec2 describe-instances \
  --filters "Name=tag:Name,Values=$KAFKA_TAG" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text 2>/dev/null || true)
if [ -n "$IDS" ]; then
  # shellcheck disable=SC2086
  aws_ ec2 terminate-instances --instance-ids $IDS >/dev/null \
    && echo "  instance(s) terminée(s) : $IDS"
else
  echo "  aucune instance à terminer"
fi
aws_ ec2 delete-security-group --group-name "$SG" 2>/dev/null \
  && echo "  groupe de sécurité $SG supprimé" || echo "  SG $SG déjà absent"

say "4/6 · delivery stream Firehose $STREAM"
aws_ firehose delete-delivery-stream --delivery-stream-name "$STREAM" >/dev/null 2>&1 \
  && echo "  stream supprimé (recréable avec : make aws-firehose-up)" \
  || echo "  stream déjà absent ou en cours de suppression"

say "5/6 · endpoints (LLM + modèle de prévision) et notebook"
# Deux endpoints distincts : l'agent LLM et le serving LightGBM.
for ep in "$ENDPOINT" "${DEMAND_ENDPOINT:-retail-demand-lgbm}"; do
  aws_ sagemaker delete-endpoint --endpoint-name "$ep" 2>/dev/null \
    && echo "  endpoint supprimé : $ep" || echo "  endpoint absent : $ep"
  for suffix in -config -config-v2; do
    aws_ sagemaker delete-endpoint-config \
      --endpoint-config-name "${ep}${suffix}" 2>/dev/null || true
  done
done
# Ressources Model créées pour le serving / les Batch Transform.
for m in retail-demand-lgbm-model; do
  aws_ sagemaker delete-model --model-name "$m" 2>/dev/null \
    && echo "  modèle supprimé : $m" || true
done
aws_ sagemaker stop-notebook-instance --notebook-instance-name "$NOTEBOOK" 2>/dev/null \
  && echo "  notebook arrêté (le disque persiste, pas de facture compute)" \
  || echo "  notebook déjà arrêté ou absent"

say "6/6 · vérification"
echo "-- EC2 --"
aws_ ec2 describe-instances \
  --filters "Name=instance-state-name,Values=running,pending,stopped" \
  --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output text || true
echo "-- Endpoints --"
aws_ sagemaker list-endpoints --query 'Endpoints[].EndpointName' --output text || true
echo "-- Règles de notre projet --"
aws_ events list-rules --query "Rules[?Name=='$RULE'].Name" --output text || true

cat <<'EOF'

Conservé volontairement (ne facture pas au repos) :
  - le bucket S3 (raw/, curated/, models/, forecast/, features/, reports/)
  - le Model Registry et la base Athena
  - les jobs Glue et la machine Step Functions (entités, pas de calcul)

Reste à faire : bouton « End Lab » côté AWS Academy pour clore la session.
EOF
