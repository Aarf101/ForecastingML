#!/usr/bin/env bash
# Wrapper CloudFormation pour la stack retail-forecast.
#
# Deux modes :
#   deploy  crée la stack de zéro (bucket neuf)
#   import  adopte les ressources déjà créées par les scripts (bucket existant)
#
# Usage :
#   bash scripts/cfn.sh deploy
#   bash scripts/cfn.sh import            # bucket + registre + base déjà présents
#   bash scripts/cfn.sh drift             # détecte les écarts avec le template
#   bash scripts/cfn.sh delete            # supprime la stack (bucket conservé)
set -euo pipefail

PROFILE="${AWS_PROFILE:-lab}"
STACK="${STACK_NAME:-retail-forecast-stack}"
TEMPLATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/infra/cloudformation/stack.yml"
REGION="${AWS_REGION:-us-east-1}"
MODE="${1:-}"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
cfn() { AWS_PROFILE="$PROFILE" aws cloudformation "$@" --region "$REGION"; }

[ -f "$TEMPLATE" ] || { echo "template introuvable : $TEMPLATE" >&2; exit 1; }

ACCOUNT=$(AWS_PROFILE="$PROFILE" aws sts get-caller-identity --query Account --output text)
BUCKET="${BUCKET_NAME:-retail-forecast-${ACCOUNT}-ayoub}"

say "1/2 · validation du template"
cfn validate-template --template-body "file://$TEMPLATE" --query 'Description' --output text

wait_stack() {
  local action="$1"
  for _ in $(seq 1 60); do
    local status
    status=$(cfn describe-stacks --stack-name "$STACK" \
      --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "PENDING")
    case "$status" in
      *COMPLETE|*FAILED|ROLLBACK*) echo "  statut : $status"; return 0 ;;
    esac
    sleep 5
  done
  echo "  [ATTENTION] délai dépassé ($action)" >&2
}

case "$MODE" in
  deploy)
    say "2/2 · création de la stack $STACK (bucket $BUCKET)"
    cfn create-stack \
      --stack-name "$STACK" \
      --template-body "file://$TEMPLATE" \
      --parameters \
          ParameterKey=BucketName,ParameterValue="$BUCKET" \
          ParameterKey=EnableVersioning,ParameterValue=true \
      --capabilities CAPABILITY_IAM \
      --tags Key=project,Value=retail-forecast
    wait_stack create
    ;;

  import)
    say "2/2 · adoption des ressources existantes dans $STACK"
    # Les deux tables Glue ont été créées par DDL : CloudFormation ne peut pas les
    # adopter (non importables) et refuserait de les recréer. On supprime donc
    # leurs métadonnées ; les fichiers Parquet dans S3 ne sont pas touchés.
    if [ "${CONFIRM_DROP_TABLES:-0}" = "1" ]; then
      PY=".venv/bin/python"
      [ -x "$PY" ] || PY=".venv/Scripts/python.exe"
      [ -x "$PY" ] || PY="python3"
      [ -x "$PY" ] || PY="python"
      ( cd "$(dirname "$TEMPLATE")/.." && "$PY" -m infra.athena.setup --drop )
    else
      echo "  [ATTENTION] relancer avec CONFIRM_DROP_TABLES=1 pour supprimer les
              tables Glue existantes (métadonnées) avant l'import." >&2
      exit 1
    fi
    # 1. créer un change set en mode IMPORT
    cfn create-change-set \
      --stack-name "$STACK" \
      --change-set-name "import-$(date +%s)" \
      --change-set-type IMPORT \
      --template-body "file://$TEMPLATE" \
      --parameters \
          ParameterKey=BucketName,ParameterValue="$BUCKET" \
          ParameterKey=EnableVersioning,ParameterValue=false \
      --resources-to-import \
          ResourceType=AWS::S3::Bucket,LogicalResourceId=DataLakeBucket,ResourceIdentifier="{BucketName=$BUCKET}" \
          ResourceType=AWS::SageMaker::ModelPackageGroup,LogicalResourceId=ModelRegistryGroup,ResourceIdentifier="{ModelPackageGroupName=retail-forecast-lgbm}" \
          ResourceType=AWS::Glue::Database,LogicalResourceId=CatalogDatabase,ResourceIdentifier="{CatalogId=$ACCOUNT,DatabaseName=retail_forecast}"
    CHANGESET=$(cfn list-change-sets --stack-name "$STACK" --query 'Summaries[0].ChangeSetName' --output text)
    cfn describe-change-set --stack-name "$STACK" --change-set-name "$CHANGESET" \
      --query 'Changes[].ResourceChange.[LogicalResourceId,Action]' --output table
    cfn execute-change-set --stack-name "$STACK" --change-set-name "$CHANGESET"
    wait_stack import
    ;;

  drift)
    say "détection de dérive sur $STACK"
    DET=$(cfn detect-stack-drift --stack-name "$STACK" --query StackDriftDetectionId --output text)
    for _ in $(seq 1 30); do
      STATUS=$(cfn describe-stack-drift-detection-status --stack-drift-detection-id "$DET" \
        --query StackDriftStatus --output text)
      [ "$STATUS" != "DETECTION_IN_PROGRESS" ] && break
      sleep 3
    done
    echo "  dérive : $STATUS"
    cfn describe-stack-resource-drifts --stack-name "$STACK" --stack-resource-drift-status-filters MODIFIED DELETED \
      --query 'StackResourceDrifts[].[LogicalResourceId,StackResourceDriftStatus]' --output table
    ;;

  delete)
    say "suppression de $STACK (le bucket est conservé : DeletionPolicy Retain)"
    cfn delete-stack --stack-name "$STACK"
    wait_stack delete
    ;;

  *)
    echo "usage : bash scripts/cfn.sh {deploy|import|drift|delete}" >&2
    exit 1
    ;;
esac

say "sorties"
cfn describe-stacks --stack-name "$STACK" --query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' --output table 2>/dev/null || true
