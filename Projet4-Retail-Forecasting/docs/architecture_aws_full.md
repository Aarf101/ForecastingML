# Architecture complète dans AWS — état et mise en route

Objectif : **aucun composant d'exécution hors du compte AWS**. Ce document liste
chaque brique du diagramme, le service AWS qui l'implémente, son nom de ressource,
et la commande qui la crée.

## 1. La chaîne complète

```
                 ┌─────────────── EventBridge cron(1 jour) ───────────────┐
                 ▼                                                        │
        Step Functions retail-forecast-pipeline                          │
                 │                                                        │
   1. RunETL ────┤ Glue retail-etl-raw-to-curated                         │
                 │      lit s3://…/raw/sales/*.gz + raw/historique/       │
                 │      écrit s3://…/curated/sales_glue/                  │
   2. TrainModel ┤ SageMaker Training Job  (ml.m5.large)                  │
                 │      lit curated/  → écrit models/<job>/output/        │
   3. GateWAPE ──┤ Lambda retail-register-model                           │
                 │      lit metrics.json → Approved / PendingManualApproval
   4. Choice ────┘ WAPE ≤ seuil ? → Succeed : Fail                        │
                                                                          │
   Ingestion (amont) :                                                    │
   Lambda retail-replay-to-kinesis → Kinesis retail-sales-stream          │
        → Firehose retail-sales-to-s3 → s3://…/raw/sales/  ───────────────┘

   Aval (service) :
   Model Registry  →  Batch Transform  →  s3://…/forecast/table/
   Athena (curated_sales, forecast)   ·   Streamlit sur instance SageMaker
   Endpoint serverless retail-llm-hf (LLM)
```

## 2. Brique par brique

| # | Brique du diagramme | Service AWS | Nom de la ressource | Déploiement |
|---|---|---|---|---|
| 1 | Producteur d'événements | **Lambda** | `retail-replay-to-kinesis` | `make aws-ingest-up` |
| 2 | Transport d'événements | **Kinesis Data Streams** | `retail-sales-stream` (1 shard) | `make aws-ingest-up` |
| 3 | Livraison vers le lac | **Firehose** | `retail-sales-to-s3` | `make aws-ingest-up` |
| 4 | Lac brut | **S3** | `raw/`, `curated/`, `models/`, `forecast/`, `reports/` | `make aws-setup` |
| 5 | ETL / nettoyage | **Glue** (PySpark) | `retail-etl-raw-to-curated` | console ou `make cfn-*` |
| 6 | Features | **SageMaker** (dans le conteneur d'entraînement) | `train_entry.py` | inclus dans le code |
| 7 | Entraînement | **SageMaker Training** | `retail-lgbm-<run_id>` | état `TrainModel` |
| 8 | Évaluation + gate | **Lambda** | `retail-register-model` | `make aws-pipeline-up` |
| 9 | Registre de modèles | **SageMaker Model Registry** | `retail-forecast-lgbm` | état `GateWAPE` |
| 10 | Inférence par lots | **SageMaker Batch Transform** | `retail-transform-*` | `make sagemaker-transform` |
| 11 | Catalogue / KPI | **Athena** + Glue Data Catalog | `retail_forecast.*` | `make athena` |
| 12 | LLM | **SageMaker Endpoint** (Hugging Face, serverless) | `retail-llm-hf` | `python -m infra.sagemaker.llm_endpoint` |
| 13 | Dashboard | **Instance SageMaker** (Streamlit :8501) | `retail-forecast-nb` | console |
| 14 | Orchestration | **Step Functions** | `retail-forecast-pipeline` | `make aws-pipeline-up` |
| 15 | Planification | **EventBridge** | `retail-forecast-daily` | `make aws-pipeline-up` |
| 16 | Journalisation | **CloudWatch Logs** (automatique) | `/aws/sagemaker/*`, `/aws/lambda/*` | implicite |

## 3. Mise en route, dans l'ordre

```bash
# .env : S3_ENDPOINT_URL=  AWS_ACCESS_KEY_ID=  AWS_SECRET_ACCESS_KEY=  AWS_PROFILE=lab

make aws-setup            # bucket + dataset historique
make aws-ingest-prep      # parquet -> morceaux JSONL gzip dans replay/
make aws-ingest-up        # Kinesis + Firehose + Lambda producteur
make aws-ingest-run       # rejeu des événements dans Kinesis
make aws-pipeline-up      # state machine + règle EventBridge + Lambda de gate
make aws-pipeline-run     # exécution Glue -> Training -> Gate -> Registry
make sagemaker-transform  # Batch Transform -> forecast/
make athena               # tables externes + KPI
```

`make aws-pipeline-status` affiche l'état de la machine, de la règle et des
dernières exécutions. `make aws-pipeline-asl` imprime le JSON ASL à coller dans la
console Step Functions si l'on préfère le créer à la main.

## 4. Ce qui n'est pas dans AWS (et pourquoi)

| Élément | Nature | Justification |
|---|---|---|
| `data/simulate.py` | **création** du jeu de données synthétique | production de données de test, faite une fois ; le résultat est dans S3 |
| `scripts/prepare_replay_chunks.py` | découpage ponctuel parquet → JSONL | préparation de données, faite une fois ; sortie dans S3 |
| `infra/sagemaker/launch.py` (upload du code) | empaquetage et dépôt de `sourcedir.tar.gz` | acte de **déploiement**, pas d'exécution ; l'artefact vit dans S3 |
| `make …` | déclencheur manuel | remplacé par EventBridge pour l'exécution planifiée |

Aucune de ces quatre lignes n'est un composant du runtime : une fois l'artefact en
place, **toute l'exécution se produit dans le compte**.

## 5. Démontage

```bash
make aws-ingest-down      # Firehose + Lambda producteur + stream Kinesis
make aws-pipeline-down    # règle EventBridge + state machine + Lambda de gate
make aws-teardown-go      # endpoint LLM + notebook
```

Le bucket S3, le registre et le catalogue Athena sont conservés : ce sont les
preuves du projet.

## 6. Coût d'une session complète

| Poste | Durée / volume | Coût |
|---|---|---|
| Kinesis, 1 shard | 1 h | 0,015 $ |
| Firehose | ~2 Mo livrés | < 0,01 $ |
| Lambda (2 fonctions) | free tier | 0 $ |
| Glue | 10 min facturées | 0,15 $ |
| Training Job | ~5 min | 0,01 $ |
| Batch Transform | ~6 min | 0,01 $ |
| Athena | quelques Mo scannés | < 0,01 $ |
| Endpoint LLM (serverless) | à la requête | < 0,01 $ |
| Instance SageMaker (dashboard) | 1 h | 0,05 $ |
| **Total** | | **≈ 0,24 $** |
