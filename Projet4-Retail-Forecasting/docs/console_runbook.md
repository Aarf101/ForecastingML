# Runbook console — tout exécuter depuis l'interface AWS

Déclencher et surveiller toute la chaîne **en cliquant dans la console**.

Deux endroits demandent un terminal, mais c'est **celui de l'instance SageMaker**
(donc à l'intérieur d'AWS) : générer le fichier d'entrée du Batch Transform, et
lancer Streamlit.

Région : **N. Virginia (us-east-1)**. Compte : `654654161593`.

---

## 0. Vérifier ce qui existe déjà

| Ressource | Chemin console | Attendu |
|---|---|---|
| Bucket | S3 → `retail-forecast-654654161593-ayoub` | `raw/ curated/ models/ forecast/ code/ replay/ reports/` |
| Stream | Kinesis → `retail-sales-stream` | `Active`, 1 shard |
| Livraison | Firehose → `retail-sales-to-s3` | `Active` |
| Lambda producteur | Lambda → `retail-replay-to-kinesis` | `Active` |
| Lambda gate | Lambda → `retail-register-model` | `Active` |
| Job ETL | Glue → ETL jobs → `retail-etl-raw-to-curated` | existe |
| Machine | Step Functions → `retail-forecast-pipeline` | `Active` |
| Règle | EventBridge → Rules → `retail-forecast-daily` | `Enabled` |

Si une ligne manque : section 9 « Créer à la main ».

---

## 1. Ingestion — bouton **Test** (Lambda)

1. **Lambda** → *Functions* → `retail-replay-to-kinesis`
2. Onglet **Test** → *Create new event*, nom `replay`, JSON :
   ```json
   {"max_seconds": 300}
   ```
3. **Save** → **Test**

Attendu (onglet *Test* → *Execution results*) :

```json
{"sent": 20000, "chunks_completed": 1, "next_chunk": 1, "total_chunks": 1, "finished": true}
```

Puis **attendre 60–90 s** (buffer Firehose) et vérifier :

**S3 → `raw/sales/2026/11/11/…/retail-sales-to-s3-1-….gz`** (≈ 115 Ko)

> Pour rejouer : supprimer `replay/_state/checkpoint.json` dans S3 (coche → **Delete**).

Erreurs éventuelles : `s3://…/raw/_errors/…`.

---

## 2. Traitement (ETL) — bouton **Run** (Glue)

1. **AWS Glue** → *ETL jobs* → cocher `retail-etl-raw-to-curated`
2. Bouton **Run**
3. Onglet **Runs** : `Running` → `Succeeded` (4–6 min)
4. Cliquer la ligne du run → **Output logs** → `/aws-glue/jobs/output`

Attendu :

```
[glue] lots Kafka lus : 20,000 lignes (s3://…/raw/sales/)
[glue] historique lu : 876,000 lignes (s3://…/raw/historique/)
[glue] lignes brutes (union) : 896,000
[glue] lignes propres : 876,000 (20,000 doublons supprimés)
[glue] écrit s3://…/curated/sales/
```

Vérification : S3 → `curated/sales/` contient des `part-*.snappy.parquet`.

---

## 3. Entraînement + gate — bouton **Start execution** (Step Functions)

1. **Step Functions** → *State machines* → `retail-forecast-pipeline`
2. **Start execution**, name `test1`, input :
   ```json
   {"run_id": "test1"}
   ```
3. **Start**

Graphe : `RunETL → TrainModel → GateWAPE → ModeleApprouve → Publie` (~10 min).
Cliquer sur `GateWAPE` pour voir la sortie :

```json
{"approved": true, "status": "Approved", "wape": 0.43, "baseline_wape": 0.57}
```

Vérifications :

| Quoi | Où |
|---|---|
| Job | SageMaker → *Training jobs* → `retail-lgbm-test1` = `Completed` |
| Métriques | `models/retail-lgbm-test1/output/model.tar.gz` → `metrics.json` |
| Version | *Model registry* → `retail-forecast-lgbm` |

---

## 4. Batch Transform — assistant console

### 4.1 Générer le fichier d'entrée (une fois, terminal de l'instance)

**SageMaker → Notebook instances → `retail-forecast-nb` → Open Jupyter → New → Terminal**

```bash
cd ~/retail
export S3_BUCKET=retail-forecast-654654161593-ayoub AWS_REGION=us-east-1
python -m infra.sagemaker.batch_transform --prepare-only
```

Attendu : `s3://…/inference/input/features.jsonl` (4 800 lignes).

### 4.2 Créer le modèle

**SageMaker** → *Inference* → **Models** → **Create model**

| Champ | Valeur |
|---|---|
| Model name | `retail-lgbm-console` |
| Container | **Use an existing container** |
| Image | `683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.4-2-py312-cpu-py3` |
| Model artifacts | `s3://retail-forecast-654654161593-ayoub/models/retail-lgbm-test1/output/model.tar.gz` |
| Env `SAGEMAKER_PROGRAM` | `inference.py` |
| Env `SAGEMAKER_SUBMIT_DIRECTORY` | `s3://retail-forecast-654654161593-ayoub/code/sourcedir.tar.gz` |
| Env `SAGEMAKER_REQUIREMENTS` | `/opt/ml/model/requirements.txt` |
| IAM role | *Use an existing role* → `LabRole` |

Puis **Create model**.

### 4.3 Créer le job de transformation

**SageMaker** → *Inference* → **Batch transform jobs** → **Create**

| Champ | Valeur |
|---|---|
| Job name | `retail-transform-console` |
| Model | `retail-lgbm-console` |
| Instance | `ml.m5.large`, count `1` |
| Input data | `s3://retail-forecast-654654161593-ayoub/inference/input/` |
| Content type | `application/json` |
| **Batch strategy** | **SingleRecord** ← obligatoire (JSONL) |
| Output path | `s3://retail-forecast-654654161593-ayoub/forecast/console/` |

→ `Completed` en 5–7 min. Résultat : `forecast/console/input.jsonl.out`.

---

## 5. Athena — onglet de requêtes

1. **Athena** → *Query editor*
2. Une fois : **Edit settings** → `s3://retail-forecast-654654161593-ayoub/athena-results/`
3. Base : `retail_forecast`

```sql
SELECT * FROM retail_forecast.forecast LIMIT 10;

SELECT item_id, SUM(units) AS unites
FROM retail_forecast.curated_sales
GROUP BY item_id ORDER BY unites DESC LIMIT 10;
```

---

## 6. Endpoint LLM (Hugging Face)

1. **SageMaker** → *Inference* → **Models** → **Create model**
   * Name `retail-llm-hf`
   * Image `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.6-transformers4.51-cpu-py312-ubuntu22.04`
   * Artifacts : **vide** (le conteneur télécharge les poids)
   * Env : `HF_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct`, `HF_TASK=text-generation`, `SM_NUM_GPUS=0`
   * Rôle `LabRole`
2. *Endpoint configurations* → **Create** : nom `retail-llm-hf-config`, type **Serverless**, modèle `retail-llm-hf`, mémoire `6144 MB`, concurrency `3`
3. *Endpoints* → **Create** : nom `retail-llm-hf`, config `retail-llm-hf-config` → attendre `InService` (5–10 min)

---

## 7. Dashboard — notebook

**SageMaker** → *Notebook instances* → `retail-forecast-nb` → **Open Jupyter** → **New → Terminal**

```bash
cd ~/retail
export S3_BUCKET=retail-forecast-654654161593-ayoub AWS_REGION=us-east-1
pip install -q streamlit plotly
nohup streamlit run dashboards/streamlit/app.py --server.port 8501 --server.headless true &
```

Ouvrir : `https://retail-forecast-nb.notebook.us-east-1.sagemaker.aws/proxy/8501/`

---

## 8. Démonter (fin de session)

| Ressource | Clic |
|---|---|
| Endpoint LLM | *Inference* → **Endpoints** → `retail-llm-hf` → **Delete** |
| Config endpoint | *Endpoint configurations* → `retail-llm-hf-config` → **Delete** |
| Modèle LLM | *Models* → `retail-llm-hf` → **Delete** |
| Notebook | *Notebook instances* → `retail-forecast-nb` → **Stop** |
| Stream | Kinesis → `retail-sales-stream` → **Delete** |
| Firehose | Firehose → `retail-sales-to-s3` → **Delete** |
| Machine | Step Functions → `retail-forecast-pipeline` → **Delete** |
| Règle | EventBridge → `retail-forecast-daily` → **Delete** |

À conserver : **S3**, **Model Registry**, catalogue **Athena** (preuves du projet).

---

## 9. Créer à la main (si le lab a été réinitialisé)

| Ressource | Étapes console | Valeurs clés |
|---|---|---|
| Stream Kinesis | Kinesis → **Create data stream** | `retail-sales-stream`, *Provisioned*, 1 shard |
| Firehose | Firehose → **Create delivery stream** | source *Kinesis stream* → `retail-sales-stream` ; destination *Amazon S3* → bucket ; prefix `raw/sales/` ; buffer 5 Mo / 60 s ; **GZIP** ; rôle `LabRole` |
| Lambda producteur | Lambda → **Create function** → *Author from scratch* | `Python 3.12`, rôle `LabRole`, zip de `infra/lambda/replay_to_kinesis.py`, handler `replay_to_kinesis.lambda_handler`, timeout 15 min, env `BUCKET`, `STREAM_NAME`, `REPLAY_PREFIX=replay/sales`, `CHECKPOINT_KEY=replay/_state/checkpoint.json` |
| Lambda gate | idem | zip de `infra/lambda/register_model.py`, handler `register_model.lambda_handler`, env `WAPE_THRESHOLD=0.45` |
| Job Glue | Glue Studio → *Script editor* → Spark | coller `infra/glue/etl_spark.py` ; rôle `LabRole` ; 2 × G.1X ; retries 0 ; timeout 15 ; paramètre `--BUCKET` |
| Step Functions | Step Functions → **Create state machine** | coller le JSON de `make aws-pipeline-asl` ; rôle `LabRole` |
| Règle EventBridge | EventBridge → **Create rule** | `rate(1 day)` ; cible *Step Functions: StartExecution* ; entrée `{"run_id":"scheduled"}` ; rôle `LabRole` |

