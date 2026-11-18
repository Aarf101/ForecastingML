# Point 4 du sujet — Modélisation ML sur SageMaker

> « Entraînement, tuning et déploiement du modèle. Mise en place de SageMaker
> Pipelines. »

Ce document dit où en est chaque partie, et comment la montrer en démo.

## Où en est chaque brique

| Attendu | Implémenté | Fichier | Validé dans AWS |
|---|---|---|---|
Entraînement | ✅ Training Job `ml.m5.large`, conteneur SKLearn managé | `infra/sagemaker/launch.py` | ✅ `retail-lgbm-test2`, WAPE 0,4298 |
**Tuning** | ✅ HyperParameter Tuning Job bayésien | `infra/sagemaker/tuning.py` | 🟡 code + tests (à lancer au prochain lab) |
Déploiement | ✅ Batch Transform + Model Registry (gate WAPE) | `infra/sagemaker/batch_transform.py`, `register.py` | ✅ `forecast/table/forecast.parquet` |
**SageMaker Pipelines** | ✅ Pipeline `retail-forecast-mlops` | `infra/sagemaker/pipeline.py` | 🟡 code + tests |
Orchestration alternative | ✅ Step Functions (`retail-forecast-pipeline`) | `infra/aws/orchestrate.py` | ✅ exécution `test2` |

Deux orchestrateurs coexistent volontairement :

* **Step Functions** (`infra/aws/orchestrate.py`) : enchaîne ingestion → ETL → training →
  gate → publication. Il pilote aussi des services non-SageMaker.
* **SageMaker Pipelines** (`pipeline.py`) : ne couvre que la partie ML, avec ses
  steps natifs et sa condition `GateWAPE`. C'est ce que le sujet appelle
  « mise en place de SageMaker Pipelines ».

## Le pipeline

```
TrainLightGBM      Training Job (même image, même code que launch.py)
      │  s3://…/models/…/output/model.tar.gz
      ▼
EvaluateModel      Processing Job (stdlib, lit metrics.json de l'artefact)
      │  evaluation.json -> regression_metrics.test_wape.value
      ▼
GateWAPE           ConditionStep  (seuil = paramètre WapeThreshold)
      ├── WAPE ≤ seuil → RegisterApproved  (Model Registry, statut Approved)
      └── WAPE > seuil → WAPETropEleve     (FailStep : exécution rouge, rien de publié)
```

L'évaluation moyenne le WAPE **sur les 4 horizons** : un modèle excellent à J+1
et mauvais à J+28 ne doit pas passer le gate sur la foi du seul horizon 1.

## Commandes

```bash
# 1. tuning : 6 essais, ~18 min, ~0,03 €
make sagemaker-tune                     # = python -m infra.sagemaker.tuning --wait

# 2. pipeline : création (upsert) puis exécution
make sagemaker-pipeline                 # --upsert
make sagemaker-pipeline-run             # --run --wait
make sagemaker-pipeline-status          # 5 dernières exécutions

# 3. déploiement (hors pipeline, comme documenté)
make sagemaker-transform                # Batch Transform -> forecast/table/
make athena                             # KPI SQL sur les prévisions
```

Équivalents directs, sans `make` (Windows / `run.ps1`) :

```bash
python -m infra.sagemaker.tuning --max-jobs 6 --wait
python -m infra.sagemaker.pipeline --upsert
python -m infra.sagemaker.pipeline --run --wait
```

## Pré-requis compte AWS Academy

| Point | Détail |
|---|---|
Rôle | `LabRole` imposé (`SAGEMAKER_ROLE_ARN`) : `iam:CreateRole` est refusé dans le lab |
Dépendance | `pip install 'sagemaker>=2.214,<3'` — la v3 a déplacé l'API des pipelines |
Bucket par défaut | forcé sur `S3_BUCKET` (`PipelineSession(default_bucket=…)`) : sinon le SDK tenterait de créer `sagemaker-<région>-<compte>` |
Code uploadé | construire la définition **uploade** `sourcedir.tar.gz` et `evaluate_entry.py` : il faut donc une session valide, même pour `--dry-run` |
Permission à vérifier | `sagemaker:CreatePipeline` — si `AccessDenied`, replier sur Step Functions (déjà en place) |

## Coût

| Action | Durée | Coût |
|---|---|---|
Tuning 6 essais × 1 horizon | ~18 min | ~0,03 € |
Pipeline complet (training 4 horizons + processing) | ~8 min | ~0,02 € |
Batch Transform | ~6 min | ~0,01 € |

Garde-fous : `MaxRuntimeInSeconds=1200` par job, `MaxParallelTrainingJobs=1`,
`TrainingJobEarlyStoppingType=Auto` (les essais sans progrès s'arrêtent).

## Choix techniques à savoir justifier

| Choix | Raison |
|---|---|
`train_entry.py` émet `[hpo] test_wape=…` | le tuning lit la métrique dans CloudWatch via regex ; il faut **une seule** occurrence par événement de log |
Options argparse en double (`--learning-rate` / `--learning_rate`) | le tuning transmet le nom du paramètre tel quel ; le tiret ne doit pas faire échouer l'essai |
`ScriptProcessor(image_uri=IMAGE)` | évite la résolution d'image « processing » par version : on réutilise l'image déjà validée dans ce compte |
`EvaluateModel` sans LightGBM | l'étape lit `metrics.json` de l'artefact : aucune dépendance lourde, démarrage rapide |
Batch Transform hors pipeline | il dépend de données d'inférence préparées ; le garder dehors évite de payer une instance à chaque exécution |

## Ce qui n'est pas fait (et pourquoi)

| Absent | Raison |
|---|---|
Model Monitor | facturation horaire, hors périmètre du sujet |
TransformStep dans le pipeline | nécessite un `Model` construit depuis le package registré ; `batch_transform.py` couvre déjà le besoin |
CloudFormation du pipeline | `cloudformation:*` refusé dans le lab |
Endpoint temps réel | le sujet parle de « déploiement », Batch Transform suffit pour du J+1…J+28 |
