# Point 4 du sujet — Modélisation ML sur SageMaker

> « Entraînement, tuning et déploiement du modèle. Mise en place de SageMaker
> Pipelines. »

Ce document dit où en est chaque partie, et comment la montrer en démo.

## Où en est chaque brique

| Attendu | Implémenté | Fichier | Validé dans AWS |
|---|---|---|---|
Entraînement | ✅ Training Job `ml.m5.large`, conteneur SKLearn managé | `infra/sagemaker/launch.py` | ✅ `retail-lgbm-test2`, WAPE 0,4298 |
**Tuning** | ✅ HyperParameter Tuning Job bayésien, **objectif = WAPE de validation** | `infra/sagemaker/tuning.py` | 🟡 code + tests (à lancer au prochain lab) |
**Déploiement** | ✅ endpoint **serverless** sur le modèle approuvé (+ Batch Transform pour le lot) | `infra/sagemaker/deploy.py`, `batch_transform.py` | 🟡 code + tests |
Registre | ✅ gate WAPE partagé avec le pipeline, métadonnées sur la version | `infra/sagemaker/register.py` | ✅ v1→v4 |
**SageMaker Pipelines** | ✅ Pipeline `retail-forecast-mlops` | `infra/sagemaker/pipeline.py` | 🟡 code + tests |
Orchestration infra | ✅ Step Functions (`retail-forecast-pipeline`) | `infra/aws/orchestrate.py` | ✅ exécution `test2` |

Pourquoi deux orchestrateurs, et pas un seul : Step Functions n'a pas d'intégration
`.sync` pour `StartPipelineExecution` (contrairement à `CreateTrainingJob`), il
faudrait une Lambda de polling. La séparation actuelle — Step Functions pour la
chaîne infra (ingestion, ETL, déclenchements) et SageMaker Pipelines pour la
partie ML — est donc le découpage le plus simple à maintenir ici. C'est une dette
assumée : la partie ML existe en double (SFN + pipeline).

Deux orchestrateurs coexistent volontairement :

* **Step Functions** (`infra/aws/orchestrate.py`) : enchaîne ingestion → ETL → training →
  gate → publication. Il pilote aussi des services non-SageMaker.
* **SageMaker Pipelines** (`pipeline.py`) : ne couvre que la partie ML, avec ses
  steps natifs et sa condition `GateWAPE`. C'est ce que le sujet appelle
  « mise en place de SageMaker Pipelines ».

## Le pipeline

```
PreprocessFeatures Processing Job : features calculées UNE fois (lags, rolling)
      │  s3://…/features/<execution-id>/features.parquet   ← une seule définition
      ▼
TrainLightGBM      Training Job (même image, même code que launch.py) · cache 14 j
      │  s3://…/models/…/output/model.tar.gz
      ▼
EvaluateModel      Processing Job : RECALCULE le WAPE depuis les features + l'artefact
      │  evaluation.json -> regression_metrics.test_wape.value
      ▼
GateWAPE           ConditionStep  (seuil = paramètre WapeThreshold)
      ├── WAPE ≤ seuil → RegisterApproved  (Model Registry, statut Approved)
      └── WAPE > seuil → WAPETropEleve     (FailStep : exécution rouge, rien de publié)
```

### Pourquoi une étape de features

Sans `PreprocessFeatures`, les features sont recalculées par **chaque**
consommateur : conteneur d'entraînement, client de Batch Transform, échantillon
envoyé à l'endpoint. Si l'un d'eux change un paramètre (`LAGS`, `ROLLS`),
l'entraînement et l'inférence ne parlent plus le même espace de features — c'est
le *train/serve skew*, le défaut le plus courant et le plus silencieux en MLOps.
Ici la définition vit à un seul endroit, l'étape écrit `features/<execution-id>/`
dans S3 (un dossier par exécution : deux runs concurrents ne s'écrasent pas), et
l'entraînement **comme** l'évaluation relisent cette table (chaînage par
propriété : chaque exécution consomme les features qu'elle vient de produire).

`train_entry.py` détecte les deux cas : colonnes `lag_*` présentes → il ne
recalcule rien ; absentes (Training Job direct via `launch.py`, qui lit encore
`curated/`) → il calcule sur place. Les deux chemins restent donc valides.

L'évaluation **recalcule** le WAPE : elle recharge les boosters `model_<h>.txt`
et prédit sur les 28 derniers jours. Elle publie aussi l'écart avec les métriques
de l'artefact — ce qui prouve que le chiffre n'est pas recopié. Si le recalcul est
impossible (dépendance manquante), elle republie l'artefact en le disant
(`"independent": false`, `"source": "artifact"`), jamais en silence.

Le gate porte sur le **WAPE moyen des 4 horizons**, comme `register.py` : un
modèle bon à J+1 et mauvais à J+28 ne passe pas.

## Commandes

```bash
# 1. tuning : 6 essais, ~18 min, ~0,03 €
make sagemaker-tune                     # = python -m infra.sagemaker.tuning --wait

# 2. pipeline : création (upsert) puis exécution
make sagemaker-pipeline-check           # pré-requis lab, 0 €
make sagemaker-pipeline                 # --upsert
make sagemaker-pipeline-run             # --run --wait
make sagemaker-pipeline-status          # 5 dernières exécutions

# 3. déploiement : endpoint serverless (~0 € au repos) + lot
make sagemaker-deploy                   # crée le modèle + l'endpoint (1-5 min)
make sagemaker-deploy-invoke            # 4 prévisions J+1..J+28 (preuve « déploiement »)
make sagemaker-deploy-down              # supprime tout -> facturation 0
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
Code uploadé | construire la définition **uploade** `sourcedir.tar.gz`, `preprocess_entry.py` et `evaluate_entry.py` : il faut donc une session valide, même pour `--dry-run` |
Permission à vérifier | `sagemaker:CreatePipeline` — si `AccessDenied`, replier sur Step Functions (déjà en place) |

### Où l'exécuter (et le piège du profil)

| Contexte | Identifiants | Remarque |
|---|---|---|
Poste | profil `lab` (`~/.aws/credentials`) | `.env` contient `AWS_PROFILE=lab` |
**Notebook SageMaker** | rôle IAM de l'instance | **aucun profil `lab`** sur l'instance : `pipeline.py` détecte l'absence et retombe sur la chaîne boto3 par défaut (message `profil 'lab' absent -> chaîne par défaut`) |

Sans ce repli, `make sagemaker-pipeline` lancé depuis le notebook échouerait en
`ProfileNotFound` alors que tous les autres modules fonctionnent — c'est corrigé.

### Vérification avant dépense (`--check`, 0 €)

```bash
make sagemaker-pipeline-check      # = python -m infra.sagemaker.pipeline --check
```

Contrôle en quelques secondes : identité STS, écriture dans le bucket, lecture des
pipelines, existence du groupe `retail-forecast-lgbm`, lisibilité de `LabRole`.
Sortie attendue :

```
[check] compte 654654161593 · arn:aws:sts::654654161593:assumed-role/voclabs/…
[check] bucket retail-forecast-654654161593-ayoub : écriture OK
[check] sagemaker:ListPipelines : lecture OK
[check] sagemaker:ListModelPackages : lecture OK
[check] rôle LabRole : lisible
[check] prêt pour --upsert
```

La lecture ne prouve pas `CreatePipeline` : l'`--upsert` est le seul vrai test, et
il affiche un repli explicite vers Step Functions s'il est refusé.

## Coût

| Action | Durée | Coût |
|---|---|---|
Tuning 6 essais × 1 horizon | ~18 min | ~0,03 € |
Pipeline complet (`PreprocessFeatures` + training 4 horizons + processing) | ~10-12 min | ~0,025 € |
Endpoint serverless au repos | — | ~0 € (facturé à l'inférence, ms) |
`deploy invoke` (4 prévisions) | ~2 s | < 0,001 € |
Batch Transform | ~6 min | ~0,01 € |

Garde-fous : `MaxRuntimeInSeconds=1200` par job, `MaxParallelTrainingJobs=1`,
`TrainingJobEarlyStoppingType=Auto` (les essais sans progrès s'arrêtent),
cache 14 j sur l'entraînement, `make sagemaker-deploy-down` pour repasser à 0.

## Choix techniques à savoir justifier

| Choix | Raison |
|---|---|
`train_entry.py` émet `[hpo] valid_wape=…` et `[hpo] test_wape=…` | le tuning lit la métrique dans CloudWatch via regex (une seule occurrence par événement) ; l'objectif est la validation, le test reste le juge |
Options argparse en double (`--learning-rate` / `--learning_rate`) | le tuning transmet le nom du paramètre tel quel ; le tiret ne doit pas faire échouer l'essai |
`ScriptProcessor(image_uri=IMAGE)` | évite la résolution d'image « processing » par version : on réutilise l'image déjà validée dans ce compte |
`EvaluateModel` recalcule, avec repli | le conteneur SKLearn n'a pas LightGBM : le script l'installe (`pip`) puis recalcule ; en cas d'échec il republie l'artefact en le signalant (`independent: false`) |
Métadonnées statiques sur la version du registre | le SDK v2 n'expose pas `ModelQuality` dans `ModelMetrics` ; on écrit les métadonnées client et on pointe le rapport d'évaluation |
Endpoint **serverless** | `ml.m5.large` coûte ~0,11 €/h même sans trafic ; le serverless facture à la milliseconde et retombe à zéro |
Batch Transform hors pipeline | il dépend de données d'inférence préparées ; le garder dehors évite de payer une instance à chaque exécution |
Cache 14 j sur `TrainLightGBM` | une deuxième démonstration ne repaie pas l'entraînement si rien n'a changé |
`PreprocessFeatures` en step dédié, chaîné par propriété | une seule définition des features (pas de *train/serve skew*) et chaque exécution consomme la table qu'elle vient de produire, sans risque de lire celle d'une exécution précédente |

## Ce qui n'est pas fait (et pourquoi)

| Absent | Raison |
|---|---|
Model Monitor | facturation horaire, hors périmètre du sujet |
TransformStep dans le pipeline | nécessite un `Model` construit depuis le package registré ; `batch_transform.py` couvre déjà le besoin |
CloudFormation du pipeline | `cloudformation:*` refusé dans le lab |
Step Functions → `StartPipelineExecution` | pas d'intégration `.sync` pour les pipelines : il faudrait une Lambda de polling |
Model Metrics chiffré dans le registre via le pipeline | `ModelMetrics` v2 n'a pas de champ quality ; `register.py` (Lambda) le fait, lui, à l'exécution |
