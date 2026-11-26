# AGENTS.md — guide de travail pour un agent IA

Projet **Retail ML Forecasting** (Projet 4) : prévision des ventes observées J+1…J+16 par
(magasin, article) sur AWS. Ce fichier est la carte du dépôt. **Le lire en premier
évite d'explorer le code** : ne relire un fichier que si la tâche le concerne.

## Commandes (Makefile)

| Commande | Effet |
|---|---|
| `make test` | pytest (169 tests, aucun service externe requis) |
| `make benchmark-median` | ablation locale L2/L1/médiane + calendrier, réplication indépendante, sans publication |
| `make aws-chronos-validate` | surveillance des six partitions, Batch Transform réel et nettoyage du candidat |
| `make benchmark-chronos` | comparaison exploratoire Chronos-2 / v7, 120 séries, aucun calcul AWS |
| `make train` | évaluation locale LightGBM → `reports/local_baseline.json` |
| `make train-quick` | même chose sur 20 % des séries, 200 arbres (itération) |
| `make up` / `make down` | Redpanda (:19092, console :8080) + s3mock (:9090) |
| `make produce` / `make consume` | Phase 3 : ventes → topic `sales.events`, puis topic → `raw/` S3 |
| `make e2e` | enchaîne tout et échoue si aucun objet n'arrive dans S3 |
| `make etl` | `raw/` -> `curated/` Parquet (miroir local du job Glue) |
| `make sagemaker-train` | Training Job LightGBM sur AWS (`infra/sagemaker/launch.py --wait`) |
| `make diagram` | régénère les 5 variantes de diagrammes (matplotlib uniquement) |
| `make slides` | régénère `docs/slides.pptx` depuis `docs/build_slides.py` |
| `make sagemaker-register` | Model Registry (gate `WAPE_THRESHOLD`) |
| `make sagemaker-tune` | HyperParameter Tuning Job (6 essais, objectif `valid_wape`) |
| `make sagemaker-pipeline` | SageMaker Pipeline `retail-forecast-mlops` (upsert) |
| `make sagemaker-pipeline-run` | exécute le pipeline et attend (train → eval → gate → registry) |
| `make sagemaker-deploy` | endpoint serverless du modèle approuvé (0 € au repos) |
| `make sagemaker-transform` | Batch Transform → `forecast/` |
| `make athena` | tables externes + 2 KPI SQL |
| `make aws-env-full` | environnement AWS complet (ingestion + Glue + Step Functions + Athena), 0 donnée envoyée |
| `make dashboard` | Streamlit :8501 |
| `make brief` | rapport hebdo Ollama → `reports/` + S3 |
| `make lettre` | régénère `docs/lettre_motivation.pdf` (pdflatex, nécessite texlive) |

Toujours utiliser `.venv/bin/python` (`python3` système n'a ni lightgbm ni kafka-python).

## Carte du code

```
src/retail_forecast/
  config.py            SETTINGS figé depuis .env — SEUL endroit qui lit os.environ
  data/simulate.py     générateur de ventes (876k lignes, 20 magasins × 60 articles × 2 ans)
  data/load.py         charge un dataset quelconque (CSV/Parquet/JSON, local ou S3) -> schéma canonique
  features/build.py    lags 1→35, rolling mean/std 7/14/28, calendrier, time_split()
  evaluate/metrics.py  WAPE, MAPE, RMSE, RMSLE, pinball loss
  evaluate/benchmark_chronos.py comparaison appariée, sans publication ni réentraînement
  evaluate/benchmark_median.py ablation L1/calendrier, validation puis test réservé
  features/target_calendar.py calendrier cible optionnel (jours fériés du simulateur)
  train/train_lgbm.py  miroir local du Training Job SageMaker + baseline saisonnière
  train/profiles.py    défaut median_calendar_v1 ; compatibilité des artefacts legacy_l2
  ingest/storage.py    unique porte boto3 : raw_key(), gzip JSONL, chunking
  ingest/producer.py   rejoue le parquet → Kafka (--limit, --rate, --loop)
  ingest/to_s3.py      consumer micro-lots → S3, commit Kafka APRÈS écriture
  ingest/kafka_to_firehose.py  pont Kafka → Firehose DirectPut (500/lot, '\n' par record)
  etl/build_curated.py raw/ → curated/ (dédup, typage, Parquet)
  aws/firehose_direct.py  Firehose DirectPut (up/status/down) — livraison du schéma
  aws/ec2_kafka.py     Kafka open-source sur EC2 (up/status/down) — MSK refusé
  aws/kinesis_ingest.py ingestion 100 % AWS : Kinesis + Firehose + Lambda (up/run/status/down)
  lambda/replay_to_kinesis.py handler : dataset S3 → Kinesis (lots de 500, checkpoint S3)
  agent/brief.py       rapport LLM optionnel, prévisions et métriques actives
infra/aws/
  workflows.py         Step Functions training/inference + horaires EventBridge désactivés
  dashboard_host.py    hébergement Streamlit sur le notebook existant
infra/sagemaker/
  chronos_benchmark.py lance les jobs de benchmark CPU, sans promotion
  chronos_entry.py     backtest complet et artefact Chronos figé
  chronos_inference.py handler de serving, historiques quotidiens complets
  chronos_validate.py  métriques indépendantes et validation du vrai Batch Transform
  lifecycle.py         candidat évalué -> Batch Transform -> manifeste conditionnel
  recovery.py          reprise du registre avec preuve cloud indépendante
  serving.py           layout code/ sans modification des fichiers évalués
  train_entry.py       script exécuté DANS le conteneur (features + LightGBM)
  launch.py            boto3 create_training_job + packaging sourcedir.tar.gz
  tuning.py            HyperParameter Tuning Job (objectif = valid_wape, regex logs)
  preprocess_entry.py  étape Processing : features calculées UNE fois -> features/
  evaluate_entry.py    étape Processing : artefact -> evaluation.json (gate WAPE)
  pipeline.py          SageMaker Pipeline (features -> train -> eval -> gate -> registry)
  deploy.py            endpoint serverless du modele approuve (up/invoke/down)
  register.py          Model Registry (gate WAPE_THRESHOLD -> Approved)
docs/  architecture.md (doc de référence), learner_lab.md (accès au compte AWS),
       windows.md (poste Windows : run.ps1, Git Bash, WSL), script.md (script oral),
       slides.md, diagrams/
scripts/ run.ps1 (équivalent PowerShell du Makefile pour Windows)
tests/ test_metrics, test_features, test_ingest, test_simulate, test_etl,
       test_load_dataset, test_firehose_bridge, test_sagemaker_tuning
```

## Conventions à respecter

- Docstrings et commentaires **en français**, code et noms de variables en anglais.
- CLI en `argparse` + `if __name__ == "__main__": main()`.
- Aucun `os.environ` hors de `config.py` ; ajouter un champ à `Settings` à la place.
- Aucune fuite de données : `stock_on_hand` et `stockout` sont **exclus** des features
  (les ventes sont censurées les jours de rupture). Découpage **temporel** uniquement
  (`time_split`), jamais aléatoire.
- Les tests ne doivent dépendre d'aucun service (Docker, AWS). Un faux consumer Kafka
  existe dans `tests/test_ingest.py`.
- Style d'édition : petites retouches ciblées, pas de réécriture de fichier entier.

## État d'avancement

- ✅ Phase 0 squelette · ✅ Phase 1 données · ✅ Phase 2 features + modèle
  (WAPE **0.430** vs baseline **0.568**, soit −24,3 %)
- ✅ Phase 3 ingestion Kafka → **Firehose DirectPut** → S3 (validée de bout en bout dans AWS :
  20 000 messages, objet Firehose de 164 Ko, 20 000 lignes JSON valides)
- ✅ Phase 3b ETL `raw/` → `curated/` Parquet (dédup + typage)
- ✅ Phase 4a : 48 boosters (médiane et P10/P90 × 16 horizons), profil `median_calendar_v1` :
  WAPE AWS indépendant **0,404595**, RMSLE **0,513383**, couverture **79,22 %**.
  Référence v7 L2 : WAPE **0,422610** ; même catalogue et même test historique.
- ✅ Phase 4b : registre **v8**, artefact de serving vérifié et manifeste actif ; v7 précédent préservé.
  La v5 au mauvais layout de handler n'a jamais été activée et est Rejected.
- ✅ Phase 4c : Batch Transform **19 200 lignes**, 1 200 séries × J+1…J+16 ;
  fichiers immuables `forecast/runs/`, sélection par `deployment/active.json`.
- ✅ Phase 4d : features, training, évaluation et gate du Pipeline exécutés sur AWS.
  Exécution complète du 26 novembre réussie sans reprise, inscription comprise.
  Voir `docs/learner_lab_workflows.md` pour les preuves et les limites du test.
- ✅ HPO : six essais AWS terminés, objectif valid_wape sur J+1.
- ✅ Endpoint serverless v7 : appel réel des 16 horizons, moyenne et quantiles valides.
- ✅ Quotidien EventBridge -> Glue -> inférence -> publication testé avec succès,
  même modèle v7 le 26 novembre, nouveau lot et précédent préservé ; horaires désactivés.
- ✅ Athena : vue `forecast_active` liée au manifeste, 2 KPI exécutés sur AWS.
- ✅ Phase 6 : Streamlit hébergé sur `retail-forecast-nb`, AppTest exécuté sur AWS.
- ✅ Phase 5 rapport LLM : `retail_forecast.agent.brief` (Ollama qwen2.5:3b)
- ✅ `docs/rapport_final.md` et `output/pdf/rapport_final.pdf` : les 4 points du rendu PDF
- ✅ Quantiles P10/P90 : couverture test 78,3 % pour 80 % nominal.
- ✅ Vidéo de démonstration réelle : `output/video/demo_projet4.mp4`, 120 secondes.
- ✅ Rapport PDF, slides corrigées et script de dix minutes : `output/`, `docs/soutenance_10min.md`.
- ✅ Fin du test : notebook arrêté, endpoint supprimé, aucun job actif, horaires désactivés.
- ⏳ Reste à l’étudiant : répétition orale et présentation.

## Expérience LightGBM médian/calendrier (locale)

- `make benchmark-median` : 5 variantes × 16 horizons, 1 200 séries ;
  réplication indépendante seed 20261126, mêmes bruit et ruptures.
- Sélection uniquement sur validation : quantile alpha 0,5 + calendrier cible.
  L1 + calendrier est pratiquement à égalité sur cette validation.
- Test réservé de cette réplication : WAPE **0,394716** contre **0,413857**
  pour le témoin L2 réentraîné ; gain relatif **4,63 %**, 16 horizons améliorés.
- `docs/median_calendar_benchmark.md`, `reports/median_calendar_benchmark.json`.
  Ce nouveau score ne remplace pas celui de v7 sur l'ancien jeu.
- `evaluate/median_project_check.py` : diagnostic secondaire du candidat figé
  contre l'artefact v7, sur l'ancien test déjà consulté ; aucune sélection.
  WAPE **0,404560** contre v7 **0,422610**, 16 horizons améliorés ; le résultat
  Chronos **0,414794** porte sur ces mêmes données, mais ce test reste réutilisé.
- L'expérience initiale était locale et ponctuelle. Le profil est désormais
  intégré au train local/AWS, à l'évaluation indépendante, au Batch Transform
  et au HPO ; 169 tests passent (profils ancien/nouveau, brut/prétraité).
  Voir `docs/median_production.md` et `docs/evidence/median_aws_integration.json`
  pour le statut de l'exécution AWS et la version effectivement active.
- Migration terminée : **v8 actif**, workflow `median-calendar-20261126-214959`
  SUCCEEDED, 19 200 sorties P50/P10/P90 exactement identiques à l'artefact.
  Catalogue actif de 876 000 lignes identique au benchmark ; 50 000 lignes de
  séries retirées restent dans curated mais sont exclues du périmètre d'entraînement.
  Tous les jobs sont terminés, modèle temporaire supprimé, notebook arrêté,
  aucun endpoint et horaires désactivés. Aucun nouvel essai HPO ou endpoint v8.

## Benchmark Chronos-2 (exploratoire)

- `docs/chronos_benchmark.md`, `reports/chronos2_benchmark.json` : 120 séries,
  20 magasins, 48 articles, 16 horizons × 28 origines, 53 760 couples appariés.
- WAPE : Chronos-2 **0,413009**, LightGBM v7 **0,423396** sur ce même sous-ensemble.
  Gain relatif **2,45 %** ; couverture 79,65 % contre 79,01 %.
- Exécuté localement sur CPU, sans fine-tuning et sans nouveau job AWS.
  Lors du pilote le modèle actif était **v7**. Ne pas extrapoler le pilote aux 1 200 séries.

## Chronos-2 — catalogue complet et AWS validés

- `docs/chronos_full_validation.md`, `reports/chronos2_full_benchmark.json` :
  **1 200 séries**, 16 horizons × 28 origines, **537 600 couples appariés**.
- WAPE : Chronos **0,414794**, référence v7 **0,422610**, gain relatif **1,85 %**.
  RMSLE 0,520840 contre 0,538387 ; couverture 79,53 % contre 78,29 %.
- Six jobs `ml.m5.xlarge` terminés ; les GPU et plus grosses instances ont été
  refusés par IAM. Aucune politique IAM modifiée, aucun fine-tuning.
- Batch Transform CPU **19 200 lignes** vérifié contre le handler évalué ; modèle
  temporaire supprimé, aucun calcul actif, horaires désactivés, notebook arrêté.
- Chronos n'a pas été activé : les workflows mensuel/quotidien restent spécifiques LightGBM.
  La version active est maintenant v8 médiane (voir la migration ci-dessus).
  Ne pas prétendre que Chronos est activé ou qu’un endpoint Chronos a été testé.

## Décisions déjà arbitrées (ne pas les reproposer)

1. **Pas d'agent LLM** dans le périmètre actuel — le LLM ne sert qu'à un rapport
   textuel optionnel. Les diagrammes ont été scindés en `architecture_training.*`
   et `architecture_inference.*` pour cette raison.
2. **s3mock** remplace MinIO (MinIO retiré de Docker Hub, LocalStack 2026 exige un
   token payant). Même code boto3 : sur AWS on laisse `S3_ENDPOINT_URL=` vide.
3. **Kafka open-source sur EC2** remplace MSK sur AWS (MSK refusé : `kafka:ListClustersV2`
   → AccessDenied) ; **Firehose DirectPut** assure la livraison, via un pont
   `ingest/kafka_to_firehose.py`. En local : Redpanda + s3mock.
4. **Streamlit** remplace QuickSight. **Ollama / JumpStart** remplacent Bedrock.
   Le PDF autorise explicitement ces substitutions (slides 10-11).
5. Horizon aligné sur `aws presentation.pdf` : les 16 jours J+1 à J+16.
   Le parcours recommandé est `infra.aws.workflows` ; la publication passe par
   `deployment/active.json`, jamais par le dernier package Approved seul.
6. Diagrammes = **cible** : consulter `docs/learner_lab_workflows.md` pour
   les composants réellement validés dans AWS.

## Pièges rencontrés

- `AWS_PROFILE=""` dans `.env` fait planter boto3 → `config.py` supprime les variables
  AWS vides au chargement. Ne pas « corriger » ça.
- `kafka-python` 3.x : `value_deserializer` reçoit des `bytes`, décoder en UTF-8 soi-même.
- `.env` est gitignoré ; `cp .env.example .env` avant toute commande qui touche S3/Kafka.

## Parcours AWS corrigé

Voir `docs/learner_lab_workflows.md`. `make aws-workflows-up` déploie les deux
workflows, horaires désactivés ; `make aws-pipeline-run` entraîne et valide,
`make aws-inference-run` utilise le modèle actif. `make aws-dashboard` démarre
ou actualise Streamlit sur le notebook existant. Les dates manquantes restent
inconnues : lags, cibles, rolling et split sont calendaires, sans imputation à zéro.
Les anciennes commandes de registre et leur Lambda ne peuvent plus approuver
automatiquement sans le pipeline indépendant.
