# AGENTS.md — guide de travail pour un agent IA

Projet **Retail ML Forecasting** (Projet 4) : prévision de la demande J+1…J+28 par
(magasin, article) sur AWS. Ce fichier est la carte du dépôt. **Le lire en premier
évite d'explorer le code** : ne relire un fichier que si la tâche le concerne.

## Commandes (Makefile)

| Commande | Effet |
|---|---|
| `make test` | pytest (48 tests, ~1 s, aucun service externe requis) |
| `make train` | entraînement complet LightGBM → `reports/local_baseline.json`, `data/forecast/` |
| `make train-quick` | même chose sur 15 % des lignes (itération) |
| `make up` / `make down` | Redpanda (:19092, console :8080) + s3mock (:9090) |
| `make produce` / `make consume` | Phase 3 : ventes → topic `sales.events`, puis topic → `raw/` S3 |
| `make e2e` | enchaîne tout et échoue si aucun objet n'arrive dans S3 |
| `make etl` | `raw/` -> `curated/` Parquet (miroir local du job Glue) |
| `make sagemaker-train` | Training Job LightGBM sur AWS (`infra/sagemaker/launch.py --wait`) |
| `make diagram` | régénère les 5 variantes de diagrammes (matplotlib uniquement) |
| `make slides` | régénère `docs/slides.pptx` depuis `docs/build_slides.py` |
| `make sagemaker-register` | Model Registry (gate `WAPE_THRESHOLD`) |
| `make sagemaker-transform` | Batch Transform → `forecast/` |
| `make athena` | tables externes + 2 KPI SQL |
| `make dashboard` | Streamlit :8501 |
| `make brief` | rapport hebdo Ollama → `reports/` + S3 |
| `make lettre` | régénère `docs/lettre_motivation.pdf` (pdflatex, nécessite texlive) |

Toujours utiliser `.venv/bin/python` (`python3` système n'a ni lightgbm ni kafka-python).

## Carte du code

```
src/retail_forecast/
  config.py            SETTINGS figé depuis .env — SEUL endroit qui lit os.environ
  data/simulate.py     générateur de ventes (876k lignes, 20 magasins × 60 articles × 2 ans)
  features/build.py    lags 1→35, rolling mean/std 7/14/28, calendrier, time_split()
  evaluate/metrics.py  WAPE, MAPE, RMSE, pinball loss
  train/train_lgbm.py  miroir local du Training Job SageMaker + baseline saisonnière
  ingest/storage.py    unique porte boto3 : raw_key(), gzip JSONL, chunking
  ingest/producer.py   rejoue le parquet → Kafka (--limit, --rate, --loop)
  ingest/to_s3.py      consumer micro-lots → S3, commit Kafka APRÈS écriture
  etl/build_curated.py raw/ → curated/ (dédup, typage, Parquet)
  aws/kinesis_ingest.py ingestion 100 % AWS : Kinesis + Firehose + Lambda (up/run/status/down)
  lambda/replay_to_kinesis.py handler : dataset S3 → Kinesis (lots de 500, checkpoint S3)
  agent/, inference/   vides (Phase 5 / Phase 4)
infra/sagemaker/
  train_entry.py       script exécuté DANS le conteneur (features + LightGBM)
  launch.py            boto3 create_training_job + packaging sourcedir.tar.gz
  register.py          Model Registry (gate WAPE_THRESHOLD -> Approved)
docs/  architecture.md (doc de référence), learner_lab.md (accès au compte AWS),
       windows.md (poste Windows : run.ps1, Git Bash, WSL), script.md (script oral),
       slides.md, diagrams/
scripts/ run.ps1 (équivalent PowerShell du Makefile pour Windows)
tests/ test_metrics, test_features, test_ingest, test_simulate
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
- ✅ Phase 3 ingestion Kafka → micro-lots gzip → S3 (validée de bout en bout)
- ✅ Phase 3b ETL `raw/` → `curated/` Parquet (dédup + typage)
- ✅ Phase 4a Training Job SageMaker (`ml.m5.large`, 2,7 min, ~0,005 €) :
  WAPE **0.4298** vs baseline 0.5683 côté cloud — identique au local (0.4291)
- ✅ Phase 4b Model Registry : v1 `PendingManualApproval`, v2 `Approved`
  (`WAPE_THRESHOLD=0.45` ; la v1 montre le fonctionnement du gate)
- ✅ Phase 4c Batch Transform → `forecast/table/forecast.parquet` (4 800 lignes,
  4 horizons) · `inference.py` + `SAGEMAKER_REQUIREMENTS` dans l'artefact
- ✅ Athena : base `retail_forecast`, tables `curated_sales` / `forecast`, 2 KPI
- ✅ Phase 6 dashboard Streamlit (`make dashboard`, S3 ou local)
- ✅ Phase 5 rapport LLM : `retail_forecast.agent.brief` (Ollama qwen2.5:3b)
- ✅ `docs/report.md` : les 4 points du rendu PDF
- ⏳ Reste : vidéo 2 min, répétition orale, quantiles P10/P90

## Décisions déjà arbitrées (ne pas les reproposer)

1. **Pas d'agent LLM** dans le périmètre actuel — le LLM ne sert qu'à un rapport
   textuel optionnel. Les diagrammes ont été scindés en `architecture_training.*`
   et `architecture_inference.*` pour cette raison.
2. **s3mock** remplace MinIO (MinIO retiré de Docker Hub, LocalStack 2026 exige un
   token payant). Même code boto3 : sur AWS on laisse `S3_ENDPOINT_URL=` vide.
3. **Redpanda** remplace MSK en local (API Kafka identique). Sur AWS : MSK ou Kinesis.
4. **Streamlit** remplace QuickSight. **Ollama / JumpStart** remplacent Bedrock.
   Le PDF autorise explicitement ces substitutions (slides 10-11).
5. Horizon configuré à 28 jours (`FORECAST_HORIZON`) ; les diapositives parlent
   parfois de 16 jours — écart connu, non tranché.
6. Diagrammes = **cible**, pas « as-built » : seules les étapes données, Kafka, S3 et
   l'entraînement local tournent réellement aujourd'hui.

## Pièges rencontrés

- `AWS_PROFILE=""` dans `.env` fait planter boto3 → `config.py` supprime les variables
  AWS vides au chargement. Ne pas « corriger » ça.
- `kafka-python` 3.x : `value_deserializer` reçoit des `bytes`, décoder en UTF-8 soi-même.
- `.env` est gitignoré ; `cp .env.example .env` avant toute commande qui touche S3/Kafka.
