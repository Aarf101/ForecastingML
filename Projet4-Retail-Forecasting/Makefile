.PHONY: help setup env up down ps logs data test train train-quick diagram \
        produce consume e2e slides lettre dashboard clean aws-setup etl \
        aws-firehose-up aws-firehose-status aws-firehose-down ingest-firehose \
        kafka-all kafka-up kafka-run kafka-status kafka-down \
        dataset-info ingest-dataset \
        aws-ingest-all aws-ingest-all-down \
        sagemaker-train

# --- Détection de l'OS -------------------------------------------------------
# Linux / macOS / WSL : .venv/bin        Windows (Git Bash, cmd) : .venv/Scripts
ifeq ($(OS),Windows_NT)
  VENV_BIN := .venv/Scripts
  PYTHON   ?= python
else
  VENV_BIN := .venv/bin
  PYTHON   ?= python3
endif

PY  := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip

help:
	@echo "make setup        venv (.venv) + dépendances"
	@echo "make env          copie .env.example -> .env (S3 local par défaut)"
	@echo "make up / down    Redpanda + console + S3 local (docker compose)"
	@echo "make aws-setup    première mise en route AWS : bucket S3 + upload (profil lab)"
	@echo "make cfn-import   adopte le bucket + registre + catalogue dans CloudFormation"
	@echo "make cfn-deploy   cree la stack de zero (bucket neuf)"
	@echo "make cfn-drift    detecte les ecarts template <-> ressources"
	@echo "make aws-teardown arret de l'endpoint et du notebook (dry-run)"
	@echo "make aws-teardown-go  idem, en supprimant vraiment"
	@echo "make etl          raw/ -> curated/ Parquet (miroir local du job Glue)"
	@echo "make aws-ingest-prep  parquet -> morceaux JSONL gzip dans s3://.../replay/"
	@echo "make aws-ingest-up    cree Kinesis + Firehose + Lambda (ingestion 100% AWS)"
	@echo "make aws-ingest-run   rejoue le dataset dans Kinesis (via Lambda)"
	@echo "make aws-ingest-status etat Kinesis/Firehose/Lambda + objets raw/sales/"
	@echo "make aws-ingest-all  TOUT : bucket, Firehose, EC2+Kafka, ingestion, Glue"
	@echo "make aws-setup-only  cree l'environnement et s'arrete (test via la console)"
	@echo "make aws-ingest-all-down idem puis supprime EC2 + Firehose"
	@echo "make dataset-info  inspecte un dataset quelconque (SOURCE=...)"
	@echo "make ingest-dataset convertit un dataset vers S3 (SOURCE=... DEST=...)"
	@echo "make kafka-all    EC2 + Kafka + deps + producteur + pont Firehose (7 min)"
	@echo "make kafka-up     cree l'EC2 Kafka et installe pip + le depot"
	@echo "make kafka-run    rejoue 20 000 ventes et pousse vers Firehose"
	@echo "make kafka-status etat du Kafka (IP + conteneur)"
	@echo "make kafka-down   supprime l'EC2 et le groupe de securite"
	@echo "make aws-firehose-up  Kafka -> Firehose DirectPut (schema de la diapo)"
	@echo "make ingest-firehose  pont Kafka -> Firehose (sur l'EC2, dans le compte)"
	@echo "make aws-ingest-down  supprime Kinesis + Firehose + Lambda"
	@echo "make aws-pipeline-up  Step Functions + EventBridge (orchestration AWS)"
	@echo "make aws-pipeline-run execute le pipeline de bout en bout"
	@echo "make aws-pipeline-status etat machine/regle/executions"
	@echo "make aws-pipeline-down supprime machine + regle + lambda"
	@echo "make aws-full-run     pipeline complet dans AWS (ingestion -> transform -> athena)"
	@echo "make aws-upload-repo  depose le code dans S3 (pour bootstrap dans le lab)"
	@echo "make aws-run          idem aws-full-run, en Python pur (sans bash)"
	@echo "make aws-bootstrap    cree TOUTE l infrastructure (bucket, Kinesis, Glue, Step Functions)"
	@echo "make aws-inventory    inventaire des ressources presentes"
	@echo "make sagemaker-train  Training Job LightGBM sur SageMaker (--wait)"
	@echo "make sagemaker-register  enregistre le modele (gate WAPE)"
	@echo "make sagemaker-transform Batch Transform -> forecast/"
	@echo "make athena       tables externes + KPI SQL"
	@echo "make dashboard    Streamlit :8501 (prevu vs realise)"
	@echo "make brief        rapport hebdo redige par Ollama"
	@echo "make data         876k lignes synthétiques -> data/raw/sales.parquet"
	@echo "make test         pytest"
	@echo "make train        LightGBM local complet vs baseline saisonnière"
	@echo "make train-quick  idem sur 20 % des séries (~3 s)"
	@echo "make produce      rejoue 5 000 ventes dans Kafka (sales.events)"
	@echo "make consume      écrit les micro-lots dans S3 local (raw/sales/...)"
	@echo "make e2e          up + produce + consume + vérification de bout en bout"
	@echo "make diagram      régénère docs/diagrams/architecture.{png,svg}"
	@echo "make slides       régénère docs/slides.pptx (5 slides, exposé idée)"
	@echo "make lettre       régénère docs/lettre_motivation.pdf (lettre de motivation)"

setup:
	$(PYTHON) -m venv --system-site-packages .venv
	$(PIP) install -e . -r requirements.txt

env:
	@test -f .env || cp .env.example .env
	@echo ".env prêt (S3 local local par défaut)"

up:
	docker compose up -d
	@echo "Kafka localhost:19092 | console http://localhost:8080 | S3 local http://localhost:9090"

down:
	docker compose down

ps:
	docker compose ps

logs:
	docker compose logs --tail=50 -f

data:
	$(PY) -m retail_forecast.data.simulate --out data/raw/sales.parquet

test:
	$(PY) -m pytest -q

train:
	$(PY) -m retail_forecast.train.train_lgbm

train-quick:
	$(PY) -m retail_forecast.train.train_lgbm --sample 0.2 --estimators 200

produce:
	$(PY) -m retail_forecast.ingest.producer --limit 5000

consume:
	$(PY) -m retail_forecast.ingest.to_s3 --max-messages 5000

e2e:
	bash scripts/e2e_ingest.sh

diagram:
	$(PY) docs/diagrams/architecture.py
	$(PY) docs/diagrams/architecture_simple.py
	$(PY) docs/diagrams/architecture_medium.py
	$(PY) docs/diagrams/architecture_split.py

slides:
	$(PY) docs/build_slides.py

lettre:
	cd docs && pdflatex -interaction=nonstopmode -halt-on-error lettre_motivation.tex > /dev/null
	@rm -f docs/lettre_motivation.aux docs/lettre_motivation.log docs/lettre_motivation.out
	@echo "docs/lettre_motivation.pdf régénéré"

sagemaker-register:
	$(PY) -m infra.sagemaker.register

sagemaker-transform:
	$(PY) -m infra.sagemaker.batch_transform

athena:
	$(PY) -m infra.athena.setup

brief:
	$(PY) -m retail_forecast.agent.brief

dashboard:
	$(PY) -m streamlit run dashboards/streamlit/app.py --server.port 8501

aws-setup:
	@bash scripts/aws_setup.sh $(SUFFIX)

cfn-import:
	@CONFIRM_DROP_TABLES=1 bash scripts/cfn.sh import

cfn-deploy:
	@bash scripts/cfn.sh deploy

cfn-drift:
	@bash scripts/cfn.sh drift

aws-teardown:
	@bash scripts/aws_teardown.sh --dry-run

aws-teardown-go:
	@bash scripts/aws_teardown.sh --yes

etl:
	$(PY) -m retail_forecast.etl.build_curated

# --- ingestion 100 % AWS : Kinesis -> Firehose -> S3 raw/sales/ ---------------
aws-ingest-prep:
	$(PY) scripts/prepare_replay_chunks.py --rows 20000 --delete

# --- n'importe quel jeu de données -> schéma du projet ------------------------
# make ingest-dataset SOURCE=data/raw/m5.csv DEST=s3://$S3_BUCKET/raw/historique/m5.parquet
SOURCE ?=
DEST   ?=

dataset-info:
	$(PY) -m retail_forecast.data.load --source "$(SOURCE)"

ingest-dataset:
	$(PY) -m retail_forecast.data.load --source "$(SOURCE)" --out "$(DEST)"

# --- ingestion de bout en bout : une seule commande, tout créé ---------------
aws-ingest-all:
	$(PY) -m infra.aws.ingest_all

aws-setup-only:
	$(PY) -m infra.aws.ingest_all --setup-only

aws-ingest-all-down:
	$(PY) -m infra.aws.ingest_all --down

# --- Kafka sur EC2 : tout en un (instance + deps + producteur + pont) ---------
kafka-all:
	$(PY) -m infra.aws.ec2_kafka all --limit 20000

kafka-up:
	$(PY) -m infra.aws.ec2_kafka up --with-deps

kafka-run:
	$(PY) -m infra.aws.ec2_kafka run --limit 20000

kafka-status:
	$(PY) -m infra.aws.ec2_kafka status

kafka-down:
	$(PY) -m infra.aws.ec2_kafka down


# Kafka est auto-hébergé (MSK refusé : kafka:* -> AccessDenied) ; Firehose fait
# le tamponnage 60 s / 5 Mo, la compression gzip et le rangement sous raw/sales/.
aws-firehose-up:
	$(PY) -m infra.aws.firehose_direct up

aws-firehose-status:
	$(PY) -m infra.aws.firehose_direct status

aws-firehose-down:
	$(PY) -m infra.aws.firehose_direct down

ingest-firehose:
	$(PY) -m retail_forecast.ingest.kafka_to_firehose --max-messages 20000

aws-ingest-up:
	$(PY) -m infra.aws.kinesis_ingest up

aws-ingest-run:
	$(PY) -m infra.aws.kinesis_ingest run --max-seconds 600

aws-ingest-status:
	$(PY) -m infra.aws.kinesis_ingest status

aws-ingest-down:
	$(PY) -m infra.aws.kinesis_ingest down

# --- orchestration 100 % AWS : Step Functions + EventBridge ------------------
aws-pipeline-up:
	$(PY) -m infra.aws.orchestrate up

aws-pipeline-run:
	$(PY) -m infra.aws.orchestrate run

aws-pipeline-status:
	$(PY) -m infra.aws.orchestrate status

aws-pipeline-asl:
	$(PY) -m infra.aws.orchestrate print

aws-pipeline-down:
	$(PY) -m infra.aws.orchestrate down

aws-full-run:
	@bash scripts/aws_full_run.sh $(ARGS)

aws-upload-repo:
	@bash scripts/upload_repo.sh

aws-run:
	$(PY) -m infra.aws.full_run $(ARGS)

aws-bootstrap:
	$(PY) -m infra.aws.setup $(ARGS)

aws-inventory:
	$(PY) -m infra.aws.setup --inventory


sagemaker-train:
	$(PY) -m infra.sagemaker.launch --wait

clean:
	rm -rf .pytest_cache reports/*.json
	find . -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} +

