.PHONY: help setup env up down ps logs data test train train-quick diagram \
        produce consume e2e slides lettre dashboard clean aws-setup etl \
        sagemaker-train

PY  := .venv/bin/python
PIP := .venv/bin/pip

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
	python3 -m venv --system-site-packages .venv
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

sagemaker-train:
	$(PY) -m infra.sagemaker.launch --wait

clean:
	rm -rf .pytest_cache reports/*.json
	find . -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} +

