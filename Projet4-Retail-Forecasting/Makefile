.PHONY: help setup env up down ps logs data test train train-quick diagram \
        produce consume e2e dashboard clean

PY  := .venv/bin/python
PIP := .venv/bin/pip

help:
	@echo "make setup        venv (.venv) + dépendances"
	@echo "make env          copie .env.example -> .env (S3 local par défaut)"
	@echo "make up / down    Redpanda + console + S3 local (docker compose)"
	@echo "make data         876k lignes synthétiques -> data/raw/sales.parquet"
	@echo "make test         pytest"
	@echo "make train        LightGBM local complet vs baseline saisonnière"
	@echo "make train-quick  idem sur 20 % des séries (~3 s)"
	@echo "make produce      rejoue 5 000 ventes dans Kafka (sales.events)"
	@echo "make consume      écrit les micro-lots dans S3 local (raw/sales/...)"
	@echo "make e2e          up + produce + consume + vérification de bout en bout"
	@echo "make diagram      régénère docs/diagrams/architecture.{png,svg}"

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

dashboard:
	@echo "Phase 6 — pas encore implémenté"
	@exit 1

clean:
	rm -rf .pytest_cache reports/*.json
	find . -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} +

