.PHONY: help setup data test train train-quick diagram dashboard clean

PY  := .venv/bin/python
PIP := .venv/bin/pip

help:
	@echo "make setup        venv (.venv) + dépendances"
	@echo "make data         876k lignes synthétiques -> data/raw/sales.parquet"
	@echo "make test         pytest (30 tests)"
	@echo "make train        LightGBM local complet vs baseline saisonnière"
	@echo "make train-quick  idem sur 20 % des séries (~3 s)"
	@echo "make diagram      régénère docs/diagrams/architecture.{png,svg}"
	@echo "make dashboard    Streamlit (Phase 6, pas encore implémenté)"
	@echo "make clean        caches + rapports"

setup:
	python3 -m venv --system-site-packages .venv
	$(PIP) install -e . -r requirements.txt

data:
	$(PY) -m retail_forecast.data.simulate --out data/raw/sales.parquet

test:
	$(PY) -m pytest -q

train:
	$(PY) -m retail_forecast.train.train_lgbm

train-quick:
	$(PY) -m retail_forecast.train.train_lgbm --sample 0.2 --estimators 200

diagram:
	$(PY) docs/diagrams/architecture.py

dashboard:
	@echo "Phase 6 — pas encore implémenté"
	@exit 1

clean:
	rm -rf .pytest_cache reports/*.json
	find . -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} +
