# Travailler sous Windows

Le code applicatif est portable (2 800 lignes de Python, `pathlib` partout).
Seul l'outillage Unix — `Makefile` et les 7 scripts `.sh` — demande un
interpréteur compatible. Trois chemins possibles, du plus simple au plus complet.

## Ce qui marche tel quel

| Élément | Statut |
|---|---|
| `src/retail_forecast/**` | ✅ `pathlib` uniquement, aucun appel shell |
| `infra/**` (SageMaker, AWS, Lambda, Glue) | ✅ boto3 + `sys.executable` (`full_run.py`) |
| `docs/diagrams/*.py` | ✅ matplotlib |
| `docs/build_slides.py` | ✅ python-pptx |
| `docker-compose.yml` | ✅ Docker Desktop (conteneurs Linux) |
| `pyproject.toml` | ✅ `requires-python >=3.10` |

Les chemins `/opt/ml/...` dans `infra/sagemaker/train_entry.py` ne posent pas de
problème : ce fichier s'exécute **dans le conteneur SageMaker** (Linux), jamais
sur Windows.

## Ce qui demande un adaptateur

| Élément | Raison | Solution |
|---|---|---|
| `make` | absent de Windows | `scripts/run.ps1` (ci-dessous) |
| `scripts/*.sh` | bash : `set -euo pipefail`, `$(...)` | Git Bash ou WSL |
| `make lettre` | pdflatex | MiKTeX (`choco install miktex`) |

Le `Makefile` détecte désormais l'OS (`ifeq ($(OS),Windows_NT)`) et pointe sur
`.venv/Scripts/python.exe` au lieu de `.venv/bin/python`. Les scripts `.sh`
cherchent aussi `.venv/Scripts/python.exe` puis retombent sur `python`.

---

## Chemin 1 — PowerShell natif (recommandé)

```powershell
# 1. environnement
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e . -r requirements.txt

# 2. données et modèle
python -m retail_forecast.data.simulate --out data/raw/sales.parquet
python -m retail_forecast.train.train_lgbm
python -m pytest -q

# 3. dashboard
python -m streamlit run dashboards/streamlit/app.py --server.port 8501
```

Raccourci : le même enchaînement via le script fourni.

```powershell
.\scripts\run.ps1 setup
.\scripts\run.ps1 test
.\scripts\run.ps1 train
.\scripts\run.ps1 dashboard
.\scripts\run.ps1 help        # liste des tâches
```

Si PowerShell refuse l'exécution :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 test
```

## Chemin 2 — Git Bash (accès aux scripts .sh)

Installez Git for Windows, puis dans **Git Bash** :

```bash
make setup
make test
make train
make up && make produce && make consume && make e2e
```

Fonctionne aussi depuis PowerShell si `make` et `bash` sont dans le `PATH` :

```powershell
& bash scripts/e2e_ingest.sh
```

## Chemin 3 — WSL2 (le plus proche de la référence)

```powershell
wsl --install -d Ubuntu
```

Puis dans Ubuntu, montez le dépôt :

```bash
cd /mnt/c/Users/<vous>/projaws
sudo apt install -y python3-venv make docker.io
make setup && make test
```

⚠️ Docker Desktop doit avoir l'intégration WSL activée (Settings → Resources →
WSL integration), sinon `make up` échoue sur `docker: command not found`.

---

## Identifiants AWS sous Windows

Le fichier est au même endroit relatif, mais sous `%USERPROFILE%` :

```
C:\Users\<vous>\.aws\credentials
C:\Users\<vous>\.aws\config
```

Contenu (session AWS Academy) :

```ini
[lab]
aws_access_key_id = ASIA...
aws_secret_access_key = ...
aws_session_token = ...
```

```ini
[profile lab]
region = us-east-1
```

Puis, dans PowerShell :

```powershell
$env:AWS_PROFILE = "lab"
aws sts get-caller-identity
```

Pour la session courante seulement. Pour la rendre persistante :

```powershell
[Environment]::SetEnvironmentVariable("AWS_PROFILE", "lab", "User")
```

---

## Erreurs fréquentes et correctifs

| Message | Cause | Correctif |
|---|---|---|
| `make : command not found` | pas de make | `scripts/run.ps1`, ou `choco install make` |
| `'\r': command not found` | scripts `.sh` convertis en CRLF | `.gitattributes` présent → `git add --renormalize .` puis re-clone si besoin |
| `The term 'Activate.ps1' is not recognized` | politique d'exécution | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `python : not found` (Git Bash) | Python non dans le PATH | utiliser `py -3` ou réinstaller Python en cochant « Add to PATH » |
| `Unable to create process ... .venv\bin\python` | ancien venv Linux | supprimer `.venv` puis `.\scripts\run.ps1 setup` |
| `docker compose` échoue | Docker Desktop arrêté | démarrer Docker Desktop, attendre le statut « Running » |
| Chemins trop longs | limite 260 caractères | activer `git config --system core.longpaths true` |

## Fins de ligne

`.gitattributes` force `eol=lf` pour les `.sh` et `eol=lf` pour les `.ps1`
(cmd/PowerShell lisent le LF). Si votre clone a été fait **avant** l'ajout de ce
fichier :

```bash
git add --renormalize .
git commit -m "normalisation des fins de ligne"
```
