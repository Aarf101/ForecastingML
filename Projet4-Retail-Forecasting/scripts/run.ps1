#requires -Version 5.1
<#
    run.ps1 — équivalent PowerShell des cibles Makefile, pour Windows natif.

    Le code applicatif est identique ; seuls les scripts .sh (bash) ne passent
    pas sous cmd/PowerShell sans Git Bash. Ce script couvre les tâches usuelles.

    Usage :
        .\scripts\run.ps1 setup
        .\scripts\run.ps1 test
        .\scripts\run.ps1 train
        .\scripts\run.ps1 etl
        .\scripts\run.ps1 dashboard
        .\scripts\run.ps1 aws-inventory

    Si l'exécution est bloquée :
        powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 test
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet(
        'help', 'setup', 'env', 'test', 'data', 'train', 'train-quick',
        'etl', 'diagram', 'slides', 'dashboard', 'brief',
        'aws-inventory', 'aws-setup', 'aws-bootstrap', 'aws-run',
        'aws-upload-repo', 'aws-teardown'
    )]
    [string]$Task = 'help'
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

# --- Interpréteur : priorité au venv local ---------------------------------
$py = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    $py = (Get-Command python -ErrorAction SilentlyContinue).Source
}
if (-not $py) {
    Write-Error "Aucun Python trouvé. Lancez d'abord : .\scripts\run.ps1 setup (ou installez Python 3.10+)."
    exit 1
}

function Invoke-Py {
    param([string[]]$Arguments)
    Write-Host "> python $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $py @Arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Task) {

    'help' {
        Write-Host @"
Usage : .\scripts\run.ps1 <tache>

  setup        crée .venv et installe les dépendances
  env          copie .env.example -> .env
  test         pytest (48 tests, aucun service requis)
  data         génère data/raw/sales.parquet (876k lignes)
  train        entraînement LightGBM local (~40 s)
  train-quick  idem sur 15 % des lignes
  etl          raw/ -> curated/ Parquet (miroir local du job Glue)
  diagram      régénère les 5 diagrammes
  slides       régénère docs/slides.pptx
  dashboard    Streamlit sur http://localhost:8501
  brief        rapport hebdo via Ollama

  aws-inventory    inventaire des ressources AWS présentes
  aws-setup        bucket S3 + upload du dataset
  aws-bootstrap    crée toute l'infrastructure AWS
  aws-run          pipeline complet (ingestion -> transform -> athena)
  aws-upload-repo  dépose le code dans S3 (pour bootstrap dans le lab)
  aws-teardown     arrête endpoint + notebook

Les scripts .sh (make up / produce / consume) nécessitent Git Bash ou WSL.
"@
    }

    'setup' {
        if (-not (Test-Path '.venv')) {
            Invoke-Py @('-m', 'venv', '.venv')
        }
        $pip = Join-Path (Get-Location) '.venv\Scripts\pip.exe'
        Write-Host "> pip install -e . -r requirements.txt" -ForegroundColor DarkGray
        & $pip install -e . -r requirements.txt
    }

    'env' {
        if (Test-Path '.env') { Write-Host '.env existe déjà.' }
        else { Copy-Item '.env.example' '.env'; Write-Host '.env créé.' }
    }

    'test'          { Invoke-Py @('-m', 'pytest', '-q') }
    'data'          { Invoke-Py @('-m', 'retail_forecast.data.simulate', '--out', 'data/raw/sales.parquet') }
    'train'         { Invoke-Py @('-m', 'retail_forecast.train.train_lgbm') }
    'train-quick'   { Invoke-Py @('-m', 'retail_forecast.train.train_lgbm', '--sample', '0.2', '--estimators', '200') }
    'etl'           { Invoke-Py @('-m', 'retail_forecast.etl.build_curated') }
    'diagram' {
        foreach ($d in 'architecture', 'architecture_simple', 'architecture_medium',
                       'architecture_split', 'architecture_overview') {
            $f = "docs/diagrams/$d.py"
            if (Test-Path $f) { Invoke-Py @($f) }
        }
    }
    'slides'        { Invoke-Py @('docs/build_slides.py') }
    'dashboard'     { Invoke-Py @('-m', 'streamlit', 'run', 'dashboards/streamlit/app.py', '--server.port', '8501') }
    'brief'         { Invoke-Py @('-m', 'retail_forecast.agent.brief') }

    'aws-inventory'    { Invoke-Py @('-m', 'infra.aws.setup', '--inventory') }
    'aws-setup'        { Invoke-Py @('-m', 'infra.aws.setup') }
    'aws-bootstrap'    { Invoke-Py @('-m', 'infra.aws.setup') }
    'aws-run'          { Invoke-Py @('-m', 'infra.aws.full_run') }
    'aws-upload-repo'  { & bash scripts/upload_repo.sh }
    'aws-teardown'     { & bash scripts/aws_teardown.sh }
}
