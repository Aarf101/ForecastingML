"""Étape de préparation des features du SageMaker Pipeline (conteneur Processing).

Calcule **une seule fois** les features (calendrier, lags 1→35, moyennes et
écarts-types glissants 7/14/28) et les écrit dans ``s3://<bucket>/features/``.
L'entraînement, l'évaluation, le Batch Transform et l'endpoint relisent ensuite
cette même table.

Pourquoi une étape dédiée : sans elle, les features sont recalculées par chaque
consommateur — conteneur d'entraînement, client de Batch Transform, échantillon
envoyé à l'endpoint. Si l'un d'eux change un paramètre (``LAGS``, ``ROLLS``),
l'entraînement et l'inférence ne parlent plus le même espace de features : c'est
le *train/serve skew*, le défaut le plus courant et le plus silencieux en MLOps.
Avec cette étape, la définition des features vit à un seul endroit.

Montage SageMaker :

* entrée  : ``/opt/ml/processing/input/curated/`` (Parquet de ``curated/sales/``)
* entrée  : ``/opt/ml/processing/input/code/sourcedir.tar.gz`` (paquet du projet)
* sortie  : ``/opt/ml/processing/output/features/features.parquet``
* sortie  : ``/opt/ml/processing/output/features.json`` (contrôle d'exécution)

Ce script tourne **dans le job Processing**, jamais en local.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

INPUT = Path("/opt/ml/processing/input/curated")
CODE = Path("/opt/ml/processing/input/code")
OUTPUT = Path("/opt/ml/processing/output")

# Colonnes produites par build_features() et attendues par les consommateurs :
# leur présence signale une table déjà préparée (voir train_entry / evaluate_entry).
PREPROCESSED = ("lag_1", "roll_mean_7")


def _install(packages: list[str]) -> bool:
    """Installe les dépendances absentes (l'image SKLearn n'a pas pyarrow)."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *packages],
            check=True,
        )
        print(f"[preprocess] dépendances installées : {' '.join(packages)}")
        return True
    except Exception as exc:  # noqa: BLE001 - message clair plutôt que traceback
        print(f"[preprocess] installation impossible : {exc}")
        return False


def _load_package() -> None:
    """Réextrait le paquet du projet pour importer ``retail_forecast``."""
    source = next(CODE.glob("*.tar.gz"), None)
    if source is None:
        raise SystemExit(f"paquet de code absent de {CODE}")
    tmp = Path(tempfile.mkdtemp(prefix="pkg-"))
    with tarfile.open(source, mode="r:gz") as tar:
        tar.extractall(tmp)
    sys.path.insert(0, str(tmp))
    print(f"[preprocess] paquet du projet chargé depuis {source.name}")


def main() -> int:
    try:
        import pandas as pd
    except ImportError:
        if not _install(["pandas", "pyarrow"]):
            raise SystemExit("pandas indisponible dans le conteneur")
        import pandas as pd

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        _install(["pyarrow"])

    fichiers = sorted(INPUT.rglob("*.parquet"))
    if not fichiers:
        raise SystemExit(f"aucun Parquet dans {INPUT}")

    frames = [pd.read_parquet(f) for f in fichiers]
    sales = pd.concat(frames, ignore_index=True)
    sales["date"] = pd.to_datetime(sales["date"])
    print(
        f"[preprocess] entrée : {len(sales):,} lignes · "
        f"{len(fichiers)} fichier(s) · {sales['store_id'].nunique()} magasins"
    )

    _load_package()
    from retail_forecast.features.build import build_features

    base = build_features(sales, horizon=1, dropna=False)
    cols = sorted(c for c in base.columns if c.startswith(("lag_", "roll_")))
    print(
        f"[preprocess] features calculées : {len(base):,} lignes · "
        f"{len(cols)} colonnes lag/roll"
    )

    dest = OUTPUT / "features"
    dest.mkdir(parents=True, exist_ok=True)
    sortie = dest / "features.parquet"
    base.to_parquet(sortie, index=False, compression="snappy")

    rapport = {
        "rows": int(len(base)),
        "columns": int(base.shape[1]),
        "feature_columns": cols,
        "date_min": str(base["date"].min().date()),
        "date_max": str(base["date"].max().date()),
        "stores": int(base["store_id"].nunique()),
        "items": int(base["item_id"].nunique()),
        "source_files": len(fichiers),
    }
    # Écrit DANS le dossier de sortie : le fichier est ainsi déposé dans S3 avec
    # la table, ce qui permet de contrôler une exécution depuis la console. Le
    # canal d'entraînement ne lit que les ``*.parquet`` : ce JSON est ignoré.
    (dest / "features.json").write_text(json.dumps(rapport, indent=2))

    print(f"[preprocess] écrit {sortie.name} ({sortie.stat().st_size / 1e6:.1f} Mo)")
    print(
        "[preprocess] resume "
        + json.dumps(
            {
                "rows": rapport["rows"],
                "features": len(cols),
                "periode": f"{rapport['date_min']} -> {rapport['date_max']}",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
