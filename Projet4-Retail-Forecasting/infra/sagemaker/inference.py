"""Handler d'inférence du modèle (Batch Transform et endpoint).

SageMaker appelle ``model_fn`` puis ``predict_fn``. Un modèle **par horizon**
est stocké dans l'artefact : ``model_1.txt``, ``model_7.txt``, etc. L'entrée
d'inférence porte donc une colonne ``horizon`` qui choisit le bon booster.

Format d'entrée (une ligne JSON par série × horizon) ::

    {"store_id": "S01", "item_id": "I001", "horizon": 7, "lag_7": 4.0, ...}

La sortie est la liste des quantités prédites, dans l'ordre des lignes reçues.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_HORIZON = 1


def input_fn(input_data, content_type: str):
    """Décode l'entrée : JSONL (plusieurs objets par requête) accepté aussi.

    Le conteneur SKLearn par défaut fait ``json.loads`` sur tout le corps, ce qui
    échoue quand SageMaker regroupe plusieurs lignes (``Extra data``). On gère
    donc explicitement le cas « plusieurs JSON séparés par des retours ligne ».
    """
    import io

    if isinstance(input_data, bytes):
        input_data = input_data.decode("utf-8")
    if "json" in content_type:
        try:
            return json.loads(input_data)
        except json.JSONDecodeError:
            return [json.loads(line) for line in input_data.splitlines() if line.strip()]
    if "csv" in content_type:
        return pd.read_csv(io.StringIO(input_data))
    raise ValueError(f"content_type non géré : {content_type}")


def model_fn(model_dir: str) -> dict:
    """Charge tous les boosters ``model_<h>.txt`` et la liste des features."""
    from pathlib import Path

    root = Path(model_dir)
    models = {
        int(p.stem.split("_")[1]): lgb.Booster(model_file=str(p))
        for p in sorted(root.glob("model_*.txt"))
    }
    if not models:
        raise RuntimeError(f"aucun model_<h>.txt dans {root}")
    metadata = json.loads((root / "metadata.json").read_text())
    print(f"[inference] horizons chargés : {sorted(models)}")
    return {"models": models, "features": metadata["features"]}


def _as_frame(data, features: list[str]) -> pd.DataFrame:
    """Normalise les formats d'entrée du conteneur (dict, liste de dicts, tableau)."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, dict):
        # un seul enregistrement, ou dict de colonnes {"lag_1": [..], ...}
        if all(not isinstance(v, (list, tuple)) for v in data.values()):
            return pd.DataFrame([data])
        return pd.DataFrame(data)
    if isinstance(data, np.ndarray):
        return pd.DataFrame(
            data.reshape(1, -1) if data.ndim == 1 else data,
            columns=(features + ["horizon"])[: data.shape[-1]],
        )
    return pd.DataFrame(data)


def predict_fn(data, model: dict) -> list[float]:
    """Prédit par groupe d'horizon, dans l'ordre des lignes d'entrée."""
    frame = _as_frame(data, model["features"])
    features = model["features"]

    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise ValueError(f"features manquantes dans l'entrée : {missing[:5]}")

    horizons = (
        frame["horizon"].astype(int).to_numpy()
        if "horizon" in frame.columns
        else np.full(len(frame), DEFAULT_HORIZON)
    )
    out = np.full(len(frame), np.nan, dtype=float)
    for horizon, booster in model["models"].items():
        mask = horizons == horizon
        if mask.any():
            pred = booster.predict(frame.loc[mask, features])
            out[mask] = np.clip(pred, 0.0, None)
    if np.isnan(out).any():
        unknown = sorted(set(horizons[np.isnan(out)]))
        raise ValueError(f"aucun modèle pour les horizons {unknown}")
    return out.tolist()
