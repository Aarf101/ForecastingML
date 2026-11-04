"""Feature engineering pour la prévision de la demande.

Règles anti-fuite appliquées ici (question classique en soutenance) :

* on modélise en **direct multi-horizon** : pour une origine ``t``, la cible est
  ``units[t + horizon]`` et toutes les features n'utilisent que l'information
  disponible **jusqu'à t inclus** (lags, fenêtres glissantes, calendrier,
  promotions planifiées) ;
* ``stock_on_hand`` et ``stockout`` sont **exclus** des features : ce sont des
  conséquences de la demande (censoring), donc une fuite ;
* la séparation train/valid/test est **temporelle**, jamais aléatoire.

Le même module tourne en local (pandas), dans un SageMaker Processing Job et
dans un job Glue — c'est ce qui garantit la cohérence du pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["store_id", "item_id"]
TARGET = "y"
LAGS = (1, 7, 14, 21, 28, 35)
ROLLS = (7, 14, 28)

REQUIRED_COLUMNS = (
    "date",
    "store_id",
    "item_id",
    "category",
    "unit_price",
    "promo",
    "is_holiday",
    "lead_time_days",
    "units",
)

# Colonnes volontairement non prédictives (identifiants, cible, fuites).
EXCLUDED = {
    "date",
    "store_id",
    "item_id",
    "category",
    "units",
    TARGET,
    "stock_on_hand",
    "stockout",
}


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"colonnes manquantes : {missing}")
    if df.duplicated(subset=[*KEYS, "date"]).any():
        raise ValueError("doublons (store_id, item_id, date) : dédupliquer avant features")


def _units_group(df: pd.DataFrame):
    return df.groupby(KEYS, sort=False, observed=True)["units"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendrier + encodage des catégorielles (features disponibles à t)."""
    out = df.copy()
    out["store_code"] = out["store_id"].astype("category").cat.codes.astype("int16")
    out["item_code"] = out["item_id"].astype("category").cat.codes.astype("int16")
    out["category_code"] = out["category"].astype("category").cat.codes.astype("int8")

    d = out["date"].dt
    out["dow"] = d.dayofweek.astype("int8")
    out["day"] = d.day.astype("int8")
    out["month"] = d.month.astype("int8")
    out["week"] = d.isocalendar().week.astype("int8")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    out["is_holiday"] = out["is_holiday"].astype("int8")
    out["promo"] = out["promo"].astype("int8")

    out["log_price"] = np.log1p(out["unit_price"]).astype("float32")
    out["price_x_promo"] = (out["unit_price"] * out["promo"]).astype("float32")
    return out


def add_lag_features(
    df: pd.DataFrame, lags: tuple[int, ...] = LAGS, rolls: tuple[int, ...] = ROLLS
) -> pd.DataFrame:
    """Lags et fenêtres glissantes par série (magasin, article)."""
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = _units_group(out).shift(lag)
    for w in rolls:
        min_periods = max(2, w // 2)
        out[f"roll_mean_{w}"] = _units_group(out).transform(
            lambda s, w=w, mp=min_periods: s.rolling(w, min_periods=mp).mean()
        )
        out[f"roll_std_{w}"] = _units_group(out).transform(
            lambda s, w=w, mp=min_periods: s.rolling(w, min_periods=mp).std()
        )
    out["roll_max_14"] = _units_group(out).transform(
        lambda s: s.rolling(14, min_periods=7).max()
    )
    return out


def add_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Cible directe : ventes à ``t + horizon`` pour chaque série."""
    if horizon < 1:
        raise ValueError("horizon doit être >= 1")
    out = df.copy()
    out[TARGET] = _units_group(out).shift(-horizon)
    return out


def build_features(
    df: pd.DataFrame,
    horizon: int = 1,
    lags: tuple[int, ...] = LAGS,
    rolls: tuple[int, ...] = ROLLS,
    dropna: bool = True,
) -> pd.DataFrame:
    """Pipeline complet de features (calendrier -> lags -> cible -> dropna)."""
    _validate(df)
    out = df.sort_values([*KEYS, "date"]).reset_index(drop=True)
    out = add_calendar_features(out)
    out = add_lag_features(out, lags=lags, rolls=rolls)
    out = add_target(out, horizon=horizon)
    if dropna:
        feat_cols = [c for c in out.columns if c.startswith(("lag_", "roll_"))]
        out = out.dropna(subset=[*feat_cols, TARGET]).reset_index(drop=True)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Colonnes numériques utilisables comme features (hors cible / fuites)."""
    return [
        c
        for c in df.columns
        if c not in EXCLUDED and pd.api.types.is_numeric_dtype(df[c])
    ]


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare features / cible."""
    return df[feature_columns(df)], df[TARGET]


def time_split(
    df: pd.DataFrame, valid_days: int = 56, test_days: int = 28
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpe temporelle train / valid / test (jamais de shuffle)."""
    if valid_days <= 0 or test_days <= 0:
        raise ValueError("valid_days et test_days doivent être > 0")
    dates = np.sort(df["date"].unique())
    if len(dates) <= valid_days + test_days:
        raise ValueError(
            f"historique trop court : {len(dates)} jours pour {valid_days}+{test_days}"
        )
    test_start = dates[-test_days]
    valid_start = dates[-(test_days + valid_days)]

    train = df[df["date"] < valid_start]
    valid = df[(df["date"] >= valid_start) & (df["date"] < test_start)]
    test = df[df["date"] >= test_start]
    return (
        train.reset_index(drop=True),
        valid.reset_index(drop=True),
        test.reset_index(drop=True),
    )
