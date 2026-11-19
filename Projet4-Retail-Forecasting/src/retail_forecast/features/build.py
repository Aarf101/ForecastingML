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
    "target_date",
    "seasonal_naive",
    "stock_on_hand",
    "stockout",
}


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"colonnes manquantes : {missing}")
    if df.duplicated(subset=[*KEYS, "date"]).any():
        raise ValueError("doublons (store_id, item_id, date) : dédupliquer avant features")
    if df["date"].isna().any() or not df["date"].eq(df["date"].dt.normalize()).all():
        raise ValueError("dates journalières valides requises")


def _units_group(df: pd.DataFrame):
    return df.groupby(KEYS, sort=False, observed=True)["units"]


def active_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Isole le catalogue courant sans mélanger des historiques abandonnés.

    Une série vue dans la dernière semaine mais absente du dernier jour est
    une livraison incomplète potentielle : on échoue plutôt que la supprimer.
    """
    if df.empty:
        raise ValueError("historique vide")
    last = df["date"].max()
    current = df.loc[df["date"] == last, KEYS].drop_duplicates()
    recent = df.loc[df["date"] > last - pd.Timedelta(days=7), KEYS].drop_duplicates()
    if len(recent) != len(current):
        raise ValueError("catalogue récent incomplet à la dernière date : vérifier l'ingestion")
    return df.merge(current, on=KEYS, how="inner")


def categorical_levels(df: pd.DataFrame) -> dict[str, list]:
    """Fige le vocabulaire des identifiants pour l'artefact d'inférence."""
    return {c: df[c].astype("category").cat.categories.tolist()
            for c in [*KEYS, "category"]}


def add_calendar_features(
    df: pd.DataFrame, levels: dict[str, list] | None = None
) -> pd.DataFrame:
    """Calendrier + encodage des catégorielles (features disponibles à t)."""
    out = df.copy()
    levels = categorical_levels(out) if levels is None else levels
    for source, dest in [("store_id", "store_code"), ("item_id", "item_code"),
                         ("category", "category_code")]:
        out[dest] = pd.Categorical(out[source], categories=levels[source]).codes.astype("int32")

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
    values = out.set_index([*KEYS, "date"])["units"]
    for lag in lags:
        index = pd.MultiIndex.from_arrays(
            [out[k] for k in KEYS] + [out["date"] - pd.Timedelta(days=lag)]
        )
        out[f"lag_{lag}"] = values.reindex(index).to_numpy()

    def rolling(s, window, minimum, statistic):
        # Fenêtre en jours réels ; aucune vente n'est inventée pour les trous.
        dated = pd.Series(s.to_numpy(), index=out.loc[s.index, "date"])
        return getattr(dated.rolling(f"{window}D", min_periods=minimum), statistic)().to_numpy()

    for w in rolls:
        min_periods = max(2, w // 2)
        out[f"roll_mean_{w}"] = _units_group(out).transform(
            lambda s, w=w, mp=min_periods: rolling(s, w, mp, "mean")
        )
        out[f"roll_std_{w}"] = _units_group(out).transform(
            lambda s, w=w, mp=min_periods: rolling(s, w, mp, "std")
        )
    out["roll_max_14"] = _units_group(out).transform(
        lambda s: rolling(s, 14, 7, "max")
    )
    return out


def add_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Cible directe : ventes à ``t + horizon`` pour chaque série."""
    if horizon < 1:
        raise ValueError("horizon doit être >= 1")
    out = df.copy()
    out["target_date"] = out["date"] + pd.Timedelta(days=horizon)
    for name, values in [("target_dow", out["target_date"].dt.dayofweek),
                         ("target_month", out["target_date"].dt.month),
                         ("target_day", out["target_date"].dt.day)]:
        out[name] = values.astype("int8")
    # Recherche par date réelle : l'ordre des partitions Parquet est indifférent.
    values = out.set_index([*KEYS, "date"])["units"]

    def lookup(dates):
        index = pd.MultiIndex.from_arrays([out[k] for k in KEYS] + [dates])
        return values.reindex(index).to_numpy()

    out[TARGET] = lookup(out["target_date"])
    # Même jour de semaine que la cible, dernière observation connue à l'origine.
    seasonal_date = out["target_date"] - pd.Timedelta(days=7 * ((horizon + 6) // 7))
    out["seasonal_naive"] = lookup(seasonal_date)
    return out


def build_features(
    df: pd.DataFrame,
    horizon: int = 1,
    lags: tuple[int, ...] = LAGS,
    rolls: tuple[int, ...] = ROLLS,
    dropna: bool = True,
    levels: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Pipeline complet de features (calendrier -> lags -> cible -> dropna)."""
    _validate(df)
    out = df.sort_values([*KEYS, "date"]).reset_index(drop=True)
    out = add_calendar_features(out, levels=levels)
    out = add_lag_features(out, lags=lags, rolls=rolls)
    out = add_target(out, horizon=horizon)
    if dropna:
        feat_cols = [c for c in out.columns if c.startswith(("lag_", "roll_"))]
        out = out.dropna(subset=[*feat_cols, TARGET, "seasonal_naive"]).reset_index(drop=True)
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
    if not len(dates) or (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days < valid_days + test_days:
        raise ValueError(
            f"historique trop court : {len(dates)} jours pour {valid_days}+{test_days}"
        )
    test_start = pd.Timestamp(dates[-1]) - pd.Timedelta(days=test_days - 1)
    valid_start = test_start - pd.Timedelta(days=valid_days)

    # Une cible d'entraînement ne doit pas appartenir à la période suivante.
    target_dates = df["target_date"] if "target_date" in df else df["date"]
    train = df[(df["date"] < valid_start) & (target_dates < valid_start)]
    valid = df[(df["date"] >= valid_start) & (df["date"] < test_start)
               & (target_dates < test_start)]
    test = df[df["date"] >= test_start]
    if train.empty or valid.empty or test.empty:
        raise ValueError("historique trop court après purge des cibles aux frontières")
    return (
        train.reset_index(drop=True),
        valid.reset_index(drop=True),
        test.reset_index(drop=True),
    )
