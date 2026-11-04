"""Générateur de ventes synthétiques (plan B sans token Kaggle).

Le schéma produit est celui attendu par le reste du pipeline (ingestion Kafka,
ETL S3, features, LightGBM) : il est donc interchangeable avec un extrait
M5 / Rossmann nettoyé. La série contient de la saisonnalité hebdomadaire et
annuelle, des promotions, des jours fériés, une tendance, du bruit et des
ruptures de stock — de quoi justifier de vraies décisions de modélisation
(quantiles, stock de sécurité) dans le rapport.

Usage::

    python -m retail_forecast.data.simulate --out data/raw/sales.csv
    python -m retail_forecast.data.simulate --stores 5 --items 8 --days 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ("grocery", "beverages", "household", "personal_care", "electronics")
HOLIDAYS_MMDD = ("01-01", "05-01", "07-14", "11-11", "12-24", "12-25", "12-31")


def simulate_sales(
    n_stores: int = 20,
    n_items: int = 60,
    days: int = 730,
    start: str = "2023-01-01",
    seed: int = 42,
    promo_rate: float = 0.12,
) -> pd.DataFrame:
    """Simule un historique de ventes quotidiennes par (magasin, article)."""
    if min(n_stores, n_items, days) < 1:
        raise ValueError("n_stores, n_items et days doivent être >= 1")

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=days, freq="D")
    stores = [f"S{i:02d}" for i in range(1, n_stores + 1)]
    items = [f"I{i:03d}" for i in range(1, n_items + 1)]

    # --- structure magasins / articles --------------------------------------
    store_level = rng.lognormal(0.0, 0.35, n_stores)
    store_level /= store_level.mean()
    item_pop = rng.lognormal(0.0, 0.55, n_items)
    item_pop /= item_pop.mean()
    item_cat = rng.integers(0, len(CATEGORIES), n_items)
    list_price = np.round(rng.uniform(1.5, 45.0, n_items), 2)
    lead_time = rng.integers(2, 8, n_stores)

    # --- calendrier ---------------------------------------------------------
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()
    dow_factor = np.array([0.92, 0.90, 0.95, 1.00, 1.10, 1.28, 1.08])[dow]
    seasonal = (
        1.0
        + 0.18 * np.sin(2 * np.pi * (doy - 60) / 365.25)
        + 0.10 * np.cos(2 * np.pi * doy / 365.25)
    )
    is_holiday = np.isin(dates.strftime("%m-%d"), HOLIDAYS_MMDD)
    holiday_factor = np.where(is_holiday, 1.45, 1.0)

    # --- intensité attendue : (magasin, article, jour) ----------------------
    base = np.outer(store_level, item_pop) * 6.0  # ~6 unités/jour en moyenne
    mu = (
        base[:, :, None]
        * dow_factor[None, None, :]
        * seasonal[None, None, :]
        * holiday_factor[None, None, :]
    )

    promo = rng.random(mu.shape) < promo_rate
    promo_lift = np.where(promo, np.exp(rng.normal(0.30, 0.12, mu.shape)), 1.0)
    trend = np.linspace(0.90, 1.12, days)[None, None, :]
    noise = rng.gamma(shape=8.0, scale=1.0 / 8.0, size=mu.shape)  # moyenne 1

    lam = mu * promo_lift * trend * noise
    demand = rng.poisson(lam)

    # --- stock disponible : couverture de quelques jours, ruptures possibles
    # Couverture < lead time => ruptures fréquentes (5-10 %, réaliste en retail)
    coverage = lead_time[:, None, None] * rng.uniform(0.55, 1.15, mu.shape)
    stock = np.round(mu * coverage).astype(np.int32)
    stockout = demand > stock
    units = np.minimum(demand, stock).astype(np.int32)

    unit_price = np.round(
        np.where(promo, list_price[None, :, None] * 0.80, list_price[None, :, None]), 2
    )

    per_store = n_items * days
    df = pd.DataFrame(
        {
            "date": np.tile(dates.to_numpy(), n_stores * n_items),
            "store_id": np.repeat(stores, per_store),
            "item_id": np.tile(np.repeat(items, days), n_stores),
            "category": np.tile(
                np.repeat(np.asarray(CATEGORIES)[item_cat], days), n_stores
            ),
            "unit_price": unit_price.astype("float32").ravel(),
            "promo": promo.ravel().astype("int8"),
            "is_holiday": np.tile(is_holiday, n_stores * n_items).astype("int8"),
            "lead_time_days": np.repeat(lead_time, per_store).astype("int16"),
            "stock_on_hand": stock.ravel(),
            "stockout": stockout.ravel().astype("int8"),
            "units": units.ravel(),
        }
    )
    return df.sort_values(["store_id", "item_id", "date"], ignore_index=True)


def write_sales(df: pd.DataFrame, path: str | Path) -> Path:
    """Écrit le DataFrame en CSV ou Parquet selon l'extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"extension non supportée : {path.suffix} (csv|parquet)")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Génère un historique de ventes.")
    parser.add_argument("--stores", type=int, default=20)
    parser.add_argument("--items", type=int, default=60)
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/raw/sales.csv")
    args = parser.parse_args(argv)

    df = simulate_sales(
        n_stores=args.stores,
        n_items=args.items,
        days=args.days,
        start=args.start,
        seed=args.seed,
    )
    out = write_sales(df, args.out)
    print(f"[simulate] {len(df):,} lignes -> {out}")
    print(
        f"[simulate] {df.store_id.nunique()} magasins x {df.item_id.nunique()} articles"
        f" du {df.date.min():%Y-%m-%d} au {df.date.max():%Y-%m-%d}"
        f" | units moy={df.units.mean():.2f} | ruptures={df.stockout.mean():.1%}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
