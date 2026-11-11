"""Chargement d'un jeu de données quelconque vers le schéma canonique du projet.

Le pipeline attend une table de ventes avec ces colonnes :

    date, store_id, item_id, units, unit_price, promo, is_holiday,
    lead_time_days, stock_on_hand, stockout, category

Ce module accepte **n'importe quelle source** — CSV, CSV gzip, Parquet, JSON
Lines — en local (`data/raw/x.csv`) ou dans S3 (`s3://bucket/key`), et la ramène
au schéma attendu :

1. renommage automatique via des alias connus (`store_nbr` → `store_id`,
   `sales` → `units`, `ds` → `date`, …) ;
2. renommage explicite avec ``--mapping`` pour les cas particuliers ;
3. colonnes optionnelles absentes -> valeur par défaut ;
4. typage coercitif, lignes sans date / magasin / article / ventes supprimées.

Usage :

    # inspecter un dataset sans rien écrire
    python -m retail_forecast.data.load --source data/raw/m5.csv

    # convertir et déposer dans S3, prêt pour l'ingestion
    python -m retail_forecast.data.load --source data/raw/m5.csv \
        --out s3://bucket/raw/historique/sales.parquet

    # mapping explicite quand les alias ne suffisent pas
    python -m retail_forecast.data.load --source x.parquet \
        --mapping "Store=store_id,Item=item_id,Qty=units,Date=date"

Ensuite, l'ingestion est identique : `producer.py --data <fichier ou s3://…>`.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd

from retail_forecast.config import SETTINGS

CANONICAL = [
    "date",
    "store_id",
    "item_id",
    "category",
    "unit_price",
    "promo",
    "is_holiday",
    "lead_time_days",
    "stock_on_hand",
    "stockout",
    "units",
]

REQUIRED = ["date", "store_id", "item_id", "units"]

DEFAULTS = {
    "category": "unknown",
    "unit_price": 0.0,
    "promo": 0,
    "is_holiday": 0,
    "lead_time_days": 7,
    "stock_on_hand": 0,
    "stockout": 0,
}

# Alias fréquents (comparés en minuscules) : Kaggle, ERP, exports de caisse.
ALIASES = {
    "date": ["ds", "day", "vente_date", "date_vente", "transaction_date", "datum"],
    "store_id": ["store", "store_nbr", "store_number", "magasin", "shop", "site", "id_magasin"],
    "item_id": ["item", "item_nbr", "sku", "article", "product", "product_id", "id_article"],
    "units": ["sales", "qty", "quantity", "demand", "quantite", "quantité", "sold", "volume"],
    "unit_price": ["price", "sell_price", "unitprice", "prix", "prix_unitaire", "amount"],
    "promo": ["onpromotion", "promotion", "is_promo", "en_promo", "discount"],
    "is_holiday": ["holiday", "ferie", "férié", "is_holiday_flag"],
    "category": ["family", "dept", "department", "categorie", "catégorie", "segment"],
    "lead_time_days": ["lead_time", "leadtime", "delai_livraison", "délai"],
    "stock_on_hand": ["stock", "inventory", "on_hand", "qty_on_hand"],
    "stockout": ["stock_out", "rupture", "out_of_stock"],
}

INT_COLS = ["units", "promo", "is_holiday", "lead_time_days", "stock_on_hand", "stockout"]


def _s3_bytes(uri: str) -> bytes:
    """Lit un objet S3 (unique porte boto3 : ``ingest.storage.s3_client``)."""
    from retail_forecast.ingest.storage import s3_client

    rest = uri[len("s3://") :]
    bucket, _, key = rest.partition("/")
    body = s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    return body


def _frame_from_bytes(data: bytes, name: str) -> pd.DataFrame:
    """Devine le format à partir de l'extension du fichier."""
    low = name.lower()
    if low.endswith((".parquet", ".pq")):
        return pd.read_parquet(io.BytesIO(data))
    if low.endswith(".jsonl"):
        return pd.read_json(io.BytesIO(data), lines=True)
    if low.endswith(".json"):
        return pd.read_json(io.BytesIO(data))
    compression = "gzip" if low.endswith(".gz") else "infer"
    return pd.read_csv(io.BytesIO(data), compression=compression, low_memory=False)


def read_any(source: str) -> pd.DataFrame:
    """Lit un CSV/Parquet/JSON depuis un chemin local ou une URI ``s3://``."""
    if source.startswith("s3://"):
        return _frame_from_bytes(_s3_bytes(source), source)
    path = Path(source)
    if not path.exists():
        raise SystemExit(f"{source} introuvable")
    return _frame_from_bytes(path.read_bytes(), str(path))


def parse_mapping(spec: str | None) -> dict[str, str]:
    """``"Store=store_id,Qty=units"`` -> ``{"Store": "store_id", "Qty": "units"}``."""
    if not spec:
        return {}
    out = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise SystemExit(f"mapping invalide « {pair} » — attendu source=cible")
        src, _, dst = pair.partition("=")
        out[src.strip()] = dst.strip()
    return out


def normalize(df: pd.DataFrame, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    """Ramène un DataFrame au schéma canonique (renommage, défauts, typage)."""
    mapping = mapping or {}
    rename: dict[str, str] = {}

    by_lower = {str(c).lower().strip(): c for c in df.columns}
    for canon in CANONICAL:
        if canon in df.columns:
            continue
        for alias in [canon, *ALIASES.get(canon, [])]:
            src = by_lower.get(alias.lower())
            if src is not None and src not in rename:
                rename[src] = canon
                break

    # Le mapping explicite gagne toujours.
    for src, dst in mapping.items():
        if src in df.columns:
            rename[src] = dst

    out = df.rename(columns=rename)

    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise SystemExit(
            "colonnes obligatoires absentes : "
            + ", ".join(missing)
            + f"\ncolonnes trouvées : {list(df.columns)[:15]}"
            + "\nutiliser --mapping source=cible pour les préciser"
        )

    for col, value in DEFAULTS.items():
        if col not in out.columns:
            out[col] = value

    out = out.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date", "store_id", "item_id", "units"])

    for col in INT_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("int32")
    out["unit_price"] = (
        pd.to_numeric(out["unit_price"], errors="coerce").fillna(0.0).astype("float32")
    )
    for col in ("store_id", "item_id", "category"):
        out[col] = out[col].astype(str)

    out = out.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    return out[CANONICAL]


def load_dataset(source: str, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    """Lecture + normalisation en une étape (utilisé par le producteur)."""
    return normalize(read_any(source), mapping)


def write_any(df: pd.DataFrame, dest: str) -> int:
    """Écrit le DataFrame en Parquet vers un chemin local ou une URI ``s3://``."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="snappy")
    data = buf.getvalue()

    if dest.startswith("s3://"):
        from retail_forecast.ingest.storage import s3_client

        rest = dest[len("s3://") :]
        bucket, _, key = rest.partition("/")
        s3_client().put_object(Bucket=bucket, Key=key, Body=data)
    else:
        path = Path(dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return len(data)


def summary(df: pd.DataFrame) -> str:
    """Résumé lisible d'un dataset normalisé."""
    return (
        f"lignes    : {len(df):,}\n"
        f"période   : {df.date.min().date()} -> {df.date.max().date()}\n"
        f"magasins  : {df.store_id.nunique():,}\n"
        f"articles  : {df.item_id.nunique():,}\n"
        f"séries    : {df.groupby(['store_id', 'item_id']).ngroups:,}\n"
        f"ventes    : {int(df.units.sum()):,} unités"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Charge un dataset quelconque (CSV/Parquet/JSON, local ou S3) "
        "et le ramène au schéma du projet."
    )
    parser.add_argument("--source", required=True, help="fichier local ou s3://…")
    parser.add_argument("--out", default=None, help="parquet de sortie (local ou s3://…)")
    parser.add_argument("--mapping", default=None, help="source=cible,source=cible")
    parser.add_argument("--limit", type=int, default=None, help="nb max de lignes")
    args = parser.parse_args(argv)

    raw = read_any(args.source)
    print(f"[load] {args.source} : {len(raw):,} lignes · {len(raw.columns)} colonnes")
    print(f"[load] colonnes : {list(raw.columns)[:12]}")

    df = normalize(raw, parse_mapping(args.mapping))
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)

    print(f"[load] schéma canonique appliqué\n{summary(df)}")

    if args.out:
        size = write_any(df, args.out)
        print(f"[load] écrit {args.out} ({size / 1e6:.2f} Mo)")
        print(f"\nIngestion :\n  python -m retail_forecast.ingest.producer --data {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
