"""Tests du chargeur générique : n'importe quel dataset -> schéma canonique."""

from __future__ import annotations

import pandas as pd
import pytest

from retail_forecast.data import load


def _raw_m5_like() -> pd.DataFrame:
    """Ressemble à un export M5 / Rossmann : noms et types différents."""
    return pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Store": [1, 1, 1],
            "Item": [101, 101, 102],
            "Sales": [10, 0, 7],
            "onpromotion": [0, 1, 0],
        }
    )


# --- alias de colonnes -------------------------------------------------------


def test_alias_renames_to_canonical():
    out = load.normalize(_raw_m5_like())
    assert list(out.columns) == load.CANONICAL
    assert out.store_id.tolist() == ["1", "1", "1"]
    assert out.units.tolist() == [10, 0, 7]
    assert out.promo.tolist() == [0, 1, 0]


def test_date_parsed_as_datetime():
    out = load.normalize(_raw_m5_like())
    assert out.date.dtype.kind == "M"
    assert str(out.date.min().date()) == "2024-01-01"


def test_explicit_mapping_wins_over_alias():
    df = pd.DataFrame(
        {
            "jour": ["2024-02-01"],
            "mag": ["S09"],
            "ref": ["A1"],
            "qte": [5],
        }
    )
    out = load.normalize(
        df, {"jour": "date", "mag": "store_id", "ref": "item_id", "qte": "units"}
    )
    assert out.loc[0, "store_id"] == "S09"
    assert out.loc[0, "units"] == 5


# --- valeurs par défaut et typage -------------------------------------------


def test_missing_optional_columns_get_defaults():
    out = load.normalize(_raw_m5_like())
    assert out.category.iloc[0] == "unknown"
    assert out.unit_price.iloc[0] == 0.0
    assert out.lead_time_days.iloc[0] == 7
    assert out.stockout.iloc[0] == 0


def test_numeric_strings_are_coerced():
    df = _raw_m5_like()
    df["Sales"] = ["10", "0", "7"]
    out = load.normalize(df)
    assert out.units.dtype == "int32"
    assert out.units.sum() == 17


def test_invalid_dates_are_dropped():
    df = _raw_m5_like()
    df.loc[1, "Date"] = "pas une date"
    out = load.normalize(df)
    assert len(out) == 2
    assert "pas une date" not in out.date.astype(str).tolist()


def test_null_units_are_dropped():
    df = _raw_m5_like()
    df.loc[0, "Sales"] = None
    out = load.normalize(df)
    assert len(out) == 2


def test_rows_sorted_by_series_then_date():
    df = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-01", "2024-01-01"],
            "store_id": ["S02", "S01", "S02"],
            "item_id": ["I1", "I1", "I1"],
            "units": [1, 2, 3],
        }
    )
    out = load.normalize(df)
    assert out.store_id.tolist() == ["S01", "S02", "S02"]
    assert out.date.is_monotonic_increasing


# --- erreurs ----------------------------------------------------------------


def test_missing_required_column_raises():
    df = _raw_m5_like().drop(columns=["Sales"])
    with pytest.raises(SystemExit, match="colonnes obligatoires absentes"):
        load.normalize(df)


def test_parse_mapping():
    assert load.parse_mapping("Store=store_id,Qty=units") == {
        "Store": "store_id",
        "Qty": "units",
    }
    assert load.parse_mapping(None) == {}


def test_parse_mapping_rejects_malformed_pair():
    with pytest.raises(SystemExit, match="mapping invalide"):
        load.parse_mapping("Store")


# --- lecture / écriture ------------------------------------------------------


def test_read_any_csv_and_write_roundtrip(tmp_path):
    src = tmp_path / "m5.csv"
    _raw_m5_like().to_csv(src, index=False)

    df = load.load_dataset(str(src))
    assert len(df) == 3

    dest = tmp_path / "sub" / "out.parquet"
    size = load.write_any(df, str(dest))
    assert size > 0
    assert pd.read_parquet(dest).shape == (3, len(load.CANONICAL))


def test_summary_reports_counts():
    text = load.summary(load.normalize(_raw_m5_like()))
    assert "magasins  : 1" in text
    assert "articles  : 2" in text


# --- colonnes constantes (--set) --------------------------------------------


def test_constants_satisfy_missing_required_column():
    """Rossmann n'a pas d'identifiant article : --set item_id=TOTAL le fournit."""
    df = pd.DataFrame(
        {
            "Date": ["2015-07-31", "2015-08-01"],
            "Store": [1, 1],
            "Sales": [5263, 6064],
        }
    )
    out = load.normalize(df, {"Date": "date", "Store": "store_id", "Sales": "units"},
                         {"item_id": "TOTAL"})
    assert out.item_id.tolist() == ["TOTAL", "TOTAL"]
    assert out.units.tolist() == [5263, 6064]


def test_constants_override_existing_column():
    out = load.normalize(_raw_m5_like(), constants={"category": "forced"})
    assert set(out.category) == {"forced"}

