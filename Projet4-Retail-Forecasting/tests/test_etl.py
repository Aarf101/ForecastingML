"""Tests de l'ETL raw -> curated (déduplication et typage, sans S3)."""

import pandas as pd
import pytest

from retail_forecast.etl.build_curated import clean


def _raw(rows):
    return pd.DataFrame(rows)


def test_clean_deduplicates_keeping_last_event():
    raw = _raw(
        [
            {
                "date": "2024-01-01",
                "store_id": "S01",
                "item_id": "I001",
                "category": "food",
                "unit_price": 10.0,
                "promo": 0,
                "is_holiday": 0,
                "lead_time_days": 3,
                "units": 5,
                "event_time": "2024-01-02T10:00:00Z",
            },
            {
                "date": "2024-01-01",
                "store_id": "S01",
                "item_id": "I001",
                "category": "food",
                "unit_price": 10.0,
                "promo": 0,
                "is_holiday": 0,
                "lead_time_days": 3,
                "units": 7,  # rejoué : c'est cette valeur qui doit rester
                "event_time": "2024-01-02T11:00:00Z",
            },
        ]
    )
    out = clean(raw)
    assert len(out) == 1
    assert out["units"].iloc[0] == 7


def test_clean_casts_types_and_normalises_date():
    raw = _raw(
        [
            {
                "date": "2024-03-05 14:32:00",
                "store_id": "S02",
                "item_id": "I009",
                "units": 3,
            }
        ]
    )
    out = clean(raw)
    assert out["date"].iloc[0] == pd.Timestamp("2024-03-05")
    assert out["store_id"].dtype == "string"
    assert str(out["units"].dtype) == "int32"


def test_clean_sorts_by_series_and_date():
    raw = _raw(
        [
            {"date": "2024-01-02", "store_id": "S02", "item_id": "I001", "units": 1},
            {"date": "2024-01-01", "store_id": "S02", "item_id": "I001", "units": 2},
        ]
    )
    out = clean(raw)
    assert out["date"].is_monotonic_increasing


def test_clean_rejects_missing_columns():
    with pytest.raises(ValueError, match="colonnes manquantes"):
        clean(_raw([{"date": "2024-01-01", "store_id": "S01"}]))
