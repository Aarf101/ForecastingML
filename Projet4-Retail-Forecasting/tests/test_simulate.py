"""Tests du générateur de ventes synthétiques."""

import pandas as pd
import pytest

from retail_forecast.data.simulate import simulate_sales, write_sales

EXPECTED = {
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
}


def test_shape_and_columns():
    df = simulate_sales(n_stores=3, n_items=4, days=30, seed=1)
    assert len(df) == 3 * 4 * 30
    assert set(df.columns) == EXPECTED
    assert df.store_id.nunique() == 3
    assert df.item_id.nunique() == 4
    assert df.date.nunique() == 30


def test_units_respect_stock_and_are_non_negative():
    df = simulate_sales(n_stores=2, n_items=3, days=60, seed=2)
    assert (df.units >= 0).all()
    assert (df.units <= df.stock_on_hand).all()
    assert 0.0 <= df.stockout.mean() <= 1.0


def test_series_are_sorted_by_date():
    df = simulate_sales(n_stores=2, n_items=3, days=60, seed=3)
    ordered = df.groupby(["store_id", "item_id"])["date"].apply(
        lambda s: s.is_monotonic_increasing
    )
    assert ordered.all()


def test_same_seed_is_reproducible():
    a = simulate_sales(n_stores=2, n_items=2, days=30, seed=11)
    b = simulate_sales(n_stores=2, n_items=2, days=30, seed=11)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_values():
    a = simulate_sales(n_stores=2, n_items=2, days=30, seed=11)
    b = simulate_sales(n_stores=2, n_items=2, days=30, seed=12)
    assert not a["units"].equals(b["units"])


def test_promo_lowers_price():
    df = simulate_sales(n_stores=2, n_items=3, days=90, seed=5)
    assert (df.loc[df.promo == 1, "unit_price"].mean()
            < df.loc[df.promo == 0, "unit_price"].mean())


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        simulate_sales(n_stores=0)


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_write_roundtrip(tmp_path, suffix):
    df = simulate_sales(n_stores=2, n_items=2, days=20, seed=7)
    path = write_sales(df, tmp_path / f"sales{suffix}")
    assert path.exists()
    back = pd.read_csv(path) if suffix == ".csv" else pd.read_parquet(path)
    assert len(back) == len(df)
    assert set(back.columns) == EXPECTED


def test_write_rejects_unknown_extension(tmp_path):
    df = simulate_sales(n_stores=1, n_items=1, days=5, seed=8)
    with pytest.raises(ValueError):
        write_sales(df, tmp_path / "sales.txt")
