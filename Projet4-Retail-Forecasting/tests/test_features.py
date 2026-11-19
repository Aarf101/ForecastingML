"""Tests du feature engineering : alignement des lags, absence de fuite, split."""

import numpy as np
import pandas as pd
import pytest

from retail_forecast.data.simulate import simulate_sales
from retail_forecast.features.build import (
    TARGET,
    build_features,
    feature_columns,
    time_split,
    xy,
)


@pytest.fixture(scope="module")
def small() -> pd.DataFrame:
    return simulate_sales(n_stores=2, n_items=3, days=45, seed=7)


def _series(df: pd.DataFrame, store: str = "S01", item: str = "I001") -> pd.DataFrame:
    return (
        df[(df.store_id == store) & (df.item_id == item)]
        .sort_values("date")
        .reset_index(drop=True)
    )


def test_lag_alignment(small):
    feats = build_features(small, horizon=1, lags=(1, 7), rolls=(7,), dropna=False)
    raw = _series(small)
    got = _series(feats)

    assert np.isnan(got.loc[0, "lag_1"])
    assert got.loc[1, "lag_1"] == raw.loc[0, "units"]
    assert got.loc[8, "lag_7"] == raw.loc[1, "units"]


def test_target_is_future_units(small):
    feats = build_features(small, horizon=1, lags=(1,), rolls=(7,), dropna=False)
    raw = _series(small)
    got = _series(feats)

    assert got.loc[0, TARGET] == raw.loc[1, "units"]
    assert np.isnan(got.loc[len(got) - 1, TARGET])


def test_horizon_shifts_target(small):
    h1 = _series(build_features(small, horizon=1, dropna=False))
    h7 = _series(build_features(small, horizon=7, dropna=False))
    raw = _series(small)
    assert h7.loc[0, TARGET] == raw.loc[7, "units"]
    assert h7[TARGET].iloc[0] != h1[TARGET].iloc[0]


def test_rolling_mean_is_past_only(small):
    feats = build_features(small, horizon=1, lags=(1,), rolls=(7,), dropna=False)
    raw = _series(small)
    got = _series(feats)
    # à t=10, roll_mean_7 couvre units[4..10] inclus (info connue à t)
    expected = raw.loc[4:10, "units"].mean()
    assert got.loc[10, "roll_mean_7"] == pytest.approx(expected)


def test_no_leaky_columns_in_features(small):
    feats = build_features(small, horizon=1)
    cols = feature_columns(feats)
    for forbidden in ("stock_on_hand", "stockout", "units", TARGET, "date"):
        assert forbidden not in cols
    assert "lag_1" in cols and "roll_mean_28" in cols


def test_dropna_removes_warmup_rows_only(small):
    with_nan = build_features(small, horizon=1, dropna=False)
    clean = build_features(small, horizon=1, dropna=True)
    assert 0 < len(clean) < len(with_nan)
    assert not clean[feature_columns(clean)].isna().any().any()
    assert not clean[TARGET].isna().any()


def test_xy_shapes(small):
    feats = build_features(small, horizon=1)
    X, y = xy(feats)
    assert len(X) == len(y) == len(feats)
    assert TARGET not in X.columns


def test_time_split_is_chronological(small):
    feats = build_features(small, horizon=1)
    train, valid, test = time_split(feats, valid_days=4, test_days=3)
    assert len(train) and len(valid) and len(test)
    assert train["date"].max() < valid["date"].min()
    assert valid["date"].max() < test["date"].min()
    assert test["date"].nunique() == 3


def test_duplicates_are_rejected(small):
    dup = pd.concat([small, small.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="doublons"):
        build_features(dup, horizon=1)


def test_missing_columns_are_rejected(small):
    with pytest.raises(ValueError, match="colonnes manquantes"):
        build_features(small.drop(columns=["promo"]), horizon=1)


def test_too_short_history_is_rejected(small):
    feats = build_features(small, horizon=1)
    with pytest.raises(ValueError, match="historique trop court"):
        time_split(feats, valid_days=40, test_days=30)


@pytest.mark.parametrize("horizon", [1, 7, 28])
def test_split_purges_future_labels(horizon):
    sales = simulate_sales(n_stores=1, n_items=1, days=180, seed=7)
    feats = build_features(sales, horizon=horizon)
    train, valid, test = time_split(feats, valid_days=56, test_days=28)
    assert train.target_date.max() < valid.date.min()
    assert valid.target_date.max() < test.date.min()
    assert "seasonal_naive" not in feature_columns(feats)


@pytest.mark.parametrize("horizon,offset", [(1, -6), (7, 0), (8, -6), (28, 0)])
def test_seasonal_baseline_matches_target_weekday(small, horizon, offset):
    raw = _series(small)
    feats = _series(build_features(small, horizon=horizon, dropna=False))
    assert feats.loc[14, "seasonal_naive"] == raw.loc[14 + offset, "units"]
    assert feats.loc[14, "target_date"].dayofweek == raw.loc[14 + offset, "date"].dayofweek


def test_missing_day_is_not_silently_treated_as_one_day(small):
    incomplete = small[~((small.store_id == "S01") & (small.item_id == "I001")
                         & (small.date == small.date.min() + pd.Timedelta(days=10)))]
    feats = _series(build_features(incomplete, dropna=False))
    next_day = feats[feats.date == small.date.min() + pd.Timedelta(days=11)].iloc[0]
    assert pd.isna(next_day.lag_1)
    before_gap = feats[feats.date == small.date.min() + pd.Timedelta(days=9)].iloc[0]
    assert pd.isna(before_gap.y)


def test_target_lookup_ignores_partition_order(small):
    from retail_forecast.features.build import add_target

    shuffled = small.sample(frac=1, random_state=8)
    actual = add_target(shuffled, horizon=7).set_index(["store_id", "item_id", "date"])
    expected = build_features(small, horizon=7, dropna=False).set_index(
        ["store_id", "item_id", "date"]
    )
    pd.testing.assert_series_equal(actual.y.sort_index(), expected.y.sort_index())


def test_split_windows_are_calendar_days_with_gaps():
    sales = simulate_sales(n_stores=1, n_items=1, days=180, seed=7)
    missing = sales.date.max() - pd.Timedelta(days=2)
    sales = sales[sales.date != missing]
    feats = build_features(sales, horizon=1)
    train, valid, test = time_split(feats, valid_days=35, test_days=14)
    assert test.date.min() >= feats.date.max() - pd.Timedelta(days=13)
    assert train.target_date.max() < valid.date.min()
    assert valid.target_date.max() < test.date.min()


def test_active_catalog_excludes_unrelated_old_histories(small):
    from retail_forecast.features.build import active_catalog
    old = small.copy()
    old['date'] -= pd.Timedelta(days=365)
    old['store_id'] = 'legacy'
    mixed = pd.concat([small, old], ignore_index=True)
    current = active_catalog(mixed)
    assert set(current.store_id) == set(small.store_id)
    assert len(current) == len(small)


def test_active_catalog_rejects_partial_recent_delivery(small):
    from retail_forecast.features.build import active_catalog
    partial = small[~((small.date == small.date.max()) & (small.store_id == 'S01'))]
    with pytest.raises(ValueError, match='catalogue récent incomplet'):
        active_catalog(partial)
