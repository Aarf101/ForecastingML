"""Calendrier connu et sélection sans lecture du test final."""

import numpy as np
import pandas as pd
import pytest

from retail_forecast.evaluate.benchmark_median import VARIANTS, select_candidate, temporal_parts
from retail_forecast.features.target_calendar import add_target_calendar


def test_calendar_uses_only_target_date():
    dates = pd.to_datetime(["2024-12-25", "2024-12-28", "2025-01-01"])
    expected = add_target_calendar(pd.DataFrame({"target_date": dates}))
    hostile = add_target_calendar(pd.DataFrame({"target_date": dates, "promo": 999,
                                               "is_holiday": 0, "units": -999}))
    pd.testing.assert_frame_equal(expected, hostile[expected.columns])
    assert expected.target_is_holiday.tolist() == [1, 0, 1]
    assert expected.target_is_weekend.tolist() == [0, 1, 0]
    assert expected.target_is_month_start.tolist() == [0, 0, 1]
    assert np.isfinite(expected.filter(regex="sin|cos")).all().all()


def test_split_purges_labels_and_reserves_common_test_dates():
    dates = pd.date_range("2024-01-01", periods=150)
    for horizon in (1, 16):
        frame = pd.DataFrame({"date": dates, "target_date": dates + pd.Timedelta(days=horizon)})
        frame = frame[frame.target_date <= dates.max()]
        valid_start, test_start = dates[60], dates[120]
        train, valid, test = temporal_parts(frame, valid_start, test_start)
        assert train.target_date.max() < valid.date.min()
        assert valid.target_date.max() < test.target_date.min()
        assert test.target_date.min() == test_start
        assert test.target_date.max() == dates.max()


def test_selection_ignores_test_scores_and_requires_all_horizons():
    rows = [{"variant": name, "horizon": horizon,
             "valid_wape": 0.3 if name == "l1_calendar" else 0.4,
             "test_wape": 1.0 if name == "l1_calendar" else 0.0}
            for name in VARIANTS for horizon in range(1, 17)]
    assert select_candidate(rows) == "l1_calendar"
    with pytest.raises(ValueError, match="incomplète"):
        select_candidate(rows[:-1])
