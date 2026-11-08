"""Tests des métriques (valeurs calculées à la main)."""

import math

import pytest

from retail_forecast.evaluate.metrics import bias, evaluate, mape, pinball_loss, rmse, wape


def test_wape_hand_computed():
    # |10-12| + |20-18| = 4 ; sum(y) = 30
    assert wape([10, 20], [12, 18]) == pytest.approx(4 / 30)


def test_wape_zero_demand():
    assert wape([0, 0], [0, 0]) == 0.0
    assert math.isinf(wape([0, 0], [1, 2]))


def test_mape_ignores_zero_targets():
    # seule la ligne y=10 compte : |10-12| / 10 = 0.2
    assert mape([0, 10], [5, 12]) == pytest.approx(0.2)


def test_rmse():
    assert rmse([0, 0], [2, -2]) == pytest.approx(2.0)


def test_pinball_median_is_half_mae():
    assert pinball_loss([10, 20], [12, 18], 0.5) == pytest.approx(0.5 * 2.0)


def test_pinball_penalises_under_forecast_more_at_p90():
    under = pinball_loss([10], [8], 0.9)  # stock trop bas : rupture
    over = pinball_loss([10], [12], 0.9)  # sur-stock : moins grave
    assert under > over


def test_bias_sign_and_value():
    # surestimation : (12-10) + (18-20) = 0 -> biais nul malgré les erreurs
    assert bias([10, 20], [12, 18]) == pytest.approx(0.0)
    # sous-estimation systématique : (8-10) + (16-20) = -6 ; sum(y) = 30
    assert bias([10, 20], [8, 16]) == pytest.approx(-6 / 30)
    # surestimation systématique
    assert bias([10, 20], [12, 24]) == pytest.approx(6 / 30)


def test_bias_zero_demand():
    assert bias([0, 0], [0, 0]) == 0.0
    assert math.isinf(bias([0, 0], [1, 2]))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        wape([1, 2], [1])


def test_empty_raises():
    with pytest.raises(ValueError):
        wape([], [])


def test_evaluate_keys_and_prefix():
    res = evaluate([1, 2], [1.1, 2.1], prefix="valid_")
    assert set(res) == {"valid_wape", "valid_mape", "valid_rmse", "valid_bias"}
    assert res["valid_wape"] == pytest.approx(0.2 / 3)
    assert res["valid_bias"] == pytest.approx(0.2 / 3)
