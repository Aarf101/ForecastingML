"""Métriques de prévision de la demande.

WAPE est la métrique principale : elle reste interprétable quand certaines
séries ont des ventes nulles, contrairement au MAPE (qui explose ou est
indéfini). Le pinball loss sert aux prévisions quantiles (P10/P50/P90) utilisées
pour le stock de sécurité dans le dashboard.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_arrays(y_true: Iterable[float], y_pred: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    if t.shape != p.shape:
        raise ValueError(f"formes incompatibles : y_true={t.shape} vs y_pred={p.shape}")
    if t.size == 0:
        raise ValueError("tableaux vides")
    return t, p


def wape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Weighted Absolute Percentage Error = sum|y-yhat| / sum|y| (0 = parfait)."""
    t, p = _as_arrays(y_true, y_pred)
    denom = np.abs(t).sum()
    if denom == 0:
        return 0.0 if np.allclose(p, 0) else float("inf")
    return float(np.abs(t - p).sum() / denom)


def mape(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error, exprimé en fraction (0.12 = 12 %).

    Les lignes où ``y_true < eps`` sont ignorées : le MAPE n'y est pas défini.
    """
    t, p = _as_arrays(y_true, y_pred)
    mask = np.abs(t) > eps
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(t[mask] - p[mask]) / np.abs(t[mask])))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Root Mean Squared Error (même unité que la cible)."""
    t, p = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def pinball_loss(
    y_true: Iterable[float], y_pred: Iterable[float], quantile: float
) -> float:
    """Quantile (pinball) loss — plus petite = meilleure calibration.

    ``y_pred`` est le quantile prédit (ex. P90 -> quantile=0.9).
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile doit être dans ]0, 1[")
    t, p = _as_arrays(y_true, y_pred)
    diff = t - p
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def evaluate(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    prefix: str = "",
) -> dict[str, float]:
    """Toutes les métriques de régression sous forme de dict (log / MLflow)."""
    return {
        f"{prefix}wape": wape(y_true, y_pred),
        f"{prefix}mape": mape(y_true, y_pred),
        f"{prefix}rmse": rmse(y_true, y_pred),
    }
