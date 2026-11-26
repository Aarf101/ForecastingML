"""Calendrier cible optionnel, calculable avant toute observation future."""

import numpy as np
import pandas as pd

from retail_forecast.data.simulate import HOLIDAYS_MMDD


def add_target_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Enrichit la date cible avec le calendrier du jeu synthétique uniquement.

    Les jours fériés sont ceux du simulateur ; un autre jeu de données exige
    son propre calendrier métier. Aucun prix ni promotion futur n'est lu.
    """
    out = df.copy()
    dates = pd.to_datetime(out["target_date"])
    day = dates.dt.dayofyear
    out["target_is_weekend"] = (dates.dt.dayofweek >= 5).astype("int8")
    out["target_is_holiday"] = dates.dt.strftime("%m-%d").isin(HOLIDAYS_MMDD).astype("int8")
    out["target_is_month_start"] = dates.dt.is_month_start.astype("int8")
    out["target_is_month_end"] = dates.dt.is_month_end.astype("int8")
    for period in (1, 2):
        phase = 2 * np.pi * period * day / 365.25
        out[f"target_year_sin_{period}"] = np.sin(phase).astype("float32")
        out[f"target_year_cos_{period}"] = np.cos(phase).astype("float32")
    return out
