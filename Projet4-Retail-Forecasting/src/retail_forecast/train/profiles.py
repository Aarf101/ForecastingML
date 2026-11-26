"""Contrat versionné des modèles, conservé dans chaque artefact."""

from retail_forecast.features.target_calendar import add_target_calendar

DEFAULT_PROFILE = "median_calendar_v1"
LEGACY_PROFILE = "legacy_l2"
PROFILES = (DEFAULT_PROFILE, LEGACY_PROFILE)


def profile_params(profile):
    """Paramètres du point central ; les intervalles ont leur propre alpha."""
    if profile == DEFAULT_PROFILE:
        return {"objective": "quantile", "alpha": 0.5, "metric": "l1",
                "deterministic": True, "force_col_wise": True}
    if profile == LEGACY_PROFILE:
        return {"objective": "regression", "metric": "l1"}
    raise ValueError(f"profil de modèle inconnu : {profile}")


def target_features(frame, profile):
    """Recalcule le calendrier pour chaque horizon, jamais depuis une cible observée."""
    profile_params(profile)
    return add_target_calendar(frame) if profile == DEFAULT_PROFILE else frame
