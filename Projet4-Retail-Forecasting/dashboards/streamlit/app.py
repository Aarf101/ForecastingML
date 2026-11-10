"""Dashboard Streamlit — prévision de la demande (rendu n°4 du PDF).

Lit les prévisions produites par le Batch Transform (``forecast/table/``) et
l'historique ``curated/``, en S3 si ``S3_BUCKET`` est configuré, sinon les
fichiers locaux. Aucune écriture : c'est un outil de lecture.

Lancement::

    make dashboard            # ou : streamlit run dashboards/streamlit/app.py
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Le dashboard tourne dans deux contextes : en local (avec le paquet installé)
# et sur une instance SageMaker, où seul ce fichier est déposé. D'où le repli
# sur l'environnement : S3_BUCKET suffit, le rôle de l'instance fournit les droits.
try:
    from retail_forecast.config import ROOT, SETTINGS

    BUCKET = SETTINGS.bucket
except Exception:  # mode « fichier seul » (SageMaker, Colab, etc.)
    ROOT = Path.cwd()
    BUCKET = os.environ.get("S3_BUCKET", "retail-forecast-654654161593-ayoub")


def _s3():
    """Client S3 minimal : identifiants de l'environnement (rôle IAM)."""
    import boto3

    return boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))


st.set_page_config(page_title="Retail ML Forecasting", page_icon="🛒", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def load_forecast() -> tuple[pd.DataFrame, str]:
    """Prévisions : S3 d'abord, sinon data/forecast/."""
    local = ROOT / "data" / "forecast" / "forecast.parquet"
    if BUCKET:
        try:
            body = _s3().get_object(
                Bucket=BUCKET, Key="forecast/table/forecast.parquet"
            )["Body"].read()
            return pd.read_parquet(io.BytesIO(body)), f"s3://{BUCKET}"
        except Exception:  # pas encore de prévision sur S3
            pass
    if local.exists():
        return pd.read_parquet(local), "local"
    st.error("Aucune prévision trouvée : lancer `make sagemaker-transform`.")
    st.stop()


@st.cache_data(ttl=300, show_spinner=False)
def load_history() -> pd.DataFrame:
    """Historique des ventes : S3 (curated) sinon data/raw/."""
    if BUCKET:
        try:
            body = _s3().get_object(
                Bucket=BUCKET, Key="curated/sales/sales.parquet"
            )["Body"].read()
            df = pd.read_parquet(io.BytesIO(body))
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            pass
    local_raw = ROOT / "data" / "raw" / "sales.parquet"
    if not local_raw.exists():
        st.error(
            "Historique introuvable : ni `curated/sales/` dans S3, ni "
            "`data/raw/sales.parquet` en local."
        )
        st.stop()
    df = pd.read_parquet(local_raw)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_metrics() -> dict:
    """Métriques du dernier entraînement (copie locale du rapport)."""
    if BUCKET:
        try:
            body = _s3().get_object(
                Bucket=BUCKET, Key="reports/cloud_metrics.json"
            )["Body"].read()
            return json.loads(body)
        except Exception:
            pass
    for name in ("cloud_metrics.json", "local_baseline.json"):
        path = ROOT / "reports" / name
        if path.exists():
            return json.loads(path.read_text())
    return {}  # sur SageMaker les métriques ne sont pas déposées avec l'app


forecast, source = load_forecast()
history = load_history()
metrics = load_metrics()

st.title("🛒 Prévision de la demande — J+1 à J+28")
st.caption(
    f"source prévisions : {source} · origine {forecast['origin_date'].max()} · "
    f"modèle {forecast['model_version'].iloc[0]} ({forecast['model_job'].iloc[0]})"
)

# ---- filtres ---------------------------------------------------------------
stores = sorted(forecast["store_id"].unique())
with st.sidebar:
    st.header("Filtres")
    chosen_stores = st.multiselect("Magasins", stores, default=stores[:3])
    horizons = st.multiselect(
        "Horizons (jours)", sorted(forecast["horizon"].unique()), default=[1, 7]
    )
    st.divider()
    st.caption("WAPE du modèle (test) :")
    if metrics:
        wape_model = metrics.get("test_wape", 0)
        wape_base = metrics.get("baseline_wape", 0)
        st.metric("Modèle", f"{wape_model:.1%}", f"{(wape_model - wape_base):+.1%} vs baseline")
        st.metric("Baseline saisonnière", f"{wape_base:.1%}")

view = forecast[forecast["store_id"].isin(chosen_stores or stores)]
view = view[view["horizon"].isin(horizons or sorted(forecast["horizon"].unique()))]

# ---- KPI -------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Unités prévues", f"{view['forecast_units'].sum():,.0f}")
k2.metric("Séries couvertes", f"{view[['store_id', 'item_id']].drop_duplicates().shape[0]:,}")
k3.metric("Dates cibles", view["target_date"].nunique())
k4.metric("Horizons", ", ".join(f"J+{h}" for h in sorted(view["horizon"].unique())))

# ---- prévu vs réalisé ------------------------------------------------------
st.subheader("Prévu vs réalisé — une série")
series = (
    view.groupby(["store_id", "item_id"])["forecast_units"].sum().sort_values(ascending=False)
)
if len(series):
    store, item = series.index[0]
    c1, c2 = st.columns([2, 1])
    with c1:
        label = st.selectbox(
            "Série (magasin · article)",
            [f"{s} · {i}" for s, i in series.index[:50]],
        )
        store, item = (part.strip() for part in label.split("·"))
        hist = history[
            (history["store_id"] == store) & (history["item_id"] == item)
        ].sort_values("date").tail(120)
        fut = view[
            (view["store_id"] == store) & (view["item_id"] == item)
        ].sort_values("target_date")

        fig = px.line(hist, x="date", y="units", labels={"units": "unités", "date": ""})
        fig.update_traces(name="réalisé", line_color="#7aa2f7")
        if len(fut):
            fig.add_scatter(
                x=fut["target_date"],
                y=fut["forecast_units"],
                mode="lines+markers",
                name="prévision",
                line=dict(color="#f7a85b", dash="dash"),
            )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        total_hist = hist["units"].tail(28).mean()
        total_fut = fut["forecast_units"].mean() if len(fut) else 0
        st.metric("Moyenne 28 j réalisée", f"{total_hist:.1f} u/j")
        st.metric("Moyenne prévue", f"{total_fut:.1f} u/j",
                  f"{(total_fut - total_hist) / total_hist:+.1%}" if total_hist else None)
        st.caption("Un écart fort signale un pic attendu (promo, férié) "
                   "ou une dérive du modèle.")

# ---- top articles ----------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Top 10 articles — unités prévues")
    top = (
        view.groupby("item_id")["forecast_units"].sum().sort_values(ascending=False).head(10)
    )
    fig = px.bar(top[::-1], orientation="h", labels={"value": "unités", "item_id": ""})
    fig.update_layout(height=360, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Prévisions par magasin")
    by_store = view.groupby("store_id")["forecast_units"].sum().sort_values(ascending=False)
    fig = px.bar(by_store, labels={"value": "unités", "store_id": ""})
    fig.update_layout(height=360, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---- liste de risque -------------------------------------------------------
st.subheader("⚠️ Séries à surveiller (volatilité élevée)")
recent = history[history["date"] > history["date"].max() - pd.Timedelta(days=28)]
stats = (
    recent.groupby(["store_id", "item_id"])["units"]
    .agg(moyenne="mean", ecart_type="std")
    .reset_index()
)
stats["variation"] = stats["ecart_type"] / stats["moyenne"].replace(0, pd.NA)
risk = (
    stats.merge(
        view.groupby(["store_id", "item_id"])["forecast_units"].sum().reset_index(),
        on=["store_id", "item_id"],
        how="inner",
    )
    .sort_values("variation", ascending=False)
    .head(15)
)
st.dataframe(
    risk.rename(
        columns={
            "store_id": "magasin",
            "item_id": "article",
            "moyenne": "ventes moy. 28 j",
            "ecart_type": "écart-type",
            "variation": "coefficient de variation",
            "forecast_units": "unités prévues",
        }
    ),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "Coefficient de variation = écart-type / moyenne sur 28 jours. "
    "Plus il est élevé, plus la demande est irrégulière : c'est là qu'un stock "
    "de sécurité large (P90) se justifie."
)

with st.expander("Télécharger les prévisions (CSV)"):
    st.download_button(
        "forecast.csv",
        view.to_csv(index=False).encode("utf-8"),
        file_name="forecast.csv",
        mime="text/csv",
    )

