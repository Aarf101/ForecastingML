"""Rapport hebdomadaire rédigé par un LLM local (Ollama).

C'est la boîte « Rapport IA » du diagramme d'inférence, en version gratuite :
aucun appel à Bedrock, le modèle tourne sur la machine (``qwen2.5:3b``), et le
prompt ne contient **que des agrégats** — jamais de lignes brutes.

Sortie : ``reports/weekly_brief.md`` en local et, si ``S3_BUCKET`` est
configuré, ``s3://bucket/reports/weekly_brief_<date>.md``.

Usage::

    python -m retail_forecast.agent.brief
    python -m retail_forecast.agent.brief --model qwen2.5:3b --no-s3
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from retail_forecast.config import ROOT, SETTINGS

PROMPT = """Tu es analyste supply chain. Rédige un brief de 120 mots maximum, en français, \
à partir de ces chiffres. Trois phrases de constat, puis deux actions concrètes. \
Pas de titres, pas de listes, pas d'invention de chiffres.

Unités prévues pour la journée cible, par horizon : {totaux}
Top 3 magasins : {magasins}
Séries les plus volatiles : {volatiles}
Performance du modèle : WAPE {wape:.1%} contre {baseline:.1%} pour la baseline \
saisonnière, soit {gain:.0%} de mieux ; biais {biais:+.1%}.
"""


def _ollama(prompt: str, model: str, base_url: str, timeout: int = 180) -> str:
    """Appelle l'API /api/generate d'Ollama (pas de SDK : une requête HTTP)."""
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())["response"].strip()


def _load_forecast() -> pd.DataFrame:
    """Prévisions : S3 si configuré, sinon copie locale."""
    if SETTINGS.bucket:
        from retail_forecast.ingest.storage import s3_client

        try:
            body = s3_client().get_object(
                Bucket=SETTINGS.bucket, Key="forecast/table/forecast.parquet"
            )["Body"].read()
            return pd.read_parquet(io.BytesIO(body))
        except Exception:
            pass
    return pd.read_parquet(ROOT / "data" / "forecast" / "forecast.parquet")


def _load_metrics() -> dict:
    cloud = ROOT / "reports" / "cloud_metrics.json"
    path = cloud if cloud.exists() else ROOT / "reports" / "local_baseline.json"
    return json.loads(path.read_text()) if path.exists() else {}


def build_prompt(forecast: pd.DataFrame) -> str:
    """Agrège les prévisions puis remplit le gabarit du prompt."""
    totaux = (
        forecast.groupby("horizon")["forecast_units"].sum().round(0).astype(int).to_dict()
    )
    magasins = (
        forecast.groupby("store_id")["forecast_units"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
        .round(0)
        .astype(int)
        .to_dict()
    )
    series = (
        forecast.groupby(["store_id", "item_id"])["forecast_units"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
    )
    metrics = _load_metrics()
    return PROMPT.format(
        totaux=", ".join(f"J+{h} : {v} u" for h, v in sorted(totaux.items())),
        magasins=", ".join(f"{s} ({v} u)" for s, v in magasins.items()),
        volatiles=", ".join(f"{s}/{i} ({v:.0f} u)" for (s, i), v in series.items()),
        wape=metrics.get("test_wape", 0.0),
        baseline=metrics.get("baseline_wape", 0.0),
        biais=metrics.get("test_bias", 0.0),
        gain=(
            (metrics.get("baseline_wape", 0.0) - metrics.get("test_wape", 0.0))
            / metrics.get("baseline_wape", 1.0)
            if metrics.get("baseline_wape")
            else 0.0
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Brief hebdomadaire (LLM local).")
    parser.add_argument("--model", default=SETTINGS.ollama_model)
    parser.add_argument("--base-url", default=SETTINGS.ollama_base_url)
    parser.add_argument("--no-s3", action="store_true")
    args = parser.parse_args(argv)

    forecast = _load_forecast()
    prompt = build_prompt(forecast)
    print(f"[brief] modèle {args.model} · {len(forecast):,} prévisions agrégées")
    text = _ollama(prompt, args.model, args.base_url)

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    content = f"# Brief ventes — {day}\n\n{text}\n"
    out = ROOT / "reports" / "weekly_brief.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    print(content)
    print(f"[brief] écrit {out.relative_to(ROOT)}")

    if SETTINGS.bucket and not args.no_s3:
        from retail_forecast.ingest.storage import s3_client

        key = f"reports/weekly_brief_{day}.md"
        s3_client().put_object(
            Bucket=SETTINGS.bucket,
            Key=key,
            Body=content.encode("utf-8"),
            ContentType="text/markdown",
        )
        print(f"[brief] écrit s3://{SETTINGS.bucket}/{key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
