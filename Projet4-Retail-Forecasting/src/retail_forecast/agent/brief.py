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

PROMPT = """Prévisions ponctuelles par horizon : {totaux}.
Ces horizons sont des dates isolées, pas des cumuls hebdomadaires.
Magasins aux volumes prévus les plus élevés : {magasins}.
Séries aux volumes prévus les plus élevés : {volatiles}.
Évaluation : {qualite}.
FAITS_VALIDES_JSON: {faits_json}
ACTIONS_VALIDES_JSON: {actions_json}
Réponds uniquement en JSON avec deux clés : "analyse" et "actions".
Pour "analyse", recopie EXACTEMENT les trois phrases de FAITS_VALIDES_JSON,
dans leur ordre, sans reformulation ni ajout. Pour "actions", choisis exactement
deux phrases différentes de ACTIONS_VALIDES_JSON, recopiées sans modification.
Ne crée aucune phrase : les agrégats ne permettent pas une analyse individuelle
de qualité, de tendance, de rupture ou de production.
"""


def _forecast_key():
    """Le manifeste lie atomiquement le modèle actif à son lot validé."""
    from infra.sagemaker.lifecycle import read_active
    from retail_forecast.ingest.storage import s3_client

    active, _etag = read_active(s3_client())
    return active["forecast_key"]


def _ollama(prompt: str, model: str, base_url: str, timeout: int = 180) -> str:
    """Appelle l'API /api/generate d'Ollama (pas de SDK : une requête HTTP)."""
    output_format = "json"
    choices = {line.split(": ", 1)[0]: json.loads(line.split(": ", 1)[1])
               for line in prompt.splitlines()
               if line.startswith(("FAITS_VALIDES_JSON: ", "ACTIONS_VALIDES_JSON: "))}
    if choices:
        # Le décodage contraint complète la validation après génération.
        output_format = {"type": "object", "additionalProperties": False,
            "properties": {
                "analyse": {"const": choices["FAITS_VALIDES_JSON"]},
                "actions": {"type": "array", "items": {"type": "string", "enum": choices["ACTIONS_VALIDES_JSON"]},
                            "minItems": 2, "maxItems": 2}},
            "required": ["analyse", "actions"]}
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False,
                          "format": output_format, "options": {"temperature": 0, "seed": 42}}).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())["response"].strip()


def _sagemaker(prompt: str, endpoint: str) -> str:
    """Appelle l'endpoint Hugging Face sur SageMaker (remplace Bedrock)."""
    import boto3

    kwargs: dict = {"region_name": SETTINGS.region}
    if SETTINGS.s3_access_key:
        kwargs["aws_access_key_id"] = SETTINGS.s3_access_key
        kwargs["aws_secret_access_key"] = SETTINGS.s3_secret_key
    client = boto3.client("sagemaker-runtime", **kwargs)
    # Qwen2.5 est un modèle conversationnel : sans template ChatML il répond en
    # anglais et divague. Le conteneur HF applique le template si on lui donne
    # les messages, mais on le construit explicitement pour être déterministe.
    chat = (
        "<|im_start|>system\n"
        "Tu es un analyste supply chain. Tu réponds uniquement en français, "
        "en texte suivi, sans titres ni listes.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    payload = {
        "inputs": chat,
        "parameters": {
            "max_new_tokens": 150,
            "do_sample": False,
            "return_full_text": False,
        },
    }
    response = client.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    body = json.loads(response["Body"].read())
    text = body[0]["generated_text"] if isinstance(body, list) else str(body)
    return text.strip()


def generate(prompt: str, provider: str, *, model=None, base_url=None) -> str:
    """Choisit le backend LLM (``ollama`` en local, ``sagemaker`` sur AWS)."""
    if provider == "ollama":
        return _ollama(prompt, model or SETTINGS.ollama_model, base_url or SETTINGS.ollama_base_url)
    if provider == "sagemaker":
        return _sagemaker(prompt, SETTINGS.llm_endpoint)
    if provider == "bedrock":
        raise SystemExit(
            "Bedrock n'est pas activé dans ce compte (pas de bedrock:* dans les  politiques). Utiliser --provider sagemaker."
        )
    raise SystemExit(f"provider inconnu : {provider}")


def _load_forecast() -> pd.DataFrame:
    """Prévisions : S3 si configuré, sinon copie locale."""
    if SETTINGS.bucket and not SETTINGS.s3_endpoint_url:
        from retail_forecast.ingest.storage import s3_client

        try:
            body = s3_client().get_object(
                Bucket=SETTINGS.bucket, Key=_forecast_key()
            )["Body"].read()
            return pd.read_parquet(io.BytesIO(body))
        except Exception as exc:
            raise RuntimeError("prévisions actives AWS indisponibles") from exc
    return pd.read_parquet(ROOT / "data" / "forecast" / "forecast.parquet")


def _load_metrics() -> dict:
    if SETTINGS.bucket and not SETTINGS.s3_endpoint_url:
        from infra.sagemaker.lifecycle import read_active
        from retail_forecast.ingest.storage import s3_client
        active, _etag = read_active(s3_client())
        return active.get("metrics", {})
    path = ROOT / "reports" / "local_baseline.json"
    return json.loads(path.read_text()) if path.exists() else {}


def prompt_inputs(forecast: pd.DataFrame) -> dict:
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
    if "test_wape" in metrics and "baseline_wape" in metrics:
        quality = (f"WAPE modèle {metrics['test_wape']:.1%}, "
                   f"baseline {metrics['baseline_wape']:.1%}")
        if "test_bias" in metrics:
            quality += f", biais {metrics['test_bias']:+.1%}"
    else:
        quality = "métriques indisponibles ; qualité non évaluée"
    return dict(
        totaux=", ".join(f"J+{h} : {v} u" for h, v in sorted(totaux.items())),
        magasins=", ".join(f"{s} ({v} u)" for s, v in magasins.items()),
        volatiles=", ".join(f"{s}/{i} ({v:.0f} u)" for (s, i), v in series.items()),
        qualite=quality,
        faits_json=json.dumps([
            f"Le volume total prévu sur les {len(totaux)} horizons est de {forecast.forecast_units.sum():.0f} unités.",
            f"Le magasin au plus fort volume cumulé prévu est {next(iter(magasins))} avec {next(iter(magasins.values()))} unités.",
            f"Évaluation globale : {quality}.",
        ], ensure_ascii=False),
        actions_json=json.dumps([
            "Vérifier les stocks disponibles avant de décider des réassorts.",
            "Planifier les livraisons après vérification des délais fournisseurs.",
            "Examiner les intervalles P10/P90 avec le responsable des achats.",
            "Comparer les prévisions aux ventes observées dès leur disponibilité.",
        ], ensure_ascii=False),
    )


def build_prompt(forecast: pd.DataFrame) -> str:
    """Gabarit pur utilisable dans les tests sans backend LLM."""
    return PROMPT.format(**prompt_inputs(forecast))


def validate_brief(text: str, context: str) -> str:
    """Refuse les sorties mal formées et les chiffres absents du contexte."""
    import re
    document = json.loads(text)
    if set(document) != {"analyse", "actions"}:
        raise ValueError("structure du rapport LLM invalide")
    for key, size in [("analyse", 3), ("actions", 2)]:
        if not isinstance(document[key], list) or len(document[key]) != size or not all(isinstance(v, str) and v.strip() for v in document[key]):
            raise ValueError("nombre de phrases LLM incorrect")
    result = " ".join(document["analyse"]) + "\n\n" + " ".join(document["actions"])
    numbers = lambda value: {float(v.replace(",", ".")) for v in re.findall(r"\d+(?:[.,]\d+)?", value)}
    if not numbers(result).issubset(numbers(context)):
        raise ValueError("chiffre LLM absent du contexte : publication bloquée")
    if re.search(r"stable|stabilité|volatil", result, re.IGNORECASE):
        raise ValueError("propriété non mesurée dans le rapport LLM")
    allowed = {}
    for line in context.splitlines():
        for prefix in ("FAITS_VALIDES_JSON: ", "ACTIONS_VALIDES_JSON: "):
            if line.startswith(prefix):
                allowed[prefix] = json.loads(line[len(prefix):])
    if allowed:
        if (document['analyse'] != allowed['FAITS_VALIDES_JSON: ']
                or len(set(document['actions'])) != 2
                or any(action not in allowed['ACTIONS_VALIDES_JSON: '] for action in document['actions'])):
            raise ValueError("affirmation LLM non autorisée : publication bloquée")
    return result


def generate_brief(forecast: pd.DataFrame, provider: str, *, model=None, base_url=None) -> str:
    """Chaîne LangChain : agrégation, gabarit de prompt, génération textuelle."""
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda

    def complete(prompt):
        context = prompt.to_string()
        raw = generate(context, provider, model=model, base_url=base_url)
        return validate_brief(raw, context)

    chain = (RunnableLambda(prompt_inputs) | PromptTemplate.from_template(PROMPT)
             | RunnableLambda(complete))
    return chain.invoke(forecast)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Brief hebdomadaire (LLM local).")
    parser.add_argument("--model", default=SETTINGS.ollama_model)
    parser.add_argument("--base-url", default=SETTINGS.ollama_base_url)
    parser.add_argument("--no-s3", action="store_true")
    parser.add_argument(
        "--provider", default=SETTINGS.llm_provider, choices=["ollama", "sagemaker", "bedrock"]
    )
    args = parser.parse_args(argv)

    forecast = _load_forecast()
    print(f"[brief] provider {args.provider} · {len(forecast):,} prévisions agrégées")
    text = generate_brief(forecast, args.provider, model=args.model, base_url=args.base_url)

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
