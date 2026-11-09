"""Athena : base + tables externes sur S3, puis quelques KPI en SQL.

Pas de Glue Crawler (facturé au DPU) : les tables sont créées par DDL, ce qui
coûte zéro et reste reproductible. C'est la branche « Athena — KPI historiques »
du diagramme d'inférence.

Usage::

    python -m infra.athena.setup            # crée base + tables + 2 KPI
    python -m infra.athena.setup --kpi-only # seulement les requêtes
"""

from __future__ import annotations

import argparse
import time

from retail_forecast.config import ROOT, SETTINGS
from retail_forecast.ingest.storage import s3_client

DATABASE = "retail_forecast"
WORKGROUP = "primary"
OUTPUT_PREFIX = "athena-results/"

DDL = [
    f"CREATE DATABASE IF NOT EXISTS {DATABASE}",
    f"""CREATE EXTERNAL TABLE IF NOT EXISTS {DATABASE}.curated_sales (
        date TIMESTAMP, store_id STRING, item_id STRING, category STRING,
        unit_price FLOAT, promo INT, is_holiday INT, lead_time_days INT,
        units INT, stock_on_hand INT, stockout INT
    ) STORED AS PARQUET
    LOCATION 's3://{SETTINGS.bucket}/curated/sales/'""",
    f"""CREATE EXTERNAL TABLE IF NOT EXISTS {DATABASE}.forecast (
        store_id STRING, item_id STRING, origin_date TIMESTAMP, target_date TIMESTAMP,
        horizon INT, forecast_units DOUBLE, model_version STRING,
        model_job STRING, run_date STRING
    ) STORED AS PARQUET
    LOCATION 's3://{SETTINGS.bucket}/forecast/table/'""",
]

KPI = {
    "top_articles_90j": f"""
        SELECT item_id, SUM(units) AS unites, ROUND(SUM(units * unit_price), 2) AS ca
        FROM {DATABASE}.curated_sales
        WHERE date >= TIMESTAMP '2024-10-01 00:00:00'
        GROUP BY item_id ORDER BY unites DESC LIMIT 10""",
    "prevision_par_magasin": f"""
        SELECT store_id, horizon, ROUND(SUM(forecast_units), 1) AS unites_prevues
        FROM {DATABASE}.forecast
        WHERE horizon IN (1, 7)
        GROUP BY store_id, horizon ORDER BY store_id, horizon""",
}


def _athena():
    import boto3

    kwargs: dict = {"region_name": SETTINGS.region}
    if SETTINGS.s3_access_key:
        kwargs["aws_access_key_id"] = SETTINGS.s3_access_key
        kwargs["aws_secret_access_key"] = SETTINGS.s3_secret_key
    return boto3.client("athena", **kwargs)


def run_query(client, sql: str, label: str, wait: bool = True) -> list[tuple]:
    """Exécute une requête et renvoie les lignes (liste de tuples)."""
    query_id = client.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": DATABASE},
        ResultConfiguration={"OutputLocation": f"s3://{SETTINGS.bucket}/{OUTPUT_PREFIX}"},
        WorkGroup=WORKGROUP,
    )["QueryExecutionId"]
    if not wait:
        print(f"[athena] {label} lancé ({query_id})")
        return []

    for _ in range(120):
        state = client.get_query_execution(QueryExecutionId=query_id)["QueryExecution"][
            "Status"
        ]["State"]
        if state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            break
        time.sleep(1)
    if state != "SUCCEEDED":
        reason = client.get_query_execution(QueryExecutionId=query_id)["QueryExecution"][
            "Status"
        ].get("StateChangeReason", "")
        raise SystemExit(f"[athena] {label} : {state} — {reason}")

    result = client.get_query_results(QueryExecutionId=query_id)
    rows = [
        tuple(col.get("VarCharValue") for col in row["Data"])
        for row in result["ResultSet"]["Rows"]
    ]
    print(f"[athena] {label} : {len(rows) - 1} lignes")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Athena : tables + KPI sur S3.")
    parser.add_argument("--kpi-only", action="store_true")
    args = parser.parse_args(argv)

    client = _athena()
    if not args.kpi_only:
        for sql in DDL:
            label = sql.split("(")[0].split()[:5]
            run_query(client, sql, " ".join(label))

    out_dir = ROOT / "reports" / "athena"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, sql in KPI.items():
        rows = run_query(client, sql, name)
        path = out_dir / f"{name}.csv"
        path.write_text("\n".join(",".join(r) for r in rows), encoding="utf-8")
        print(f"[athena]   -> {path.relative_to(ROOT)}")
        for row in rows[:4]:
            print("          " + " | ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
