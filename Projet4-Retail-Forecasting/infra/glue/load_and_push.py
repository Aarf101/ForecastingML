"""Glue (Spark) : charge un dataset quelconque, le normalise, l'écrit dans
``raw/historique/`` **et** pousse les lignes dans Firehose — le tout depuis la
console Glue, sans terminal, sans EC2, sans Kafka.

C'est le moyen de tester un dataset différent en 100 % clics :
    1. uploader le fichier dans ``s3://<bucket>/uploads/``
    2. coller ce script dans Glue Studio (Spark)
    3. Run

Paramètres de job (Job details -> Job parameters) :

    --BUCKET    bucket du projet                    (obligatoire)
    --SOURCE    s3://bucket/uploads/mon_fichier.csv (obligatoire)
    --DATASET   nom court -> raw/historique/<nom>.parquet   (défaut: extrait)
    --STREAM    delivery stream Firehose            (défaut: retail-sales-direct)
    --LIMIT     lignes poussées dans Firehose       (défaut: 20000)
    --MAPPING   "Store=store_id,Sales=units"        (optionnel)

Rôle IAM : ``LabRole`` (aucun rôle à créer). 2 × G.1X suffisent.
"""

import sys

import boto3
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as F

args = getResolvedOptions(sys.argv, ["BUCKET", "SOURCE", "JOB_NAME"])
opts = dict(zip(sys.argv[1::2], sys.argv[2::2]))
bucket = args["BUCKET"]
source = args["SOURCE"]
dataset = opts.get("--DATASET", "extrait")
stream = opts.get("--STREAM", "retail-sales-direct")
limit = int(opts.get("--LIMIT", "20000"))
target = f"s3://{bucket}/raw/historique/{dataset}.parquet"

# --- alias de colonnes (mêmes règles que data/load.py) -----------------------
ALIASES = {
    "ds": "date", "day": "date", "datum": "date", "vente_date": "date",
    "store": "store_id", "store_nbr": "store_id", "magasin": "store_id",
    "shop": "store_id", "site": "store_id",
    "item": "item_id", "item_nbr": "item_id", "sku": "item_id",
    "article": "item_id", "product": "item_id",
    "sales": "units", "qty": "units", "quantity": "units", "demand": "units",
    "quantite": "units", "sold": "units",
    "price": "unit_price", "sell_price": "unit_price", "prix": "unit_price",
    "onpromotion": "promo", "promotion": "promo", "is_promo": "promo",
    "holiday": "is_holiday", "ferie": "is_holiday",
    "family": "category", "dept": "category", "department": "category",
    "lead_time": "lead_time_days",
}

explicit = {}
for pair in (opts.get("--MAPPING") or "").split(","):
    if "=" in pair:
        src, _, dst = pair.partition("=")
        explicit[src.strip()] = dst.strip()

spark = GlueContext(SparkContext.getOrCreate()).spark_session
# --- 1. lecture brute -------------------------------------------------------
raw = spark.read.option("header", True).option("inferSchema", True).csv(source)
print(f"[load] {source} : {raw.count():,} lignes · {len(raw.columns)} colonnes")

rename = {}
for col in raw.columns:
    low = col.strip().lower()
    rename[col] = explicit.get(col) or ALIASES.get(low, low)
df = raw.toDF(*[rename[c] for c in raw.columns])

missing = [c for c in ("date", "store_id", "item_id", "units") if c not in df.columns]
if missing:
    raise SystemExit(f"colonnes obligatoires absentes : {missing} (colonnes: {df.columns})")

# --- 2. normalisation -------------------------------------------------------
defaults = {
    "category": "unknown", "unit_price": 0.0, "promo": 0,
    "is_holiday": 0, "lead_time_days": 7, "stock_on_hand": 0, "stockout": 0,
}
for col, value in defaults.items():
    if col not in df.columns:
        df = df.withColumn(col, F.lit(value))

df = (
    df.withColumn("date", F.to_timestamp(F.col("date").cast("string")).cast("date"))
    .withColumn("units", F.col("units").cast("int"))
    .withColumn("promo", F.col("promo").cast("int"))
    .withColumn("is_holiday", F.col("is_holiday").cast("int"))
    .withColumn("lead_time_days", F.col("lead_time_days").cast("int"))
    .withColumn("stock_on_hand", F.col("stock_on_hand").cast("int"))
    .withColumn("stockout", F.col("stockout").cast("int"))
    .withColumn("unit_price", F.col("unit_price").cast("float"))
    .withColumn("store_id", F.col("store_id").cast("string"))
    .withColumn("item_id", F.col("item_id").cast("string"))
    .withColumn("category", F.col("category").cast("string"))
    .dropna(subset=["date", "store_id", "item_id", "units"])
    .dropDuplicates(["store_id", "item_id", "date"])
)

cols = ["date", "store_id", "item_id", "category", "unit_price", "promo",
        "is_holiday", "lead_time_days", "stock_on_hand", "stockout", "units"]
df = df.select(*cols)
n = df.count()
print(f"[load] normalisé : {n:,} lignes · {df.select('store_id').distinct().count()} magasins"
      f" · {df.select('item_id').distinct().count()} articles")

# --- 3. écriture Parquet (historique) ---------------------------------------
df.write.mode("overwrite").option("compression", "snappy").parquet(target)
print(f"[load] écrit {target}")

# --- 4. poussée vers Firehose (rôle du pont) --------------------------------
firehose = boto3.client("firehose")
sample = df.limit(limit).toJSON().collect()
sent = 0
for i in range(0, len(sample), 500):
    records = [{"Data": (line + "\n").encode("utf-8")} for line in sample[i : i + 500]]
    resp = firehose.put_record_batch(DeliveryStreamName=stream, Records=records)
    if resp.get("FailedPutCount"):
        raise RuntimeError(f"Firehose a refusé {resp['FailedPutCount']} enregistrements")
    sent += len(records)
print(f"[load] {sent:,} enregistrements poussés vers Firehose {stream}")
print("[load] Firehose livre sous ~60 s -> s3://%s/raw/sales/" % bucket)
