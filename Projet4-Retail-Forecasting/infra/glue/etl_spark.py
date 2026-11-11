"""Job Glue (PySpark) — raw/ -> curated/ : version managée de l'ETL local.

À coller dans Glue Studio (onglet *Script*) ou à déposer dans S3 puis à
référencer depuis la console.

Sources lues (les deux, si présentes) :
  * ``s3://<bucket>/raw/sales/``        micro-lots gzip JSONL produits par le
    consumer Kafka (``to_s3.py``) — un fichier = un flush de 500 événements ;
  * ``s3://<bucket>/raw/historique/``   Parquet historique (rejeu initial).

Ce script fait exactement ce que fait ``retail_forecast.etl.build_curated`` :
déduplication sur (store_id, item_id, date) en gardant la **dernière réception**
(event_time le plus récent), typage, tri, écriture Parquet snappy. Il tourne sur
des DPU Glue, donc il est facturé à la minute (0,44 $/DPU-h, minimum 10 min).

Paramètres de job attendus : ``--BUCKET`` (obligatoire), ``--TARGET`` (optionnel,
défaut ``curated/sales/`` — le dossier lu par le Training Job ; le Parquet
historique reste disponible dans ``raw/historique/`` pour reconstruire un
``curated/`` propre avec ``make etl``).
"""

import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# Schéma cible commun aux deux sources : un lot JSONL et une ligne Parquet
# n'ont pas les mêmes types (date texte vs timestamp, event_time absent de
# l'historique). On aligne tout avant l'union, sinon Spark lève
# AnalysisException sur un type incompatible.
SCHEMA = {
    "date": "timestamp",
    "store_id": "string",
    "item_id": "string",
    "category": "string",
    "unit_price": "float",
    "promo": "int",
    "is_holiday": "int",
    "lead_time_days": "int",
    "stock_on_hand": "int",
    "stockout": "int",
    "units": "int",
    "event_time": "timestamp",
}

args = getResolvedOptions(sys.argv, ["JOB_NAME", "BUCKET"])
sc = SparkContext.getOrCreate()
glue_context = GlueContext(sc)
spark = glue_context.spark_session
job = Job(glue_context)
job.init(args["JOB_NAME"], args)

bucket = args["BUCKET"]
batches_path = f"s3://{bucket}/raw/sales/"
history_path = f"s3://{bucket}/raw/historique/"
target = f"s3://{bucket}/{args.get('TARGET', 'curated/sales/').lstrip('/')}"


_TYPES = {
    "timestamp": TimestampType(),
    "string": StringType(),
    "int": IntegerType(),
    "float": FloatType(),
}


def struct():
    """StructType explicite : l'inférence échoue sur les objets gzip de Firehose."""
    return StructType([StructField(n, _TYPES[t], True) for n, t in SCHEMA.items()])


def has_objects(path: str) -> bool:
    """Vrai si le préfixe S3 contient au moins un objet (hors marqueurs _)."""
    jvm = spark._jvm
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(
        jvm.java.net.URI(path), spark._jsc.hadoopConfiguration()
    )
    p = jvm.org.apache.hadoop.fs.Path(path)
    if not fs.exists(p):
        return False
    return any(not s.getPath().getName().startswith("_") for s in fs.listStatus(p))


def align(df):
    """Projette un DataFrame sur ``SCHEMA`` (colonne absente -> NULL typé)."""
    cols = []
    present = set(df.columns)
    for name, dtype in SCHEMA.items():
        if name in present:
            cols.append(F.col(name).cast(dtype).alias(name))
        else:
            cols.append(F.lit(None).cast(dtype).alias(name))
    return df.select(*cols)


frames = []

# 1. micro-lots du flux Kafka (JSONL gzip : Spark gère le .gz nativement)
if has_objects(batches_path):
    batches = spark.read.schema(struct()).json(batches_path)
    n = batches.count()
    print(f"[glue] lots Kafka lus : {n:,} lignes ({batches_path})")
    if n:
        frames.append(align(batches))
else:
    print(f"[glue] aucun lot Kafka sous {batches_path}")

# 2. historique Parquet (rejeu initial du catalogue)
if has_objects(history_path):
    history = spark.read.parquet(history_path)
    n = history.count()
    print(f"[glue] historique lu : {n:,} lignes ({history_path})")
    if n:
        frames.append(align(history))
else:
    print(f"[glue] aucun historique sous {history_path}")

if not frames:
    raise RuntimeError(f"aucune source exploitable sous s3://{bucket}/raw/")

raw = frames[0]
for extra in frames[1:]:
    raw = raw.unionByName(extra)

total = raw.count()
print(f"[glue] lignes brutes (union) : {total:,}")

# 3. déduplication : un événement peut être rejoué, la dernière réception gagne.
#    row_number() plutôt que dropDuplicates() : l'ordre est explicite, donc le
#    résultat est déterministe (event_time le plus récent, NULL en dernier).
window = Window.partitionBy("store_id", "item_id", "date").orderBy(
    F.col("event_time").desc_nulls_last()
)

deduped = (
    raw.withColumn("_rn", F.row_number().over(window))
    .filter(F.col("_rn") == 1)
    .drop("_rn")
    .orderBy("store_id", "item_id", "date")
)

clean = deduped.count()
print(f"[glue] lignes propres : {clean:,} ({total - clean:,} doublons supprimés)")

# 4. écriture Parquet (snappy) lisible par Athena et par le Training Job
(
    deduped.write.mode("overwrite")
    .option("compression", "snappy")
    .parquet(target)
)
print(f"[glue] écrit {target}")

job.commit()
