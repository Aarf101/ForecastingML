"""Job Glue (PySpark) — raw/ -> curated/ : version managée de l'ETL local.

À coller dans Glue Studio (onglet *Script*) ou à déposer dans S3 puis à
référencer depuis la console. Paramètre de job attendu : ``BUCKET``.

Ce script fait exactement ce que fait ``retail_forecast.etl.build_curated`` :
lecture du brut, déduplication sur (store_id, item_id, date), typage, écriture
Parquet snappy. Il tourne sur des DPU Glue, donc il est facturé à la minute
(0,44 $/DPU-h, minimum 10 min) — à lancer une fois pour la démonstration.
"""

import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as F

args = getResolvedOptions(sys.argv, ["JOB_NAME", "BUCKET"])
sc = SparkContext.getOrCreate()
glue_context = GlueContext(sc)
spark = glue_context.spark_session
job = Job(glue_context)
job.init(args["JOB_NAME"], args)

bucket = args["BUCKET"]
source = f"s3://{bucket}/raw/historique/"
target = f"s3://{bucket}/curated/sales/"

# 1. lecture du brut (historique Parquet)
raw = spark.read.parquet(source)
print(f"[glue] lignes lues : {raw.count():,}")

# 2. déduplication : un événement peut être rejoué, la dernière réception gagne
deduped = (
    raw.withColumn("date", F.to_date("date"))
    .dropDuplicates(["store_id", "item_id", "date"])
    .withColumn("units", F.col("units").cast("int"))
    .withColumn("unit_price", F.col("unit_price").cast("float"))
    .withColumn("promo", F.col("promo").cast("int"))
    .withColumn("is_holiday", F.col("is_holiday").cast("int"))
    .withColumn("lead_time_days", F.col("lead_time_days").cast("int"))
    .orderBy("store_id", "item_id", "date")
)
print(f"[glue] lignes propres : {deduped.count():,}")

# 3. écriture Parquet (snappy) lisible par Athena et par le Training Job
(
    deduped.write.mode("overwrite")
    .option("compression", "snappy")
    .parquet(target)
)
print(f"[glue] écrit {target}")

job.commit()
