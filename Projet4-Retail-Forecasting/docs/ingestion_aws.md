# Ingestion 100 % AWS — Kinesis → Firehose → S3 → Glue

Cette note décrit la chaîne d'ingestion **entièrement exécutée dans le compte
AWS**, sans broker ni producteur sur la machine locale. Elle remplace la version
de démonstration `Redpanda + to_s3.py`.

## 1. La chaîne

```
s3://bucket/replay/sales/chunk-*.json.gz        (dataset préparé une fois)
              │
              ▼
   Lambda retail-replay-to-kinesis               le producteur, dans AWS
   (PutRecords par lots de 500, checkpoint S3)
              │
              ▼
   Kinesis Data Streams retail-sales-stream      le transport (1 shard)
              │
              ▼
   Firehose retail-sales-to-s3                   buffer 60 s / 5 Mo
              │
              ▼
   s3://bucket/raw/sales/ingest_date=YYYY-MM-DD/<uuid>.gz
              │
              ▼
   Job Glue retail-etl-raw-to-curated            spark.read.json + historique
              │
              ▼
   s3://bucket/curated/sales_glue/               Parquet snappy
```

Chaque flèche correspond à un service managé : il n'y a **aucun processus** sur le
poste de travail une fois le déploiement fait.

## 2. Rôle de chaque brique

| Brique | Remplace | Rôle |
|---|---|---|
| **Lambda** `retail-replay-to-kinesis` | `ingest/producer.py` | relit le dataset depuis S3 et publie dans le stream ; checkpoint `(chunk, offset)` dans S3 pour reprendre après interruption |
| **Kinesis Data Streams** `retail-sales-stream` | Kafka / MSK | transport d'événements, ordre garanti par `PartitionKey` (= `store_id`) |
| **Firehose** `retail-sales-to-s3` | `ingest/to_s3.py` | bufferise (60 s ou 5 Mo), compresse en gzip, dépose dans `raw/sales/` avec partitionnement dynamique par date |
| **Glue** `retail-etl-raw-to-curated` | `etl/build_curated.py` | lit les lots `.gz` **et** l'historique Parquet, déduplique, type, écrit `curated/sales_glue/` |

## 3. Commandes

```bash
# 1. préparer les morceaux (une fois) : parquet -> JSONL gzip dans s3://.../replay/
make aws-ingest-prep

# 2. créer le stream, la livraison Firehose et le Lambda (~2 min)
make aws-ingest-up

# 3. rejouer le dataset (le Lambda s'arrête au bout du budget ou à la fin)
make aws-ingest-run

# 4. vérifier : état des 3 objets + nombre d'objets livrés
make aws-ingest-status

# 5. tout supprimer (Firehose, puis Lambda, puis stream)
make aws-ingest-down
```

Puis relancer le job Glue avec le script `infra/glue/etl_spark.py` : il consomme
désormais `raw/sales/`.

## 4. Points de conception

- **Checkpoint plutôt que rejeu complet.** Le Lambda écrit `{"chunk": i,
  "offset": j}` dans `s3://bucket/replay/_state/checkpoint.json` après chaque lot.
  Une invocation suivante reprend exactement là où la précédente s'est arrêtée :
  pas de doublon massif, pas de perte.
- **Pas de couche Lambda.** Le handler ne dépend que de `gzip`, `json` et `boto3`,
  tous fournis par le runtime `python3.12`. C'est pourquoi le dataset est
  pré-découpé en JSONL : cela évite d'installer `pandas`/`pyarrow` dans un layer.
- **Partitionnement dynamique Firehose.** `!{timestamp:yyyy-MM-dd}` recrée la
  convention `ingest_date=…` de la version locale, donc les deux écritures sont
  interchangeables côté lecteur.
- **Firehose nomme ses objets `<uuid>.gz`** (pas `.json.gz`). `build_curated.py`
  accepte désormais les deux extensions ; Spark décompresse d'après le suffixe
  `.gz`.
- **Aucun rôle IAM créé.** `LabRole` est réutilisé partout (Lambda, Firehose) :
  `iam:CreateRole` est refusé par le lab, mais `iam:PassRole` sur ce rôle
  fonctionne.

## 5. Coût

| Ressource | Tarif | Pour une démo de 2 h |
|---|---|---|
| Kinesis, 1 shard provisionné | 0,015 $/h | 0,03 $ |
| Firehose | 0,029 $/Go livré | < 0,01 $ (2 Mo) |
| Lambda | free tier (400 000 Go-s/mois) | 0 $ |
| Glue | 0,44 $/DPU-h, min 10 min | 0,15 $ |
| **Total démo** | | **≈ 0,19 $** |

`make aws-ingest-down` supprime le shard : plus aucun coût horaire.

## 6. Différence avec la version locale

| | Version locale (démo) | Version AWS (celle-ci) |
|---|---|---|
| Transport | Redpanda dans Docker | Kinesis Data Streams |
| Producteur | `producer.py` sur le poste | Lambda dans le compte |
| Routage vers S3 | `to_s3.py` (consumer) | Firehose |
| Broker sur le poste | oui | **non** |
| Coût | 0 € | ≈ 0,19 $/démo |

Les deux écrivent au même endroit, avec la même convention de clé, donc l'ETL et
tout l'aval sont identiques.
