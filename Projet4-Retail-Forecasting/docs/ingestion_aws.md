# Ingestion dans AWS — deux chaînes, et laquelle suit le schéma

Le sujet (diapositive 11) demande « **MSK ou Kafka open-source** » comme transport
et « **Lambda / Glue / Firehose** » pour la livraison. MSK est refusé par ce
compte (`kafka:ListClustersV2` → `AccessDeniedException`), donc le transport est
**Kafka open-source**, autorisé explicitement.

Deux variantes sont implémentées. **La variante B est celle qui suit le schéma.**

| | A. Kinesis → Firehose | **B. Kafka → Firehose** |
|---|---|---|
| Transport | Kinesis Data Streams | **Apache Kafka (EC2, compte AWS)** |
| Livraison | Firehose (`KinesisStreamAsSource`) | **Firehose (`DirectPut`)** |
| Fidèle à la diapo | ✗ pas de Kafka | ✓ Kafka **et** Firehose |
| Modules | `infra/aws/kinesis_ingest.py` | `infra/aws/firehose_direct.py` + `kafka_to_firehose.py` |
| Coût | 0,015 $/h | 0,04 $/h (EC2) |

---

# B. Kafka → Firehose (DirectPut) → S3 → Glue — la chaîne du schéma

## 1. La chaîne

```
producteur (sur l'EC2)                     20 000 ventes, 1 événement = 1 ligne JSON
        │
        ▼
Apache Kafka  sales.events                 3 partitions, KRaft, dans le compte
        │
        ▼
pont kafka_to_firehose                     PutRecordBatch, 500 max par requête
        │                                  chaque enregistrement se termine par \n
        ▼
Firehose  retail-sales-direct              DirectPut · tampon 60 s / 5 Mo · GZIP
        │
        ▼
s3://bucket/raw/sales/AAAA/MM/JJ/HH/*.gz  objets livrés par Firehose
        │
        ▼
Job Glue  retail-etl-raw-to-curated        union des lots + historique, dédup
        │
        ▼
s3://bucket/curated/sales/                 Parquet snappy
```

## 2. Pourquoi un pont, et pas une source Firehose « Kafka »

Firehose n'accepte que quatre familles de sources : **DirectPut, Kinesis, MSK,
Splunk**. Un Kafka auto-hébergé n'en fait pas partie. Avec MSK, Firehose
s'abonnerait nativement au cluster ; sans MSK, le `PutRecordBatch` est fait par
le pont. Le reste — tamponnage, compression, nommage, préfixe d'erreur — reste
assuré par le service, ce qui est bien le comportement décrit par la diapositive.

## 3. Commandes

```bash
# tout en un : instance + Kafka + dépendances + producteur + pont (≈7 min)
make kafka-all

# ou étape par étape
make kafka-up        # EC2 + Kafka, puis pip + dépôt dans /opt/retail
make kafka-run       # producteur 20 000 + pont vers Firehose
make kafka-status    # IP privée + état du conteneur
make kafka-down      # supprime l'instance et le SG

# 1. le delivery stream (idempotent)
make aws-firehose-up
make aws-firehose-status

# 3. attendre ~60 s (tampon Firehose), puis vérifier
aws s3 ls --recursive s3://$S3_BUCKET/raw/sales/

# 4. ETL
make aws-pipeline-run     # ou l'onglet Run du job Glue

# 5. arrêt
make aws-firehose-down
```

`make kafka-all` n'ouvre **aucun port** : les commandes sont exécutées sur
l'instance via **SSM** (`run_script`), donc ni SSH ni clé. Sur l'EC2, la variable
`SAGEMAKER_ROLE_ARN` n'est pas nécessaire — `firehose_direct._role()` retombe sur
`LabRole` du compte courant.

## 4. Points de conception

| Point | Choix | Raison |
|---|---|---|
| Délimiteur | `\n` ajouté par le pont | Firehose concatène sans séparateur ; sans lui l'objet devient `{...}{...}` et Spark échoue (`Unable to infer schema`) |
| Taille de lot | 500 enregistrements / 4 Mio | limites dures de `PutRecordBatch` |
| Tampon | 5 Mo / 60 s | laissé à Firehose, pas au code |
| Sémantique | *at-least-once* | commit Kafka **après** l'accusé Firehose → doublon possible, perte impossible |
| Déduplication | Glue, `(store_id, item_id, date)` | conséquence assumée de l'at-least-once |
| Erreurs | `raw/_errors/<type>/` | préfixe configuré sur le stream |
| Enregistrements refusés | jusqu'à 5 reprises, backoff exponentiel | `FailedPutCount` + `RequestResponses` |

## 5. Coût

| Poste | Tarif | Pour 20 000 événements (~8 Mo) |
|---|---|---|
| Firehose DirectPut | 0,029 $/Go | ~0,0003 $ |
| EC2 `t3.medium` (Kafka) | 0,0416 $/h | 0,04 $ pour une session d'une heure |

## 6. Résultat vérifié (session du 11/11)

| Étape | Mesure |
|---|---|
| Kafka | `sales.events`, 3 partitions, offsets `6779 + 6640 + 6581` = **20 000** |
| Producteur (EC2) | 20 000 messages en **3,9 s** |
| Pont → Firehose | **20 000/20 000** acceptés, **40** `PutRecordBatch`, **0 reprise** |
| Objet S3 | `raw/sales/2026/11/11/22/retail-sales-direct-1-2026-11-11-22-21-03-…gz`, 164 Ko |
| Contenu | **20 000 lignes / 20 000 JSON valides** |
| Glue | `SUCCEEDED` en 68 s |
| `curated/sales/` | 4 Parquet · **876 000 lignes** · **876 000 clés distinctes** |

Les 20 000 événements du flux dupliquent l'historique : Glue les supprime
(896 000 → 876 000). Cette égalité prouve que les deux branches — flux Kafka et
historique — arrivent bien dans le même `curated/`.

---

# A. Kinesis → Firehose (variante d'origine)

Conservée pour comparaison : elle montre une source **native** de Firehose
(`KinesisStreamAsSource`), mais elle n'a pas de Kafka, donc elle s'écarte du
schéma.

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
   s3://bucket/raw/sales/AAAA/MM/JJ/HH/<uuid>.gz
              │
              ▼
   Job Glue retail-etl-raw-to-curated            spark.read.json + historique
              │
              ▼
   s3://bucket/curated/sales/                    Parquet snappy
```

Chaque flèche correspond à un service managé : il n'y a **aucun processus** sur le
poste de travail une fois le déploiement fait.


## 2. Rôle de chaque brique

| Brique | Remplace | Rôle |
|---|---|---|
| **Lambda** `retail-replay-to-kinesis` | `ingest/producer.py` | relit le dataset depuis S3 et publie dans le stream ; checkpoint `(chunk, offset)` dans S3 pour reprendre après interruption |
| **Kinesis Data Streams** `retail-sales-stream` | Kafka / MSK | transport d'événements, ordre garanti par `PartitionKey` (= `store_id`) |
| **Firehose** `retail-sales-to-s3` | `ingest/to_s3.py` | bufferise (60 s ou 5 Mo), compresse en gzip, dépose dans `raw/sales/AAAA/MM/JJ/HH/` |
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
- **Partitionnement Firehose imposé.** Firehose range sous `AAAA/MM/JJ/HH/`.
  Le partitionnement dynamique (`!{timestamp:…}`) est **refusé** par cette API
  (« S3 Prefix should contain Dynamic Partitioning namespaces when Dynamic
  Partitioning is enabled »), même avec la syntaxe documentée. C'est pourquoi le
  job Glue active le listing récursif **et** prévoit un glob `*/*/*/*/*` en
  repli : sans cela Spark ne voit aucun fichier et renvoie 0 ligne.
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

La variante **B** (Kafka sur EC2 + Firehose DirectPut, section du haut) écrit au
même endroit, avec la même convention de clé : l'ETL et tout l'aval sont
identiques. Choisir A ou B ne change que le transport.
