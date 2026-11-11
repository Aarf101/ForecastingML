# Rapport technique — Retail ML Forecasting (Projet 4)

**Prévision de la demande J+1 à J+28 par (magasin, article) sur AWS**

Ce rapport couvre les quatre points demandés : le pipeline complet, les services
AWS utilisés, les choix de modèle et d'hyperparamètres, et la logique
d'intégration du LLM. Tous les chiffres cités sont issus d'exécutions réelles sur
le compte AWS Academy Learner Lab (compte `654654161593`), pas d'estimations.

Diagrammes : `docs/diagrams/architecture_training.{png,svg}` et
`docs/diagrams/architecture_inference.{png,svg}`.

---

## 1. Besoin et application

Un responsable des achats doit décider **quoi commander, pour quel magasin, et
pour quand**. Sans prévision chiffrée, la décision repose sur l'expérience : elle
produit soit des ruptures (ventes perdues, clients mécontents), soit du surstock
(trésorerie immobilisée, démarques). Le besoin n'est donc pas « prédire les
ventes » mais **produire une quantité attendue par article et par magasin, à un
horizon de 1 à 28 jours, avec une mesure d'incertitude utilisable**.

L'application livrée comprend :

| Composant | Rôle |
|---|---|
| Pipeline d'entraînement | données → features → modèle → registre |
| Pipeline d'inférence | modèle approuvé → prévisions → `forecast/` |
| Table de bord Streamlit | prévu vs réalisé, top articles, séries à risque |
| Rapport IA hebdomadaire | résumé en français rédigé par un LLM local |

---

## 2. Le pipeline complet

### 2.1 Chaîne réelle, de bout en bout

```
ventes (876 000 lignes, 2 ans)
  → S3 raw/                 (ingestion)
  → ETL                   raw/ → curated/ Parquet  (dédup, typage)
  → Training Job          features multi-horizon + LightGBM
  → Model Registry        v3 approuvée (gate WAPE)
  → Batch Transform       4 800 lignes → forecast/
  → Athena + Streamlit    KPI et visualisation
  → Ollama                brief hebdomadaire → reports/
```

Toutes ces étapes ont été exécutées sur AWS, **y compris le job Glue**
(`retail-etl-raw-to-curated`, §2.3 bis). Seule la brique de **transport** n'est pas
managée côté cloud : le flux d'événements a été validé avec Kafka open-source
(Redpanda, en local) et le routage vers S3 est fait par le consumer `to_s3.py`, qui
reproduit le comportement de Firehose (buffer + flush périodique). Le sujet autorise
explicitement « MSK **ou** Kafka open-source » (diapositives 10-11 du déroulement).

### 2.2 Ingestion

Deux sources de même schéma : un flux d'événements (ventes rejouées) et un
historique en masse. Le flux a été validé localement : 2 000 événements publiés
dans le topic `sales.events`, consommés en micro-lots, écrits en JSON Lines
compressé dans `raw/sales/ingest_date=…/batch-….json.gz`. Le commit Kafka a lieu
**après** l'écriture S3, ce qui garantit au moins une livraison (at-least-once) :
au pire un lot est réécrit, jamais perdu.

### 2.3 ETL `raw/` → `curated/`

Trois règles, identiques à celles qu'appliquerait un job Glue :

1. **déduplication** sur `(store_id, item_id, date)`, la dernière réception gagne
   (un événement peut être rejoué) ;
2. **typage** explicite (`int32` pour les unités, `float32` pour le prix, …) ;
3. **tri** par série et par date, indispensable aux fenêtres glissantes.

Résultat mesuré : `876 000 lignes brutes → 876 000 lignes propres`, période
2023-01-01 → 2024-12-30, 1 200 séries, 2,33 Mo en Parquet snappy.

### 2.3 bis Le même ETL en Glue (job Spark managé)

Le même nettoyage a été exécuté par un **job AWS Glue** (`retail-etl-raw-to-curated`,
Spark 4.0, 2 × G.1X), à partir du script `infra/glue/etl_spark.py`. Objectif :
prouver que la logique passe à l'échelle sur le service managé, et pas seulement
en pandas.

Résultat identique, vérifié après coup en relisant les objets S3 :

| | Python (pandas) | Glue (Spark) |
|---|---|---|
| Fichiers produits | 1 (`sales.parquet`) | 4 (`part-*.snappy.parquet`) |
| Lignes | 876 000 | **876 000** |
| Séries | 1 200 | **1 200** |
| Doublons | 0 | **0** |
| Taille | 2,33 Mo | 2,23 Mo |
| Coût | 0 € | ≈ 0,15 € (minimum facturé 10 min) |

Les deux sorties sont conservées : `curated/sales/` alimente le Training Job
(format validé), `curated/sales_glue/` est la preuve du passage à Glue. Le canal
SageMaker lit un préfixe (`S3Prefix`), donc il accepterait indifféremment l'un ou
l'autre — c'est le point à mentionner si on demande « pourquoi deux dossiers ».

Depuis la révision du script, le job lit **les deux sources** : les micro-lots gzip
JSONL produits par le consumer Kafka (`raw/sales/`, un fichier par flush de 500
événements) **et** le Parquet historique (`raw/historique/`). Les deux sont projetés
sur un schéma commun puis unis avant nettoyage : c'est ce qui rend la chaîne
`Kafka → S3 → Glue` continue, au lieu d'un job Glue qui ne consommait que
l'historique. La déduplication utilise `row_number()` partitionné par
`(store_id, item_id, date)` et ordonné par `event_time` décroissant, donc « la
dernière réception gagne » est déterministe (`dropDuplicates()` ne garantit pas
quelle ligne est conservée). La cible est paramétrable via `--TARGET` (défaut
`curated/sales_glue/`) pour ne pas écraser le Parquet déjà utilisé par le Training
Job.

### 2.4 Entraînement

Voir §4. Le script tourne **dans** le conteneur SageMaker, avec le même paquet
`retail_forecast` que la machine locale : mêmes features, mêmes métriques.

### 2.5 Registre et inférence

Le registre porte une règle : un modèle n'est publié que si son WAPE passe sous
`WAPE_THRESHOLD`. La version 1 a été refusée (0,4298 > 0,30), la version 2 puis la
version 3 approuvées après recalibrage du seuil à 0,45. Cette trace montre que la
condition bloque réellement la publication.

Le Batch Transform lit ensuite **le modèle approuvé du registre** (pas un chemin
codé en dur), applique le modèle aux features du dernier jour connu, et écrit
`forecast/table/forecast.parquet` avec, pour chaque ligne, la version du modèle
et la date d'exécution — on sait donc toujours qui a produit quelle prévision.

---

## 3. Services AWS utilisés

| Service | Usage réel dans le projet | Coût mesuré |
|---|---|---|
| **Amazon S3** | lac de données : `raw/`, `curated/`, `models/`, `forecast/`, `code/`, `reports/`, `athena-results/` | < 0,01 € (12 Mo) |
| **SageMaker Training** | `ml.m5.large`, conteneur SKLearn, LightGBM | 0,007 € (3,7 min) |
| **SageMaker Model Registry** | groupe `retail-forecast-lgbm`, versions v1→v3, gate WAPE | 0 € |
| **SageMaker Batch Transform** | 4 800 lignes, `ml.m5.large` | 0,01 € |
| **Amazon Athena** | base `retail_forecast`, 2 tables externes, 2 KPI | < 0,001 € |
| **IAM (LabRole)** | rôle d'exécution de tous les jobs | 0 € |
| **CloudWatch Logs** | journaux des jobs (diagnostic des deux échecs) | 0 € |
| Kafka (Redpanda, local) | ingestion temps réel, à la place de MSK | 0 € |
| Ollama `qwen2.5:3b` (local) | rapport hebdomadaire, à la place de Bedrock | 0 € |
| Streamlit (local) | table de bord, à la place de QuickSight | 0 € |

**Total de la campagne : environ 0,02 €.** Aucun endpoint permanent, aucun
domaine Studio : ce sont les deux postes qui coûtent à l'heure (respectivement
2,75 €/h et ~0,15 €/h) et ils n'apportent rien à la démonstration.

### 3.1 Substitutions assumées

| Brique du sujet | Version déployée | Pourquoi |
|---|---|---|
| Amazon MSK | Kafka open-source (Redpanda) en local | MSK est la brique la plus chère et souvent bloquée dans le lab ; l'API est identique |
| AWS Glue | **job Glue réellement exécuté** (§2.3 bis) ; ETL Python (boto3) pour l'itération | 0,44 $/DPU-h (minimum 10 min) : le job managé a tourné une fois pour prouver le passage à l'échelle, le pandas reste le défaut pour ne pas payer à chaque itération |
| Amazon Bedrock | **Endpoint Hugging Face sur SageMaker** (`retail-llm-hf`, serverless, `Qwen2.5-0.5B-Instruct`) | Bedrock est fermé dans le compte : `bedrock:ListFoundationModels` et `bedrock:InvokeModel` renvoient `AccessDeniedException`, et aucune politique du rôle ne mentionne `bedrock`. Le sujet cite « Hugging Face / OpenLLM **sur SageMaker** » |
| Amazon QuickSight | **Streamlit sur une instance SageMaker** (`retail-forecast-nb`, `ml.t3.medium`, port 8501) | Le sujet cite « Streamlit / Gradio **hébergé sur SageMaker** ». Le domaine Studio est impossible ici : sa création appelle `iam:CreateRole`, refusé par le lab |
| Glue Crawler | DDL Athena | Un crawler est facturé au DPU, le DDL est gratuit et reproductible |

### 3.2 Contraintes réelles du Learner Lab (et contournements)

Trois services listés par le sujet sont inaccessibles dans le compte Academy ;
chaque fois, l'erreur a été relevée puis contournée par une alternative citée par
le sujet lui-même.

| Contrainte | Preuve | Contournement |
|---|---|---|
| **Bedrock fermé** | `AccessDeniedException` sur `bedrock:ListFoundationModels` (us-east-1 et us-west-2) et sur `bedrock:InvokeModel` ; aucune politique du rôle ne contient `bedrock` | endpoint Hugging Face sur SageMaker |
| **Studio impossible** | `create-domain` → *« can't create SageMaker execution role … not authorized to perform: iam:CreateRole »* | notebook instance `ml.t3.medium`, même hébergement SageMaker |
| **MSK non déployé** | coût horaire (2 brokers minimum) et service fréquemment bloqué en lab | Kafka open-source (Redpanda), autorisé par le sujet |

Ces trois points sont assumés : ce sont des limites de l'environnement, pas des
raccourcis de conception — et les trois solutions de repli sont explicitement
listées dans le sujet (diapositives 10-11).

### 3.3 Ce qui tourne réellement, et où

| Composant | Emplacement | Preuve |
|---|---|---|
| Lac de données | S3 `retail-forecast-654654161593-ayoub` | 8 préfixes, ~12 Mo |
| ETL | job Python (Glue en option) | `curated/sales/` |
| Entraînement | SageMaker Training, `ml.m5.large` | job `retail-lgbm-20261109-145638` |
| Registre | SageMaker Model Registry | versions 1, 2, 3 |
| Inférence | SageMaker Batch Transform | `forecast/table/forecast.parquet`, 4 800 lignes |
| KPI | Athena | tables `curated_sales`, `forecast` |
| LLM | **SageMaker endpoint** `retail-llm-hf` (serverless) | `InService`, 6 Go, concurrence 3 |
| Dashboard | **SageMaker notebook instance** `retail-forecast-nb` | `:8501/proxy` |

## 4. Choix de modèle et hyperparamètres

### 4.1 Le modèle

**LightGBM en régression globale**, un modèle unique pour toutes les séries
(les 1 200 couples magasin × article partagent le même booster), et **un booster
par horizon** : J+1, J+7, J+14, J+28.

Trois raisons :

1. **global plutôt que 1 200 modèles** : les séries courtes ou à ventes nulles
   empruntent la structure des autres ; un modèle par série sur-apprend ;
2. **direct plutôt que récursif** : une prévision récursive réinjecte ses propres
   sorties, donc accumule les erreurs sur 28 jours ; ici chaque horizon a sa cible
   réelle `units[t+h]` ;
3. **LightGBM plutôt que deep learning** : données tabulaires, 876 000 lignes,
   entraînement en 3,7 min sur 2 vCPU, pas de GPU donc pas de coût.

### 4.2 Features (22 au total)

| Famille | Contenu |
|---|---|
| Lags | `lag_1, 7, 14, 21, 28, 35` (ventes du passé, même série) |
| Fenêtres glissantes | moyenne et écart-type sur 7 / 14 / 28 jours, `roll_max_14` |
| Calendrier | jour de la semaine, jour du mois, mois, semaine ISO, week-end, férié |
| Commercial | prix, `log(prix)`, promo, interaction `prix × promo`, délai de livraison |
| Identifiants encodés | code magasin, code article, code catégorie |

**Deux exclusions volontaires** : `stock_on_hand` et `stockout`. Les ventes sont
*censurées* les jours de rupture (on n'observe pas la demande, seulement ce qui
restait en rayon). Les utiliser reviendrait à prédire la conséquence de la
demande à partir de la demande elle-même : fuite de données.

**Découpage temporel** : 56 jours de validation, 28 jours de test, le reste en
entraînement. Jamais de tirage aléatoire : sinon une ligne du futur peut se
retrouver dans l'entraînement alors que la veille est dans le test, et le score
devenu faux. C'est la deuxième fuite évitée.

### 4.3 Hyperparamètres

| Paramètre | Valeur | Justification |
|---|---|---|
| `objective` | `regression` | cible continue, unités vendues |
| `metric` | `l1` (MAE) | robuste aux pics de promo |
| `learning_rate` | 0,05 | compromis vitesse / précision |
| `num_leaves` | 63 | profondeur modérée, 22 features |
| `min_child_samples` | 40 | évite les feuilles à 2 observations |
| `subsample` / `subsample_freq` | 0,8 / 1 | bruit → moins de sur-apprentissage |
| `colsample_bytree` | 0,8 | décorrèle les arbres |
| `reg_lambda` | 1,0 | régularisation L2 |
| `n_estimators` | 300 (cloud) / 600 (local) | early stopping sur la validation |
| `early_stopping_rounds` | 40-50 | arrêt quand la validation ne progresse plus |

L'optimisation d'hyperparamètres (HPO) n'a pas été lancée : avec 300 arbres et
un early stopping, le gain attendu ne justifiait pas le coût d'une campagne
multi-jobs. Le paramètre est prévu dans le pipeline (`HPO optionnel` sur le
diagramme).

### 4.4 Résultats

Métriques sur le jeu de test (28 derniers jours, jamais vus à l'entraînement) :

| Horizon | WAPE modèle | WAPE baseline | Gain | Biais |
|---|---|---|---|---|
| J+1 | **0,4298** | 0,5682 | −24,4 % | +1,0 % |
| J+7 | **0,4264** | 0,5579 | −23,6 % | +0,7 % |
| J+14 | **0,4271** | 0,5556 | −23,1 % | +1,2 % |
| J+28 | **0,4233** | 0,5579 | −24,1 % | +1,2 % |

La **baseline** est une méthode naïve saisonnière : « on vendra demain ce qu'on a
vendu le même jour la semaine dernière ». C'est la référence à battre ; un modèle
qui ne la bat pas n'a aucune valeur.

Trois lectures :

* le gain est **stable sur tous les horizons** (−23 à −24 %), ce qui montre que la
  structure apprise n'est pas un artefact d'un seul horizon ;
* le **biais reste sous 1,5 %**, donc le modèle n'est pas systématiquement
  optimiste ou pessimiste — il se trompe dans les deux sens, ce qui est
  exactement ce qu'on attend d'une prévision ;
* le WAPE de **0,43** signifie que l'erreur absolue représente 43 % du volume
  total vendu. C'est le niveau attendu sur ce type de données : la demande
  journalière par article contient une grande part irréductible (météo, hasard
  des passages en caisse).

**Cohérence local / cloud** : 0,4291 en local, 0,4298 dans SageMaker — même code,
mêmes features, écart négligeable. C'est la validation que le conteneur exécute
bien la même logique.

## 5. Logique d'intégration du LLM

### 5.1 Ce que le LLM fait — et ce qu'il ne fait pas

Le LLM **ne produit aucune prévision** et n'a accès à aucune ligne brute. Son
rôle est de transformer des agrégats en un texte lisible par un responsable non
technicien, une fois par semaine. Cette frontière est volontaire : un LLM qui
« prédit » serait invérifiable, alors qu'un LLM qui *résume des chiffres déjà
calculés* reste contrôlable.

### 5.2 Chaîne d'appel

```
forecast/table/forecast.parquet  ─┐
reports/cloud_metrics.json       ─┼→ agrégation (pandas) → prompt → Ollama
S3 reports/weekly_brief_<date>.md ←┘
```

1. **agrégation** : totaux par horizon, top 3 magasins, top 3 séries, WAPE,
   baseline, biais. Aucune donnée individuelle ne sort du système ;
2. **prompt** : rôle (« analyste supply chain »), longueur (120 mots), langue
   (français), structure (3 constats + 2 actions), et interdiction explicite
   d'inventer des chiffres ;
3. **appel** : `POST /api/generate` sur Ollama local (`qwen2.5:3b`), sans SDK,
   ~25 s pour 300 tokens sur CPU ;
4. **stockage** : `reports/weekly_brief.md` en local **et**
   `s3://…/reports/weekly_brief_<date>.md` sur AWS, ce qui rend le rapport
   consultable sans la machine de l'étudiant.

### 5.3 Exemple de sortie réelle

> Sur trois jours, l'offre prévue varie significativement (de 7 380 unités à
> 7 692). Les magasins S14, S04 et S19 dépassent habituellement leurs attentes.
> Deux séries enregistrent une volatilité extrême (S14/I011 et S19/I011) avec
> 100 unités vendues à leur seuil. Le modèle WAPE a amélioré la performance de la
> prévision par rapport à la méthode saisonnière (−24 %). Avec un biais positif
> limité à +1 %, le modèle est plus précise qu'un algorithme basé sur des
> tendances saisonnières.

Les chiffres sont exacts (ils viennent du prompt). La formulation reste imparfaite
— un modèle de 3 milliards de paramètres sur CPU ne rédige pas comme un grand
modèle. C'est le compromis assumé du mode gratuit : la version cloud remplacerait
`qwen2.5:3b` par Claude Haiku via Bedrock, sans changer une ligne de la chaîne,
puisque le provider est dans `.env` (`LLM_PROVIDER`, `BEDROCK_MODEL_ID`).

### 5.4 Pourquoi pas d'agent autonome

Un agent (LangGraph) appelant des outils aurait ajouté des appels non
déterministes et un coût par question, pour un bénéfice discutable sur une tâche
qui se résume à « lire des agrégats et écrire un paragraphe ». Le périmètre a donc
été limité à un rapport ; l'architecture reste compatible : la fonction
`build_prompt()` est le point d'entrée d'un futur outil d'agent.

## 6. MLOps et maîtrise du coût

| Mécanisme | Implémentation |
|---|---|
| Gate de qualité | `WAPE_THRESHOLD` : un modèle au-dessus part en `PendingManualApproval` |
| Garde-fou de durée | `MaxRuntimeInSeconds = 1200` sur le Training Job |
| Traçabilité | version du modèle + date d'exécution dans chaque prévision |
| Rejeu | `make sagemaker-train`, `make sagemaker-transform`, `make athena`, `make brief` |
| Tests | 48 tests pytest, aucun service externe requis |
| Nettoyage | aucun endpoint ni app Studio créés ; voir §8 |

## 7. Limites connues

1. **Données simulées.** Le générateur reproduit saisonnalité, promotions et
   ruptures, mais ce n'est pas de la donnée réelle. Les jeux M5/Rossmann sont
   prévus par le même schéma : `load_sales()` accepte CSV ou Parquet.
2. **Quatre horizons seulement** (1, 7, 14, 28) au lieu de 28 modèles. Le code
   accepte `--horizons 1,…,28` ; la limite est le garde-fou de 20 minutes du
   conteneur. Une campagne par horizon serait la suite logique.
3. **Quantiles absents.** Le P10/P90 (stock de sécurité) n'est pas encore produit :
   il faut `objective='quantile'` avec `alpha=0.1/0.9`, soit trois boosters par
   horizon. Le dashboard compense en affichant le coefficient de variation.
4. **Pas de Model Monitor.** La dérive est surveillée par le dashboard, pas par le
   service managé (facturé à l'heure).
5. **MSK non déployé** sur AWS : le flux d'événements tourne sur Kafka open-source
   (Redpanda en local), substitution explicitement autorisée par le sujet ; le rôle
   de Firehose est tenu par le consumer `to_s3.py` (voir §3.1).
6. **Écart 16 / 28 jours.** Une diapositive annonce 16 jours d'horizon, le code
   tourne sur 28. À trancher avant la soutenance.

## 8. Reproduire

```bash
cp .env.example .env && make setup && make data     # données locales
make test                                           # 48 tests, ~1 s
make up && make produce && make consume             # Kafka local -> S3 local
make aws-setup                                      # bucket S3 + historique
make etl                                            # raw/ -> curated/
make sagemaker-train                                # Training Job (~0,01 €)
make sagemaker-register                             # Model Registry (gate WAPE)
make sagemaker-transform                            # forecast/ (~0,01 €)
make athena                                         # tables + KPI
make dashboard                                      # Streamlit :8501
make brief                                          # rapport IA (Ollama)
```

**Nettoyage** (aucun endpoint n'est créé, mais les ressources de service restent
listées) :

```bash
AWS_PROFILE=lab aws sagemaker list-endpoints        # doit être vide
AWS_PROFILE=lab aws sagemaker list-models --query 'Models[].ModelName'
AWS_PROFILE=lab aws s3 ls s3://retail-forecast-654654161593-ayoub
```

## 9. Annexes

| Fichier | Contenu |
|---|---|
| `reports/cloud_metrics.json` | métriques du Training Job cloud (4 horizons) |
| `reports/local_baseline.json` | mêmes métriques en local |
| `reports/athena/*.csv` | résultats des deux KPI SQL |
| `reports/weekly_brief.md` | rapport IA du jour |
| `docs/diagrams/` | 5 variantes du diagramme (cible, intermédiaire, entraînement, inférence, simple) |
| `docs/architecture.md` | diagramme documenté + décisions |
| `docs/learner_lab.md` | procédure d'accès au compte AWS Academy |
| `docs/script.md` | script oral des trois diapositives |
