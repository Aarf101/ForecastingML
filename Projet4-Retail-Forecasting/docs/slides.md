# Exposé idée projet — 5 slides

> Projet 4 · Retail ML Forecasting · binôme · MIT 2026
> Version imprimable : `docs/slides.pptx` (généré par `make slides`).
> Version éditable Canva/Google Slides : copier le contenu ci-dessous.

---

## Slide 1 — L'application et le besoin

**Titre** : Retail ML Forecasting — anticiper la demande, magasin par magasin

**Le besoin**
- Aujourd'hui : réassort à l'expertise, pas de prévision chiffrée ni d'intervalle de confiance
- Rupture de stock → ventes perdues et clients mécontents
- Surstock → trésorerie immobilisée et démarques
- Objectif : prévoir les ventes **J+1 à J+28** par (magasin, article), avec une incertitude exploitable

**Pour qui ?**
- Responsable approvisionnement / category manager → décisions de réassort
- Chef de rayon → priorisation des ruptures
- Direction → pilotage par KPI (taux de rupture, surstock, WAPE)

**L'application** : une console de pilotage de la demande (dashboard + agent conversationnel) qui répond en langage naturel et déclenche le réentraînement.

**Bandeau KPI** : `WAPE cible < 30 %` · `horizon 28 jours` · `1 200 séries suivies`

*Notes orateur (45 s)* : on part d'un constat terrain simple — le réassort se fait à l'expérience. On veut le transformer en décision chiffrée, avec une fourchette de risque, pas juste un chiffre moyen.

---

## Slide 2 — L'architecture retenue

**Titre** : Une architecture AWS en 8 étapes, du flux de caisse au dashboard

`![Architecture](diagrams/architecture.png)`

- **Ingestion** : événements de vente → Kafka/MSK → Firehose (micro-lot 60 s) → S3 `raw/`
- **Données** : S3 `raw/` → Glue ETL (nettoyage, déduplication) → S3 `curated/` Parquet, requêtable dans Athena
- **ML** : SageMaker Processing (features) → Training (LightGBM) + HPO → évaluation WAPE avec porte de qualité → Model Registry → endpoint + batch 28 j
- **IA & restitution** : agent LangGraph (Bedrock) qui interroge les prévisions et les documents → Streamlit / QuickSight
- **Industrialisation** : EventBridge + Step Functions (réentraînement quotidien), CloudWatch, versioning S3, tags de coût

*Notes orateur (60 s)* : insister sur les deux natures de flux — le batch quotidien (prévision 28 j) et le temps réel (endpoint pour les questions de l'agent). Et sur le fait que chaque brique coûteuse a un substitut gratuit prévu pour le Learner Lab (Kafka open-source, Ollama, Streamlit), ce qui est documenté dans `docs/architecture.md`.

---

## Slide 3 — Les données

**Titre** : Des jeux publics riches, un simulateur en plan B

**Sources**
- **M5 (Walmart, Kaggle)** : 30 490 séries, 1 941 jours, prix, promotions, calendrier, hiérarchie article/magasin (~450 Mo)
- **Rossmann (Kaggle)** : 1 115 magasins, 942 jours, promotions et état des magasins
- **Plan B — simulateur maison** : 876 000 lignes (20 magasins × 60 articles × 730 jours), même schéma que M5, saisonnalité/promos/ruptures contrôlées

**Ce qu'on en fait**
- Cible : ventes à `J+h`, h = 1…28 (direct multi-horizon)
- Features : lags 1/7/14/21/28/35, moyennes et écarts-types glissants 7/14/28, jour de semaine, semaine, fériés, promo, prix, lead time magasin
- Découpage **temporel** : dernier mois = test, mois précédent = validation
- Pièges traités : **3,6 % de jours en rupture** (ventes censurées) et doublons d'ingestion → déduplication sur (magasin, article, date) ; `stock_on_hand` exclu des features (fuite)

*Notes orateur (45 s)* : le point à faire passer, c'est qu'on sait déjà quel piège va nous mordre — la censure par rupture de stock et les doublons at-least-once de Kafka. Les deux sont traités explicitement.

---

## Slide 4 — Fonctionnalités et services

**Titre** : Ce que l'application offrira, et avec quels services

**Fonctionnalités**
- Prévisions 28 jours par magasin/article avec **P10 / P50 / P90** → stock de sécurité
- Tableau de bord : prévu vs réalisé, top dérives, liste des risques de rupture
- Agent conversationnel : « quel risque sur l'article I012 au magasin S03 la semaine prochaine ? » → chiffre, cause, action de réassort
- Brief hebdomadaire généré automatiquement (Markdown + e-mail) et alertes seuil

**Services mobilisés**
- Stockage & ingestion : **S3**, **Glue / Lambda**, **Kinesis / MSK**
- ML : **SageMaker** (Processing, Training, HPO, Pipelines, Registry, Endpoint, Model Monitor)
- IA : **Bedrock** + LangChain/LangGraph, RAG FAISS sur les documents promo et lead times
- Restitution : **QuickSight** (option production) ou **Streamlit**
- Orchestration & MLOps : **EventBridge**, **Step Functions**, **CloudWatch**

*Notes orateur (60 s)* : la valeur ajoutée n'est pas le modèle seul, c'est la boucle complète — la prévision alimente une décision, et l'agent explique la décision. C'est ce qui distingue le projet d'un simple notebook.

---

## Slide 5 — Preuve de faisabilité et suite

**Titre** : Déjà exécuté en local, le reste tient au crédit Learner Lab

**Résultats obtenus (sans AWS)**
- 876 000 lignes générées, **42 tests unitaires** verts
- **WAPE 0.430** (LightGBM + features) contre **0.568** pour la baseline « même jour de la semaine précédente » → **−24,3 %**
- Ingestion **Kafka → S3** validée : 2 000 événements → 4 objets gzip partitionnés
- Diagramme et documentation d'architecture prêts

**Prochaines étapes (7 semaines)**
- S1 : données + SageMaker Studio · S2 : conception du pipeline + S3
- S3 : ingestion + ETL · S4 : modélisation SageMaker + HPO
- S5 : agent LLM · S6 : dashboard · S7 : rapport, vidéo 2 min, soutenance

*Notes orateur (45 s)* : on arrive avec un pipeline qui tourne déjà, pas avec des slides. Le reste du temps sert à porter ce pipeline sur AWS et à soigner la démonstration.
