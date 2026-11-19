# Revue du projet 4 - 19 novembre 2026

> Suite de la revue : les corrections de workflows, de promotion et des
> horizons sont décrites dans [learner_lab_workflows.md](learner_lab_workflows.md).
> Le tableau des écarts ci-dessous décrit le constat initial, avant cette suite.

## Conclusion

Le dépôt contient une base exploitable : ingestion Kafka/Firehose, ETL,
LightGBM, jobs SageMaker, registre, Batch Transform, Streamlit et rapport LLM.
Les défauts les plus importants concernent la validité de l'évaluation et le
contrat entre entraînement et inférence. Une suite verte de 87 tests ne les
couvrait pas. Cette revue corrige ces défauts et ajoute des régressions ; elle
ne constitue pas une validation du compte AWS actuel.

Sources : `1 Déroulement Cours Projet CL ML 2026.pdf`, pages 9-14, et les deux
diagrammes de `aws presentation.pdf`, inspectés visuellement. Le projet 4
demande des ventes S3/Kafka, une prévision SageMaker et un dashboard dynamique.
Le cadre général ajoute MLOps, intégration LLM et livrables de démonstration.

## Corrections réalisées

| Défaut constaté | Correction |
|---|---|
| Split par origine seulement : les cibles futures de train traversaient le début de validation ; celles de validation traversaient le test | Conservation de `target_date`, purge des cibles à chaque frontière ; aucune donnée d'une période suivante dans les labels de la période précédente |
| `lag_7` utilisé pour prédire `units[t+h]`, sans aligner le jour de semaine | Baseline sur la dernière date de même jour de semaine que la cible, connue à l'origine : `t+h-7*ceil(h/7)` |
| Cible décalée par position de ligne, donc dépendante de l'ordre des partitions | Recherche par `(store_id, item_id, date+h)` |
| Les lags par lignes pouvaient représenter plusieurs jours en présence de trous | Rejet explicite des séries non quotidiennes avant calcul ; pas d'imputation silencieuse à zéro |
| Calendrier de l'origine seulement | Ajout du jour de semaine, jour et mois de la cible ; recalcul par horizon à l'inférence |
| Encodages catégoriels recalculés sur le jeu d'inférence | Vocabulaires enregistrés dans `metadata.json` et relus par Batch Transform, la démo endpoint et l'évaluation sur données brutes ; catégorie inconnue encodée -1 |
| Évaluation indépendante dégradée en simple copie de métriques, capable de passer le gate | Échec de Processing si le recalcul est absent, partiel ou non fini ; approbation automatique du pipeline bloquée |
| Évaluation utilisant toujours 56/28 jours, même avec d'autres paramètres d'entraînement | Fenêtres enregistrées dans l'artefact et réutilisées par l'évaluateur |
| Rapport d'évaluation S3 écrasé par les exécutions suivantes | Préfixe par exécution et URI attachée à la version du registre |
| Gate Lambda limité à J+1, contrairement au gate du pipeline | Utilisation de `mean_wape` si disponible ; blocage des WAPE non finis ou négatifs ; seuil 0 respecté |
| Entrée et sortie Batch Transform partagées entre jobs | Entrée par job et lecture du fichier de sortie exact du job ; aucune collecte des anciennes sorties |
| Dashboard lisant un nom de Parquet unique incompatible avec les partitions Glue | Lecture paginée des partitions ; client S3 commun avec l'ingestion, compatible avec s3mock |
| Sélecteur du dashboard supposant toujours J+1 et J+7 | Choix et titre déduits des horizons réellement présents |
| Rapport LLM confondant volumes élevés et volatilité, métriques absentes remplacées par 0 | Libellés corrigés, qualité absente explicitement signalée, horizons décrits comme des dates ponctuelles ; options CLI Ollama appliquées |
| README annonçant un agent/RAG, des quantiles et des statuts périmés | Statuts rectifiés, anciens résultats identifiés comme historiques ; SDK optionnel contraint à la v2 utilisée par le code |

## Validation reproductible

```bash
make test
.venv/bin/python -m retail_forecast.train.train_lgbm \
  --sample 0.15 --out reports/review_baseline.json
```

107 tests passent dans l'environnement fourni, sans service Docker, AWS ou
LLM. Les nouveaux tests incluent un vrai petit entraînement LightGBM à deux
horizons, sauvegarde/rechargement du modèle, inférence et reproduction du WAPE
par l'évaluateur indépendant avec des fenêtres 35/14 jours. Un test Streamlit
charge l'application avec des partitions Glue simulées. Les tests optionnels
du dashboard et du SDK sont ignorés si leurs dépendances ne sont pas installées.

Le benchmark est réalisé sur les données synthétiques existantes, 15 % des
séries choisies de façon reproductible (seed 42), soit **180 séries** :

| Mesure | Valeur |
|---|---:|
| Horizon | J+1 |
| Train / validation / test | 109 620 / 9 900 / 5 040 lignes |
| WAPE LightGBM | 0,426808 |
| WAPE baseline corrigée | 0,567268 |
| Gain relatif | 24,76 % |
| RMSE LightGBM | 3,927355 |
| Biais relatif LightGBM | -0,718 % |

Ces chiffres ne sont ni un benchmark sur données réelles ni une preuve de
qualité sur tous les horizons. Il s'agit d'une évaluation à origines glissantes :
chaque ligne de test utilise l'historique connu à sa propre origine. Elle ne
simule pas un unique lancement de 28 prévisions à une origine fixe.
Les résultats historiques du rapport et du README ne doivent pas être comparés
directement sans harmoniser échantillon, baseline et protocole.

## Écarts restant à traiter avant le rendu

| Exigence ou architecture | État et conséquence |
|---|---|
| Horizon de 16 jours dans `aws presentation.pdf` | Le code conserve les horizons historiques 1, 7, 14, 28, selon la décision du dépôt. Ce sont quatre dates, pas une prévision quotidienne complète. Pour livrer J+1 à J+16, entraîner puis inférer les 16 modèles correspondants et aligner configuration, documentation et dashboard |
| Pipeline mensuel d'entraînement et pipeline quotidien d'inférence distincts | Les scripts existent, mais cette revue n'a pas démontré deux workflows AWS planifiés complets. La state machine d'entraînement s'arrête au registre ; un script de lancement complet n'est pas une preuve d'orchestration managée de bout en bout |
| Modèle actif mis à jour uniquement après réussite complète | Batch Transform choisit le dernier package approuvé. Cela ne prouve pas l'existence d'un pointeur actif transactionnel ni une validation de l'inférence avant promotion |
| Gate indépendant partout | Le SageMaker Pipeline impose le recalcul. Le chemin autonome `register.py` et la Lambda Step Functions lisent encore les métriques de l'artefact d'entraînement ; ils ne sont pas équivalents au pipeline d'évaluation indépendant |
| LangChain/LangGraph dans le cadre général du cours | Le rapport LLM utilise HTTP Ollama ou SageMaker directement. Il ne démontre pas cette exigence d'intégration. Conserver la décision « pas d'agent autonome », puis choisir une intégration de prompts adaptée si cette exigence est évaluée |
| Hébergement Streamlit sur SageMaker | Le code et un test applicatif existent ; aucun déploiement ni état actuel du notebook n'a été vérifié pendant cette revue. Le paquet du projet doit être installé, le mode ancien « fichier seul » ayant été retiré au profit de la configuration commune |
| WAPE, biais et RMSLE du diagramme | WAPE, biais et RMSE disponibles ; RMSLE non implémenté |
| Quantiles et stock de sécurité | Pas de modèles P10/P90 livrés. Une forte variabilité historique ne constitue pas un quantile de prévision calibré |
| Données réelles | Le chargeur adapte les schémas ; cela ne remplace pas la préparation des jeux M5/Favorita, les jointures de prix/promotions ni le traitement des jours absents. Le benchmark est synthétique |
| Rapport, oral 10 min, vidéo 2 min | Sources de rapport et scripts présents ; aucun nouvel export PDF, enregistrement vidéo ou contrôle de présentation effectué ici |

Les ventes restent une mesure censurée de la demande lors des ruptures. Exclure
les colonnes de stock évite leur usage comme features, mais ne reconstitue pas
la demande perdue. Ne pas présenter le modèle comme une estimation démontrée
de la demande non contrainte.

## Reprise cloud

Réentraîner avec ce code avant de republier les métriques ou de réutiliser les
nouveaux vocabulaires. Les anciens artefacts restent lisibles par repli sur les
anciens champs, mais ne disposent pas des garanties ajoutées. Une évaluation
indépendante nécessite exactement les données et la définition de features
ayant servi à l'entraînement ; comparer un ancien modèle à des données
actualisées n'est pas une reproduction de son ancien test.

Aucune commande de provisionnement, aucun changement du `.env` et aucun appel
AWS réel n'ont été effectués pendant cette revue. Les tests utilisent des
clients simulés. Les avertissements de dépréciation du SDK SageMaker v2 et de
kafka-python restent à traiter lors d'une migration dédiée.
