# LightGBM médian dans le pipeline principal

**Déployé et actif : v8.** Le workflow AWS complet a réussi, avec évaluation
indépendante à **40,46 % de WAPE**, RMSLE **0,5134** et couverture P10/P90
**79,22 %**. Les 19 200 prévisions publiées correspondent exactement à
l'artefact évalué, points et bornes ; le manifeste précédent v7 est préservé.
Il s'agit du même test historique que le benchmark, distinct de la réplication
synthétique indépendante. Aucun nouvel endpoint ni HPO n'a été exécuté.

Le profil par défaut des nouveaux entraînements est `median_calendar_v1`.
Il remplace le point central L2 par une régression quantile LightGBM avec
`alpha=0.5`, arrêt anticipé sur L1, et calendrier cible enrichi. Le WAPE
repose sur l'erreur absolue : le point médian correspond à cet objectif.
Il ne représente plus une espérance de ventes.

Les mêmes paramètres structurels sont conservés : 400 arbres maximum,
learning rate 0,05, 63 feuilles et patience de 40 tours dans le pipeline AWS.
Les 16 horizons possèdent chacun un booster médian et deux boosters P10/P90,
soit 48 boosters. Les bornes sont ajustées avec leur propre objectif quantile
et leur propre alpha ; elles ne sont pas copiées de v7.

## Contrat commun

- `train/profiles.py` définit les paramètres et le calendrier du profil.
- Le Training Job applique le calendrier après construction de chaque cible,
  y compris quand les lags ont été précalculés par Processing.
- L'évaluation indépendante relit `feature_profile` dans l'artefact et
  reconstruit les mêmes colonnes avant de calculer ses métriques.
- Le Batch Transform et l'appel d'endpoint utilisent le même générateur de
  lignes : le calendrier est recalculé pour chaque horizon, y compris au
  passage décembre/janvier. Aucun prix ni promotion futur observé n'est lu.
- `metadata.json` conserve `feature_profile`, `point_statistic`, les paramètres,
  les colonnes et le vocabulaire. Le manifeste actif expose aussi le profil
  et la statistique centrale ; le schéma des prévisions reste compatible.
- Un artefact sans profil est traité comme `legacy_l2`. Les anciens modèles
  utilisent uniquement leurs colonnes historiques ; leurs prédictions ne sont
  pas converties artificiellement en médianes.

Les jours fériés de ce profil sont ceux du simulateur. Un déploiement sur
un autre jeu exige un calendrier métier adapté et une nouvelle évaluation.
Stock et rupture restent exclus des features. Le découpage temporel purgé
de la production est conservé.

## Utilisation

```bash
make test
make train
make aws-workflows-up
make aws-pipeline-run
```

`make sagemaker-train`, `make sagemaker-pipeline-run` et les nouveaux essais HPO
utilisent également le profil médian. Pour réentraîner explicitement le témoin
historique sans publication automatique :

```bash
.venv/bin/python -m retail_forecast.train.train_lgbm --model-profile legacy_l2
.venv/bin/python -m infra.sagemaker.launch --model-profile legacy_l2 --wait
```

La publication normale reste : évaluation Processing indépendante, gate WAPE,
registre, Batch Transform candidat, contrôle des 19 200 lignes, puis écriture
conditionnelle de `deployment/active.json` conservant le précédent manifeste.
Un échec conserve le modèle actif précédent. Les horaires restent désactivés.

## Preuves

- [Expérience de sélection et test synthétique indépendant](median_calendar_benchmark.md).
- [Tests d'intégration](../reports/median_integration_tests.xml) : deux profils,
  données brutes ou prétraitées, véritable entraînement des 16 horizons,
  réévaluation indépendante, serving ponctuel et intervalles.
- [Exécution AWS et manifeste avant/après](evidence/median_aws_integration.json).

Les chiffres de l'expérience sur réplication indépendante ne remplacent pas
les métriques de cette nouvelle exécution AWS. Le statut terminal et le
manifeste dans la preuve AWS déterminent si la version a effectivement été activée.
