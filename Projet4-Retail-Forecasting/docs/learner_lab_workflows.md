# Workflows Learner Lab corrigés

Le parcours recommandé est désormais `infra.aws.workflows`, puis
`infra.sagemaker.lifecycle`. L'ancien parcours `infra.aws.orchestrate` reste
compatible, mais son gate et `register.py` ne peuvent plus approuver automatiquement
sur la seule foi des métriques d'entraînement : ils produisent PendingManualApproval.

## Contrat

- `retail-training` : Glue -> Processing pilote -> SageMaker Pipeline
  (features immuables par exécution -> training -> évaluation indépendante ->
  gate -> registre) -> Batch Transform candidat -> validation -> publication.
- `retail-inference` : Glue -> Processing pilote -> modèle du manifeste actif ->
  features -> Batch Transform -> validation -> publication.
- Chaque modèle couvre exactement les 16 jours J+1 à J+16. Les anciens modèles
  à quatre horizons ne peuvent pas être activés par ce parcours.
- Le manifeste `deployment/active.json` lie le package, son artefact et le fichier
  Parquet validé. Le dashboard et le rapport LLM lisent ce fichier.
- Les résultats restent sous `forecast/runs/<run>/forecast.parquet`. Une écriture
  S3 conditionnelle du manifeste interdit d'écraser une publication concurrente.
  L'échec de calcul, un résultat non fini/négatif, une couverture incomplète des séries présentes
  à la dernière origine ou
  un conflit laisse le manifeste précédent intact. Celui-ci est conservé dans
  `previous`. Le registre seul ne déclenche pas la promotion.
- Les horaires EventBridge sont définis en UTC : premier du mois à 02:00 pour
  training, chaque jour à 04:00 pour inference. Ils sont désactivés au déploiement.

## Déploiement et validation

```bash
make test
make aws-workflows-print
make aws-workflows-up
```

Puis dans la console Step Functions du Learner Lab :

1. Lancer `retail-training` avec `{}`. Observer Glue, le job Processing,
   le Pipeline SageMaker et son évaluation, puis le Batch Transform.
2. Vérifier le manifeste S3 et ses 16 horizons. Aucun modèle n'est actif avant
   cette première réussite complète.
3. Lancer `retail-inference` avec `{}`. Vérifier un nouveau fichier de prévisions
   et un manifeste contenant la même version de modèle.
4. Démarrer Streamlit avec le paquet du projet installé sur le notebook SageMaker.
   Vérifier l'origine, la version et les horizons affichés.
5. Activer les règles EventBridge seulement après validation et selon le crédit
   disponible. Le déploiement désactive l'ancienne règle `retail-forecast-daily` pour éviter
   l'ancien entraînement quotidien redondant.

Le code déployé utilise une archive en liste blanche, sans `.env` ni identifiants.
Le pilote installe les dépendances dans l'image SKLearn et utilise LabRole ; cela
nécessite une sortie réseau, `iam:PassRole`, SageMaker Processing/Training/Pipelines/
Transform, S3, Glue et Step Functions. Les permissions de confiance du rôle doivent
inclure les services qui l'assument. Un refus IAM ne doit pas être contourné par
une approbation moins rigoureuse.

Le Processing pilote reste facturé pendant l'attente des jobs imbriqués. Son
plafond est 10 500 secondes ; le pipeline interne est arrêté après deux heures.
En cas d'arrêt manuel du workflow, contrôler aussi les jobs imbriqués SageMaker
et les arrêter si nécessaire : l'annulation d'un parent ne garantit pas leur arrêt.

## Limite métier

La cible reste les ventes observées. Sans données de demande perdue, reconstituer
la demande réelle pendant les ruptures n'est pas identifiable. Le manifeste le
signale explicitement (`target: observed_sales`) ; aucune correction artificielle
n'a été appliquée aux labels. Les jours manquants restent inconnus ; lags,
cibles et fenêtres utilisent des dates réelles. Le manifeste distingue les
séries actives à la dernière origine des séries historiques plus anciennes.

## Contrôles effectués le 19 novembre 2026

L'authentification AWS fonctionne et le bucket, le registre, Glue et la lecture
Step Functions sont accessibles. Au premier inventaire aucun SageMaker Pipeline
n'était déployé. L'existence d'anciens packages Approved ne valide donc pas le
nouveau parcours. Les exécutions complètes doivent fournir leur propre preuve.

## Dashboard hébergé

```bash
.venv/bin/python -m infra.aws.dashboard_host
```

Configure un lifecycle sur `retail-forecast-nb`, sans remplacer un lifecycle
étranger, puis démarre le notebook s'il est arrêté. S'il est déjà InService avec
notre lifecycle, actualise les fichiers via l'API Jupyter authentifiée et met à
jour le code chargé au prochain démarrage. Aucun token de connexion n'est affiché.
Ouvrir Jupyter depuis la console SageMaker puis `/proxy/8501/`.
Le service HTTP a répondu `200 / ok` pendant cette validation. Le notebook reste
facturé tant qu'il est InService ; l'arrêter après la démonstration.

Les tests AWS ont aussi révélé et permis de corriger : une dépendance absente du
bundle et un conflit entre le montage automatique du script Processing et celui
du paquet du projet. Un troisième test a détecté les trous dans les dates des
926 000 ventes présentes sur S3 ; les features sont désormais calendaires,
sans fabrication de ventes à zéro.

## Cohorte entraînée

Les données AWS existantes mélangent des historiques anciens et le catalogue
courant (dates 2013 à 2024). Le Processing conserve les historiques des séries
présentes à la dernière origine, sans supprimer les sources de S3. Une série vue
pendant la dernière semaine mais absente du dernier jour fait échouer le contrôle
pour signaler une possible ingestion partielle. Cela évite de mélanger des jeux
sans lien ou de masquer une livraison manquante. Ce choix doit être remplacé par
un catalogue métier explicite si la notion d'article actif est disponible.
Le seuil WAPE reste inchangé. Le contrôle a été ajouté après détection d'une
forte dégradation sur les données mélangées, avant toute publication du modèle.

Le lancement avec `--wait` attend l'état terminal et renvoie une erreur si AWS
échoue ; un simple accusé de démarrage ne constitue pas une validation.
Les points d'entrée `aws_full_run.sh` et `infra.aws.full_run` utilisent aussi
les deux nouveaux workflows avec attente.

Références de conception : [intégration SageMaker et Step Functions](https://docs.aws.amazon.com/step-functions/latest/dg/connect-sagemaker.html)
et [écritures conditionnelles S3](https://docs.aws.amazon.com/S3/latest/userguide/conditional-writes.html).

## Reprise après évaluation réussie

Une erreur de format des métadonnées du registre a été détectée après le succès
réel de l'entraînement des 16 modèles et du gate. Les chaînes sont corrigées dans
la définition normale. SageMaker refuse une reprise sélective avec une autre
version de pipeline ; la reprise contrôlée reçoit donc dans Step Functions :

```json
{"evaluated_execution": "ARN_DE_EXECUTION_SOURCE"}
```

Ce chemin ne réentraîne pas. Il exige une exécution de ce projet dont seul le
registre a échoué, un gate `True`, des jobs Training et Evaluation terminés, le
même artefact et les mêmes features, puis un rapport indépendant complet dont
le WAPE respecte toujours le seuil configuré. L'inscription est idempotente.
La validation Batch Transform et la publication conditionnelle restent obligatoires.
Une évaluation échouée ou un gate `False` ne peuvent pas être repris ainsi.

## Contrat de serving et Athena

L'image SKLearn utilisée exige à la fois `code/inference.py`,
`code/requirements.txt` et `SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/model/code`.
Le conteneur copie alors le code dans son répertoire d'installation. Le parcours
normal prépare cette disposition pendant l'entraînement. Une reprise d'ancien
artefact ajoute seulement ces copies, puis vérifie la provenance d'évaluation.
[Source du conteneur AWS](https://github.com/aws/sagemaker-containers/blob/master/src/sagemaker_containers/_env.py)
et [chargement du module](https://github.com/aws/sagemaker-containers/blob/master/src/sagemaker_containers/_modules.py).
Les empreintes réelles des fichiers évalués et préservés sont dans
[evidence/serving_integrity.json](evidence/serving_integrity.json).
Un modèle existant avec un autre artefact, une autre image ou un mauvais
répertoire de handler est refusé. Un Batch Transform expiré est arrêté.

`make athena` crée `forecast_runs`, `active_manifest` et la vue
`forecast_active`. Cette vue sélectionne le fichier indiqué par le manifeste
avec `$path`, sans modification de catalogue à chaque publication. Les KPI
utilisent cette vue et les 16 horizons. L'ancienne table `forecast` est un
vestige historique ; elle ne participe plus aux KPI du parcours recommandé.
La période de 90 jours est relative à la dernière date du dataset.

## Résultats cloud vérifiés

Le 19 novembre 2026, le Training Job a entraîné les 16 modèles et l'évaluation
Processing indépendante a reproduit exactement leurs scores. WAPE moyen
**0,422609**, baseline **0,559139**, amélioration relative **24,4 %**.
Le pipeline source `yjcjidcc64i4` a passé le gate mais échoué à l'inscription
à cause des chaînes de métadonnées AWS. Sa définition est corrigée.
Il n'est pas présenté comme une exécution entièrement réussie.

La reprise contrôlée Step Functions
`edd8dfa5-b7e4-4dc5-9690-768aebcac892` a ensuite **réussi**, avec inscription v6,
Batch Transform, validation de **19 200 lignes** (1 200 séries × 16 jours) et
publication conditionnelle. La version v5, jamais active, a été rejetée après
le diagnostic du mauvais rangement du handler. Les poids n'ont pas été
réentraînés pour corriger l'inscription et le serving.

Athena lit exactement 19 200 lignes et 16 horizons depuis `forecast_active`.
Les deux KPI SQL ont réellement retourné 10 articles et 320 couples
(magasin, horizon). Un AppTest exécuté **sur le notebook AWS lui-même** a chargé
le dashboard, ses courbes et ses métriques sans erreur. La source affichée
est S3, modèle v6, origine 2024-12-30. Il s'agit de données synthétiques rejouées ;
les dates cibles sont 2024-12-31 à 2025-01-15, pas des prévisions datées de 2026.
Une tentative d'écriture S3 avec un ETag périmé a réellement été refusée
(HTTP 412), sans modification du manifeste actif.

Preuves : [évaluation](evidence/aws_evaluation.json),
[intégrité de serving](evidence/serving_integrity.json),
[dashboard hébergé](evidence/hosted_dashboard.json),
[Athena](evidence/athena_active.json),
[publication conditionnelle](evidence/conditional_publish.json).

## Développement local et éléments hors validation

Avec `S3_ENDPOINT_URL` configuré (s3mock), le dashboard et le brief utilisent
les prévisions et métriques locales. En AWS (`S3_ENDPOINT_URL` vide), l'absence
du manifeste actif est une erreur ; aucune ancienne prévision locale ne masque
un échec cloud. Le dashboard affiche les 16 horizons par défaut et exprime
l'écart de WAPE en points de pourcentage.

Le parcours démontré couvre ingestion existante, ETL, entraînement, évaluation,
registre, prévisions, publication, Athena et dashboard. Il ne valide pas les
exécutions HPO et endpoint serverless optionnelles, les quantiles P10/P90,
RMSLE, ni une nouvelle intégration LangChain/LangGraph du rapport LLM. La vidéo,
l'oral et l'export final du rapport restent des livrables distincts. Les anciens
constats détaillés du rendu restent consultables dans [review.md](review.md).

## Validation finale du quotidien

Le test a émis un événement EventBridge unique vers une règle de validation
éphémère, avec le même rôle et la même cible que la règle quotidienne.
L'exécution `36db742d-8175-40d7-618d-3f84226e9ce8_5f8822ea-8442-2eed-73ca-41c78eeaf246`
a terminé **SUCCEEDED** : Glue -> Processing -> Batch Transform -> publication.
Le modèle reste **v6**, le fichier de prévisions change et `previous` conserve
le résultat antérieur. Les 19 200 lignes ont été contrôlées à nouveau ; aucun
nouvel entraînement n'a eu lieu. Athena et un nouvel AppTest sur le notebook
ont ensuite lu le nouveau résultat sans erreur.

Preuves complètes : [workflows](evidence/aws_workflows.json) et
[inventaire final](evidence/aws_inventory.json). Au contrôle final, aucun job
Processing, Training ou Transform n'était encore actif. Les deux règles
récurrentes sont **DISABLED**, la règle éphémère a été supprimée, l'ancienne
règle quotidienne est absente. Le notebook reste **InService** pour ouvrir la
démonstration ; il reste facturé jusqu'à son arrêt.

Validation locale finale : **136 tests réussis**, sans services externes requis,
et `git diff --check` propre. Les alias PowerShell up/run/status/inference et
hébergement pointent aussi vers le nouveau parcours ; PowerShell n'est pas
installé sur ce poste, donc ce lanceur n'a pas été exécuté sous Windows.
`aws-pipeline-down` reste un nettoyage de l'ancienne orchestration seulement.
