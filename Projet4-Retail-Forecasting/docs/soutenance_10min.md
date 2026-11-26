# Soutenance — Projet 4, Retail ML Forecasting

Durée cible : dix minutes, dont deux minutes de vidéo. Les cinq diapositives de `slides.pptx` servent de support. Ouvrir le rapport final pour les chiffres de la dernière exécution ; les chiffres indiqués « référence » sur les slides restent ceux du test antérieur. Répéter à voix haute avant la présentation.

## 0:00–1:00 · Slide 1 — besoin et périmètre

Notre projet prévoit les ventes observées d'un article dans un magasin pour chacun des seize prochains jours. Le responsable des achats peut comparer les séries, consulter les articles les plus demandés et examiner l'incertitude avant de prendre sa décision. Le système ne passe pas de commande automatiquement.

Nous utilisons un jeu synthétique de vingt magasins et soixante articles, avec deux années d'historique. C'est un démonstrateur technique : nous ne prétendons pas avoir validé une performance commerciale sur des données d'entreprise. La dernière observation date de décembre 2024 ; les prévisions de la démonstration concernent les jours suivants, même si l'exécution AWS est effectuée en novembre 2026.

Une limite métier est essentielle : les ventes observées ne mesurent pas la demande perdue en cas de rupture. Notre modèle ne prétend donc pas reconstruire cette demande cachée.

## 1:00–2:30 · Slide 2 — architecture et contraintes du lab

L'architecture distingue ingestion, entraînement et inférence. Le transport passe par Kafka, un pont vers Firehose DirectPut, puis S3. Une première validation AWS a transporté vingt mille événements. Le test complet actuel repart de ces données déjà stockées : il ne faut pas confondre ces deux preuves.

Glue nettoie, type et déduplique les événements pour écrire du Parquet dans curated. Le workflow mensuel Step Functions déclenche la préparation des features, l'entraînement, l'évaluation indépendante et l'inscription au registre. Le workflow quotidien réutilise le modèle actif et produit un nouveau lot sans réentraîner.

Nous avons adapté les services aux autorisations du Learner Lab. Kafka sur EC2 remplace MSK, refusé par le rôle ; Streamlit remplace QuickSight. Le LLM optionnel utilise Ollama. Ces substitutions sont documentées et préservent les fonctions attendues. Les diagrammes représentent la cible ; les identifiants et statuts des exécutions indiquent ce qui a réellement été testé.

## 2:30–4:00 · Slide 3 — modèle et prévention des fuites

LightGBM convient aux données tabulaires, aux retards et aux variables calendaires. Nous entraînons un modèle direct par horizon. Pour chaque jour, un booster prédit la moyenne et deux boosters quantiles produisent P10 et P90 : quarante-huit boosters au total.

Les entrées comprennent des retards et des statistiques glissantes, calculés avec le passé disponible à l'origine. Le calendrier de la date cible est connu. Stock et rupture sont exclus des features. Les jours absents restent inconnus ; ils ne sont pas transformés artificiellement en ventes nulles.

Nous séparons les dates, et non les lignes au hasard. La validation couvre cinquante-six jours et le test vingt-huit jours. Les labels sont purgés aux frontières pour éviter qu'une cible future du train entre dans la validation ou le test. L'évaluation utilise des origines glissantes : à chaque origine, seul le passé alors disponible est autorisé.

Le modèle de référence utilise au plus quatre cents arbres, un taux d'apprentissage de 0,05, soixante-trois feuilles et un arrêt anticipé. Six essais HPO AWS explorent aussi les paramètres sur J+1 avec le WAPE de validation. Ce résultat exploratoire ne prouve pas l'optimum pour tous les horizons et ne remplace pas automatiquement le modèle publié.

## 4:00–5:30 · Slide 4 — publication sûre et rapport LLM

L'évaluation indépendante recharge réellement les poids du modèle et recalcule ses prédictions. Elle ne se contente pas de recopier le JSON d'entraînement. Le gate impose un WAPE moyen inférieur ou égal à 0,45 avant l'approbation automatique.

L'approbation ne suffit pas à rendre un modèle actif. Un Batch Transform doit produire un lot complet, avec des valeurs finies et non négatives. Le fichier est immuable. Ensuite seulement, une écriture conditionnelle du manifeste relie version, modèle, métriques et prévisions. En cas d'échec ou de publication concurrente, le résultat précédent reste disponible.

Le LLM reçoit des agrégations déterministes des prévisions actives. LangChain compose le prompt, appelle Ollama puis vérifie la structure de la réponse. Des nombres absents du contexte et certaines affirmations non mesurées sont refusés. Cette validation ne garantit pas toutes les interprétations métier : le texte demande toujours une relecture. Il s'agit d'un rapport optionnel, sans agent autonome ni RAG.

## 5:30–6:00 · Slide 5 — lire les résultats

Le rapport final fournit le WAPE du modèle, la baseline saisonnière, le RMSLE et la couverture empirique P10/P90 de la dernière exécution. Lire ces valeurs dans le tableau du rapport, sans mélanger les versions. Le gain est relatif au WAPE de la baseline, tandis que l'écart affiché dans le dashboard est exprimé en points de pourcentage.

Nous allons maintenant voir l'interface réellement hébergée sur le notebook AWS.

## 6:00–8:00 · Vidéo de démonstration

Lire `output/video/demo_projet4.mp4` en plein écran. La vidéo montre la version active, les filtres, une série avec sa bande P10/P90, les agrégations et les données téléchargeables. Ne pas afficher d'identifiants AWS ou d'URL de connexion temporaire.

## 8:00–9:00 · Slide 5 — incertitude et limites

P10 et P90 sont des régressions quantiles, pas une marge arbitraire autour de la moyenne. Nous mesurons leur couverture sur le test et ne promettons pas une couverture individuelle de quatre-vingts pour cent. Le modèle peut être mal calibré sur une série particulière. Additionner les P90 quotidiens ne donne pas non plus un P90 cumulé.

Le jeu synthétique facilite la reproductibilité, mais ne remplace pas une validation sur des ventes réelles, des changements de catalogue, des promotions futures connues et des ruptures. Une décision de stock demanderait aussi les délais d'approvisionnement et les coûts de surstock et de rupture.

## 9:00–10:00 · Exploitation et conclusion orale

Les preuves conservées comprennent les exécutions mensuelle et quotidienne, les métriques indépendantes, le HPO, l'inférence et les tests locaux. Les fichiers historiques restent disponibles pour comparer les versions. Athena sélectionne le fichier actif via le manifeste plutôt que de cumuler tous les anciens runs.

Dans le Learner Lab, nous laissons les horaires désactivés et arrêtons le notebook après la démonstration. L'endpoint de test est supprimé. Les données et les modèles sont conservés dans S3 ; le stockage n'est donc pas présenté comme gratuit.

Le résultat est une chaîne de prévision reproductible, avec séparation temporelle, évaluation indépendante et publication contrôlée. La prochaine étape métier serait une validation sur des données réelles et une calibration des décisions de réassort, avant toute automatisation de commandes.

## Questions à préparer

- Pourquoi LightGBM plutôt qu'un réseau neuronal ? Données tabulaires, temps de calcul limité et baseline reproductible ; aucune supériorité universelle revendiquée.
- Pourquoi seize jours ? Horizon retenu dans `aws presentation.pdf`.
- Peut-on utiliser le modèle sans LLM ? Oui, prévisions, métriques et dashboard sont indépendants du texte optionnel.
- Que se passe-t-il si l'entraînement échoue ? Le manifeste actif précédent reste utilisé.
- Le HPO emploie-t-il le test comme objectif ? Non, il optimise `valid_wape` sur J+1.
- Pourquoi pas un P50 ? Le booster de moyenne optimise une régression ; il n'est pas présenté comme une médiane.
