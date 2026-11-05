# Script parlé — les 3 diapositives de `pres.pdf`

> Durée totale : **≈ 3 min 30**. Débit : lent, une idée par phrase.
> Les phrases en **gras** sont à dire telles quelles. Le reste est une indication de jeu.
> ⚠️ La diapositive d'architecture indique **16 jours** d'horizon : le script suit la
> diapositive. Si vous passez à 28, changez le chiffre au paragraphe 5.

---

## Diapo 1 — « Architecture » (≈ 1 min 30)

**Ouverture (2 s, sans regarder l'écran)**
« Cette diapositive montre le trajet complet : une vente entre, une prévision sort. Je vais suivre le chemin de gauche à droite. »

**1. Les deux sources (10 s)** — *montrer les deux blocs à gauche*
« **Tout part de deux sources. En haut, un flux de ventes simulé, de type Favorita : il imite des caisses qui envoient les ventes au fur et à mesure. En dessous, l'historique : le fichier Store Sales, qu'on charge une seule fois pour entraîner le modèle.** »

**2. Kafka + Firehose (15 s)** — *suivre la flèche vers S3 raw*
« **Les deux passent par Kafka, ici MSK. C'est une file d'attente : si le traitement ralentit, rien n'est perdu. Ensuite Data Firehose regroupe les ventes toutes les soixante secondes et les écrit dans S3, dans la zone brute — des fichiers compressés, un par lot.** »

**3. Glue + zone propre (15 s)** — *montrer Glue, puis S3 curated*
« **Glue nettoie : il enlève les doublons, corrige les types, rattache les informations du magasin et de l'article. Le résultat est un fichier Parquet, beaucoup plus rapide à lire, dans la zone propre. Juste en dessous, Athena sert aux questions ponctuelles, par exemple les indicateurs historiques.** »

**4. Processing (15 s)** — *montrer Processing*
« **On arrive au cœur du calcul. Processing fabrique les variables du modèle : les ventes des sept, quatorze et vingt-huit derniers jours, le jour de la semaine, les promotions. Ce sont ces variables que le modèle lira.** »

**5. Modèle actif + Batch (20 s)** — *montrer le bloc en pointillés, puis Batch inference*
« **Ici, en pointillés, le modèle actif : c'est la version validée, et elle est mise à jour une fois par mois. Batch inference la charge et produit les prévisions pour les seize prochains jours. On écrit le résultat dans S3 forecast, avec la version du modèle et la date : on sait donc toujours qui a produit quelle prévision.** »

**6. Contrôles + rapport + dashboard (20 s)**
« **Deux vérifications suivent. Les contrôles qualité regardent les données, les sorties et les erreurs. Et le rapport IA, avec Bedrock, est optionnel : il rédige un résumé en français des écarts. Enfin, Streamlit affiche tout ça au responsable des commandes.** »

**7. MLOps (10 s)** — *montrer le bandeau du bas*
« **En bas, la partie automatique : EventBridge lance le travail chaque jour, Step Functions enchaîne les étapes, CloudWatch surveille. Personne n'a besoin de cliquer.** »

**Transition**
« **Ce pipeline a besoin de données. Voyons lesquelles.** »

---

## Diapo 2 — « Les jeux des données » (≈ 55 s)

**1. Les trois sources (20 s)** — *main gauche*
« **On ne part pas de zéro : tout est public, sur Kaggle. M5, de Walmart, c'est le plus complet : trente mille séries, presque deux ans, avec les prix, les promotions et le calendrier. Rossmann apporte onze cents magasins sur neuf cent quarante-deux jours, avec l'état des magasins. Et Store Sales, la compétition Kaggle, sert de troisième exemple si on a besoin de comparer.** »

**2. La cible (10 s)** — *main droite*
« **À droite, ce qu'on fabrique. Ce qu'on prédit, c'est simplement la quantité vendue dans n jours, avec n entre un et vingt-huit. Un modèle par horizon : ça évite d'empiler les erreurs.** »

**3. Les variables (15 s)**
« **Les repères : les ventes des sept, quatorze et vingt-huit derniers jours — la moyenne, et la variabilité. Plus le contexte : jour de la semaine, jours fériés, promotions, prix, délai de livraison du magasin.** »

**4. La règle du découpage (10 s)**
« **Et une règle qu'on ne négocie pas : on coupe dans le temps, jamais au hasard. Le dernier mois sert de test, le mois d'avant de validation. Sinon on se retrouve à prédire le passé, et le modèle a l'air parfait alors qu'il triche.** »

**Transition**
« **Avec ces données, voilà ce que l'utilisateur obtient.** »

---

## Diapo 3 — « L'application et services » (≈ 55 s)

**1. Les fonctionnalités (25 s)** — *colonne de gauche*
« **Trois fonctionnalités. La première, la prévision : pour chaque magasin et chaque article, la quantité attendue, avec trois niveaux — bas, central, haut. Le niveau haut sert à calculer le stock de sécurité : on ne commande pas le centre, on commande avec une marge. La deuxième, le tableau de bord : prévu contre réalisé, les plus grosses erreurs, et la liste des produits en risque de rupture. La troisième, le résumé : chaque semaine, un texte généré automatiquement qui explique les écarts, envoyé par mail, avec des alertes au-delà d'un seuil.** »

**2. Les services (25 s)** — *colonne de droite*
« **Côté technique, quatre étages. Recevoir et stocker : S3, Glue ou Lambda, Kinesis ou MSK. Apprendre : SageMaker, de la préparation jusqu'au registre de modèles et à la surveillance. Expliquer : Bedrock, avec LangChain, uniquement pour le résumé — c'est optionnel. Afficher : Streamlit, ou QuickSight si on passe en production.** »

**3. La phrase de sortie (8 s)**
« **Un point qu'on assume : toutes ces briques sont payantes en production. On a donc prévu une version gratuite pour le laboratoire — Kafka open source, Streamlit, et le modèle en local. Le code est le même, on change deux lignes de configuration.** »

**Clôture**
« **Voilà pour notre idée. Nous pouvons répondre à vos questions.** »

---

## Questions probables (réponses courtes, 20 s chacune)

1. **Pourquoi seize jours et pas un mois ?**
   « C'est l'horizon de réassort des magasins. Le chiffre est un paramètre : `FORECAST_HORIZON`. Passer à vingt-huit ne change pas le code, seulement le coût d'entraînement. »

2. **Pourquoi Kafka, alors qu'un fichier suffit ?**
   « Parce que les ventes arrivent en continu et par pics. Kafka garde les messages si le traitement tombe, et permet de rejouer un historique sans rien casser. »

3. **Comment savez-vous que le modèle est bon ?**
   « Découpage temporel : le dernier mois n'a jamais été vu. Sur notre version locale, l'erreur tombe à quarante-trois pour cent, contre cinquante-sept pour la méthode simple — le même jour de la semaine précédente. »

4. **Ça coûte combien ?**
   « En démonstration, presque rien : SageMaker rentre dans le crédit du laboratoire, et le reste a une version gratuite. Un endpoint est la seule ligne qui coûte à l'heure, donc on l'éteint après la démo. »

5. **Pourquoi un rapport généré par IA ?**
   « Les chiffres ne disent pas *pourquoi*. Bedrock transforme les écarts en trois phrases lisibles par un chef de rayon. Et c'est optionnel : sans Bedrock, on garde un résumé calculé localement. »

---

## Si la démonstration échoue

Ne pas s'excuser plus de trois secondes. Enchaîner :

« **Je vous montre la preuve sans interface : les fichiers sont déjà dans S3, et voici le rapport généré.** »

Puis montrer : la liste des objets `raw/sales/ingest_date=…/`, et le fichier `reports/local_baseline.json` (erreur 0,43 contre 0,57).

