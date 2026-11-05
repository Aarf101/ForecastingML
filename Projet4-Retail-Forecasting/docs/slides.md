# Exposé idée projet — 5 slides (langage simple)

> Projet 4 · Retail ML Forecasting · MIT 2026
> Version à présenter : `docs/slides.pptx`. Script parlé : `docs/script.md`.

---

## Slide 1 — L'application et le besoin

**Titre** : Prévoir la demande produit, magasin par magasin

**Le problème**
- Aujourd'hui, on commande « au feeling ».
- On commande trop peu : rupture de stock, ventes perdues.
- On commande trop : stock qui dort, démarques.
- On veut une prévision par magasin et par article, pour les 28 prochains jours.
- Et une fourchette (« entre X et Y »), pas un chiffre unique.

**Pour qui ?**
- Le responsable des commandes : décider des réassorts.
- Le chef de rayon : savoir quoi traiter en premier.
- La direction : suivre un indicateur simple, le taux d'erreur.

**L'outil** : un tableau de bord, plus un assistant qui explique chaque prévision.

**Trois chiffres** : `erreur < 30 %` · `horizon 28 jours` · `1 200 séries suivies`

*Notes orateur* : on part d'un problème simple — les commandes se font à l'expérience. On veut les transformer en décision chiffrée, avec une marge d'erreur.

---

## Slide 2 — L'architecture

**Titre** : Un chemin simple, de la vente à la prévision

`![Architecture](diagrams/architecture.png)`

- **Les ventes arrivent** en continu, une par une (Kafka).
- **Elles sont stockées** dans S3, dans une zone brute.
- **Elles sont nettoyées** (Glue) puis rangées dans une zone propre.
- **SageMaker calcule** les variables, entraîne le modèle et produit les prévisions.
- **Streamlit affiche** le résultat pour le responsable.
- **Un bandeau MLOps** lance tout seul le travail chaque jour.

*Notes orateur* : insister sur deux choses — rien n'est perdu en route (Kafka), et tout est automatique (EventBridge, Step Functions).

---

## Slide 3 — Les données

**Titre** : Des données publiques, prêtes à l'emploi

**D'où elles viennent**
- **M5 (Walmart)** : 30 490 séries, 2 ans, avec prix, promotions et calendrier.
- **Rossmann** : 1 115 magasins, 942 jours.
- **Store Sales (Kaggle)** : un troisième exemple si besoin.
- **Plan B** : un simulateur maison, 876 000 lignes, même format.

**Ce qu'on en tire**
- **Ce qu'on prédit** : la quantité vendue dans n jours.
- **Les repères** : les ventes des 7, 14 et 28 derniers jours (moyenne et variabilité).
- **Le contexte** : jour de la semaine, jours fériés, promotions, prix, délai de livraison.
- **La règle** : on coupe dans le temps. Le dernier mois sert à tester, celui d'avant à valider.

*Notes orateur* : le point à faire passer — on ne mélange jamais passé et futur, sinon on croit que le modèle marche alors qu'il triche.

---

## Slide 4 — Fonctionnalités et services

**Titre** : Ce que l'outil fera, et avec quoi

**Fonctionnalités**
- Une prévision à 28 jours, avec trois niveaux : bas, central, haut.
- Le niveau haut sert à calculer le **stock de sécurité**.
- Un **tableau de bord** : prévu contre réalisé, grosses erreurs, risques de rupture.
- Un **résumé automatique** chaque semaine, par mail, avec des alertes.

**Services**
- **Stocker et recevoir** : S3, Glue ou Lambda, Kinesis ou MSK.
- **Apprendre** : SageMaker (préparation, entraînement, registre, surveillance).
- **Expliquer** : Bedrock, en option, pour rédiger le résumé.
- **Afficher** : Streamlit, ou QuickSight en option.

*Notes orateur* : la valeur n'est pas le modèle seul, c'est la chaîne complète : la prévision devient une décision, et le résumé explique la décision.

---

## Slide 5 — Où on en est

**Titre** : Ce qui marche déjà, sans AWS

- 876 000 lignes de données générées et **42 tests** qui passent.
- Le flux **Kafka → S3** fonctionne : 2 000 ventes donnent 4 fichiers.
- Modèle local : **erreur 43 %**, contre **57 %** pour la méthode simple (« le même jour la semaine dernière »), soit **24 % de mieux**.
- Le schéma et la documentation sont prêts.

**La suite** : entraîner sur SageMaker, brancher le tableau de bord, puis le résumé automatique.

*Notes orateur* : on arrive avec quelque chose qui tourne, pas avec des slides. Le temps restant sert à le porter sur AWS et à préparer la démonstration.
