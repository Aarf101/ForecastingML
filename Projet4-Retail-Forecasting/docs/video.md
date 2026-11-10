# Vidéo de démonstration — 2 minutes

Rendu n°5 du PDF. Contrainte : **120 secondes**, une seule prise si possible.
Tout est enregistré à l'écran (OBS ou `ffmpeg`), sans voix off improvisée : lire
ce texte, il est calibré pour tenir dans le temps.

Préparation (à faire **avant** d'appuyer sur Record) :

```bash
# 1. Notifications coupées, onglets fermés, zoom navigateur à 110 %
# 2. Repo prêt : terminal dans /home/aarf101/projaws
make test                       # doit afficher 48 passed
# 3. Dashboard accessible (notebook démarré + streamlit lancé)
# 4. Contexte AWS valide : AWS_PROFILE=lab aws sts get-caller-identity
```

| # | Temps | Écran | Ce qu'on dit |
|---|---|---|---|
| 1 | 0:00–0:10 | Slide titre | « Projet 4 : prévision de la demande J+1 à J+28 par magasin et par article, sur AWS. » |
| 2 | 0:10–0:25 | Terminal → `make test` | « 48 tests, aucun service externe. La donnée vient d'un générateur : 876 000 lignes, 1 200 séries, deux ans. » |
| 3 | 0:25–0:45 | S3 console → `raw/`, `curated/` | « Le lac de données : le brut, puis le même contenu nettoyé en Parquet — dédupliqué sur magasin, article, date. » |
| 4 | 0:45–1:05 | SageMaker → Training jobs → job `Completed` | « Un job d'entraînement LightGBM, un modèle par horizon. WAPE 0,43 contre 0,57 pour la baseline saisonnière : 24 % de mieux. » |
| 5 | 1:05–1:20 | Model registry → v1/v2/v3 | « Le registre bloque la publication si le WAPE dépasse le seuil : la v1 est restée en attente, la v3 est approuvée. » |
| 6 | 1:20–1:40 | Athena → résultat de la requête KPI | « Les prévisions et l'historique sont interrogeables en SQL directement sur S3, sans base de données. » |
| 7 | 1:40–1:55 | Dashboard Streamlit (URL visible) | « Le dashboard lit S3 : prévu contre réalisé, les articles à risque, et le brief hebdomadaire rédigé par le LLM. » |
| 8 | 1:55–2:00 | Diapo « coût maîtrisé » | « Coût total de la démonstration : moins de 40 centimes. » |

## Règles de tournage

1. **Pas de commande longue en direct.** Les jobs prennent 3 à 10 minutes :
   montrer la page de résultat, pas l'exécution. Sauf `make test` (1 s), qui est
   la seule commande à lancer devant la caméra.
2. **Le numéro de version du modèle doit être lisible** : `v3` apparaît dans
   `forecast/` et dans le registre — c'est ce qui prouve la traçabilité.
3. **Si un endpoint LLM apparaît**, préciser « Bedrock n'est pas activé dans le
   lab, le modèle tourne sur un endpoint Hugging Face de SageMaker ».
4. **Couper le Wi-Fi n'est pas une option** : le dashboard lit S3 en direct.

## Ce qu'il ne faut PAS montrer

- Le `s3mock` local et Redpanda : ils servent au développement, pas à la démo
  finale (le jury verrait surtout qu'ils ne sont pas sur AWS).
- Les échecs des deux premiers Transform Jobs — ils sont excellents à l'oral,
  mais 2 minutes ne suffisent pas pour les expliquer.

## Après l'enregistrement

```bash
ffprobe -v error -show_entries format=duration -of csv=p=0 demo.mp4   # ≈120 s
```
