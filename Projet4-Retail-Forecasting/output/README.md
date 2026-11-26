# Rendu du projet 4

- `pdf/rapport_final.pdf` : rapport technique basé sur les preuves du 26 novembre 2026.
- `pdf/slides.pdf` : version PDF des cinq diapositives ; source éditable `../docs/slides.pptx`.
- `video/demo_projet4.mp4` : deux minutes d'interactions réelles avec Streamlit sur AWS.
- `../docs/soutenance_10min.md` : script oral, dont deux minutes réservées à la vidéo.
- `../docs/evidence/completion_2026-11-26.json` : métriques et identifiants d'exécution.

Reconstruction du rapport : `.venv/bin/python -m pip install -e '.[docs]'`, puis `make report-final`.
Les diagrammes et slides se reconstruisent avec `make diagram` et `make slides`.
La vidéo est un enregistrement du navigateur ; elle reste consultable après l'arrêt du lab.
Les données sont synthétiques et l'origine des prévisions est le 30 décembre 2024.
