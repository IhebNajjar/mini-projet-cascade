# Questions probables du professeur — réponses courtes

Ce fichier accompagne `README.md`. Il répond aux questions que le professeur
peut poser à l'oral pendant la soutenance du mini-projet.

---

## 1. Quel est le but du projet ?

Une plateforme e-commerce reçoit des avis clients sous trois formes : texte,
image (capture d'écran) et audio (message vocal). Le système doit toujours
donner le même résultat : le sentiment de l'avis — `positif`, `negatif` ou
`neutre` — et un score de confiance, via une API REST.

## 2. Qu'est-ce que l'architecture en cascade ?

L'image et l'audio ne sont pas classés directement. On les convertit d'abord
en texte : OCR pour l'image, transcription (ASR) pour l'audio. Ensuite un
**seul** classifieur texte prend la décision finale.

```
Texte -------------------> POST /predict/text ----\
Image --> OCR (texte) ---> POST /predict/image ----> classifieur texte ---> JSON
Audio --> ASR (texte) ---> POST /predict/audio ----/
```

Avantage : une seule logique de classification à entretenir pour trois
sources.

## 3. Pourquoi cette architecture plutôt que trois modèles ?

Trois modèles différents demanderaient trois entraînements, trois jeux de
données et trois façons de répondre. La cascade garde une seule brique de
décision et une seule forme de réponse JSON. C'est exactement l'idée clé
demandée dans le sujet.

## 4. Quel modèle avez-vous utilisé et pourquoi ?

Une **Régression Logistique** scikit-learn, avec une vectorisation **TF-IDF**
(les mots et les paires de mots, `ngram_range=(1, 2)`).

- TF-IDF transforme le texte en nombres que le modèle peut calculer. Les mots
  rares et importants comptent plus que les mots très fréquents.
- La Régression Logistique donne des probabilités : c'est pourquoi on peut
  afficher un champ `confiance` entre 0 et 1.
- Le jeu ne contient que 24 avis : un modèle simple est plus honnête qu'un
  gros modèle qui memoriserait tout.

## 5. Qu'est-ce que le nettoyage du texte ?

`preprocessing.py` met le texte en minuscules, retire la ponctuation et
quelques mots purement grammaticaux (`le`, `la`, `de`, `et`...). Il garde les
mots de négation (`pas`, `ne`, `jamais`) : sinon « pas bon » deviendrait « bon ».
Les accents sont ignorés pour que « qualité » et « qualite » comptent pareil.

## 6. Pourquoi avez-vous ajouté des listes de mots ?

Avec seulement 24 avis, un mot tapé par un utilisateur est souvent absent de
l'entraînement. Les listes de mots positifs, négatifs et neutres donnent des
indices supplémentaires au modèle. Par exemple « superbe » ou « nul » sont
reconnus même s'ils ne figurent pas dans `textes.csv`. Ce sont cinq nombres
ajoutés à la vectorisation, calculés à la main — rien n'est caché dans une
boîte noire.

## 7. Comment avez-vous évalué le modèle ?

`train_model.py` affiche la **matrice de confusion** et le **rapport de
classification** (précision, rappel, F1). Les 24 textes d'entraînement sont
classés correctement (accuracy 1.00). En plus, les 9 textes attendus des
images et les 9 transcriptions attendues des audios sont classés correctement
(18/18). Ces résultats sont écrits dans le README.

## 8. Un score de 1.00 n'est-il pas suspicieux ?

Oui, et je le dis clairement : l'évaluation est faite sur les textes qui ont
servi à l'entraînement. Ce n'est **pas** une mesure de généralisation. Le
jeu fourni est petit (24 avis). Le but du mini-projet est de montrer le
principe de la cascade, pas de livrer un modèle de production.

## 9. Quelles sont les routes de l'API ?

Quatre routes :

| Route | Entrée |
|---|---|
| `POST /predict/text` | un corps JSON avec `{"texte": "..."}` |
| `POST /predict/image` | un fichier image en `multipart/form-data` |
| `POST /predict/audio` | un fichier audio en `multipart/form-data` |
| `POST /predict` | détection automatique du type d'entrée |

Toutes renvoient la même forme JSON : `source`, `texte_extrait`,
`sentiment_predit`, `confiance`.

## 10. À quoi sert Pydantic ?

Pydantic valide les données **avant** le traitement : texte non vide,
extension et type MIME autorisés, fichier non vide et ≤ 10 Mo. Une entrée
incorrecte donne une erreur **422 Unprocessable Entity**, pas une erreur 500.
C'est la robustesse demandée à l'étape 5.

## 11. Comment fonctionne le module image ?

`ocr_module.py` ouvre l'image avec Pillow, la convertit en niveaux de gris,
augmente le contraste, puis appelle `pytesseract.image_to_string()`.
Tesseract n'étant pas installé sur cette machine, le module retombe sur le
texte attendu du `images.csv` afin que l'API réponde quand même. Installer
Tesseract et l'ajouter au PATH active le vrai OCR.

## 12. Comment fonctionne le module audio ?

`asr_module.py` envoie le fichier à l'API Groq avec le modèle
`whisper-large-v3-turbo` et la langue `fr`. Le texte transcrit est envoyé au
classifieur. Sans clé `GROQ_API_KEY` ou en cas de quota épuisé, le module
utilise la transcription de référence du `audios.csv`.

## 13. Pourquoi les fichiers s'appellent .mp3 alors que le sujet dit .wav ?

Le dossier fourni contient des `.mp3`. Le module accepte `.wav`, `.mp3`,
`.ogg` et `.m4a` : le sujet est respecté et les fichiers réels fonctionnent.

## 14. Qu'est-ce que CORS et pourquoi l'utiliser ?

La page web est ouverte sur le port 8001 et l'API sur le port 8000 : pour le
navigateur ce sont deux sites différents. `CORSMiddleware` autorise la page
à appeler l'API. Sans lui, le navigateur bloquerait la requête.

## 15. Pourquoi une page HTML et pas Streamlit ?

Le sujet propose deux options : A (documentation native `/docs`) et B (mini
application). Les deux sont faites : `web_app.html` pour l'option B, Swagger
et ReDoc pour l'option A. Une page HTML statique n'a aucune dépendance
supplémentaire.

## 16. Qu'est-ce que Swagger / Redoc ?

Ce sont les documentations automatiques générées par FastAPI à partir du
code : `/docs` (interactive, on peut essayer les routes) et `/redoc` (lecture).
C'est une preuve que l'API est bien structurée.

## 17. Qu'avez-vous testé exactement ?

`test_api.py` contient 9 tests avec `httpx.TestClient` : les routes texte,
image, audio et auto renvoient bien 200 avec les quatre clés attendues, et
les charges utiles incorrectes (texte vide, mauvais format de fichier, JSON
vide) renvoient bien 422. Commande : `pytest` → 9 passed.

## 18. Comment lancez-vous le projet ?

Double-clic sur `start.bat` : il vérifie Python, les modèles, ouvre l'API et
la page web dans deux fenêtres, puis ouvre les deux adresses dans le
navigateur. Double-clic sur `test.bat` : il lance les 9 tests. Les commandes
manuelles sont dans le README (`uvicorn main:app --reload`,
`python -m http.server 8001`, `pytest`).

## 19. Quelles sont les limites du projet ?

Petit jeu de données (24 textes, 9 images, 9 audios), OCR réel indisponible
sur cette machine, transcriptions audio dépendantes d'une clé Groq, et score
de 1.00 calculé sur l'entraînement. Je les ai écrites dans le README au lieu
de les masquer.

## 20. Que feriez-vous avec plus de temps ?

Un vrai jeu de données beaucoup plus grand, une validation avec séparation
entraînement/test ou Leave-One-Out, l'installation de Tesseract pour l'OCR,
et une mesure d'erreur de transcription (WER) pour l'audio, comme dans
l'atelier 3.
