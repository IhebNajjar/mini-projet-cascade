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
classés correctement (accuracy 1.00). J'ajoute ensuite une évaluation
**Leave-One-Out** : on retire chaque avis à tour de rôle, on réentraîne sur
les 23 autres, et on prédit l'avis retiré. Résultat : **0.958**, soit 23 avis
sur 24 corrects ; un avis positif est prédit neutre. Les 9 textes réellement
lus par Tesseract et les 9 transcriptions réellement produites par Whisper
sont classés correctement (18/18).

## 8. Un score de 1.00 n'est-il pas suspicieux ?

Oui, et je le dis clairement : ce 1.00 est mesuré sur les textes qui ont servi
à l'entraînement. Ce n'est **pas** une mesure de généralisation. C'est
pourquoi j'ai fait le Leave-One-Out : **0.958**. Et avec TF-IDF seul, sans mes
listes de mots, le Leave-One-Out tombe à **0.125** — un artefact d'un jeu
minuscule et parfaitement équilibré, où retirer un exemple rend sa classe plus
rare. La conclusion est importante : ce sont les listes de mots qui apportent
la connaissance préalable et font passer la généralisation de 0.125 à 0.958.
Le but du mini-projet est de montrer le principe de la cascade, pas de livrer
un modèle de production.

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
augmente le contraste, puis appelle `pytesseract.image_to_string(image,
lang="fra")`. C'est le **vrai OCR** : le texte vient de l'image, jamais du
`images.csv`. J'ai installé Tesseract 5.4 avec le pack de langue française
sur cette machine, et les 9 images sont lues correctement. Si Tesseract est
absent, la route renvoie **503** ; si l'image est illisible ou ne contient
aucun texte, elle renvoie **422**. Sur Windows il faut installer Tesseract +
`fra.traineddata` et redémarrer le terminal ; le module détecte aussi
automatiquement `C:\Program Files\Tesseract-OCR\tesseract.exe` si le PATH
n'est pas encore à jour.

## 12. Comment fonctionne le module audio ?

`asr_module.py` utilise **faster-whisper** en local, modèle `base`, langue
`fr`, sur CPU en `int8`. Le modèle est chargé une seule fois à la première
requête, puis mis en cache. C'est une **vraie transcription** du fichier
envoyé : aucun retour au `audios.csv`. Pas besoin de clé API ni de ffmpeg
système sur Windows. Le premier appel télécharge le modèle (~100 Mo), donc
je le fais une fois avant la démonstration. Si rien n'est entendu, la route
renvoie **422**. Avec un français perfectible, on peut passer à `"small"`
(plus lent).

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

`test_api.py` contient **11 tests** avec `httpx.TestClient` : les routes texte,
image, audio et auto renvoient bien 200 avec les quatre clés attendues, la
transcription audio contient plus de 3 mots (donc elle vient bien du son), et
les charges utiles incorrectes renvoient bien 422 : texte vide, image blanche
(aucun texte détecté), image corrompue, mauvais format de fichier, JSON vide.
Commande : `pytest` → 11 passed.

## 18. Comment lancez-vous le projet ?

Double-clic sur `start.bat` : il vérifie Python, les modèles, ouvre l'API et
la page web dans deux fenêtres, puis ouvre les deux adresses dans le
navigateur. Double-clic sur `test.bat` : il lance les 11 tests. Les commandes
manuelles sont dans le README (`uvicorn main:app --reload`,
`python -m http.server 8001`, `pytest`).

## 19. Quelles sont les limites du projet ?

Petit jeu de données (24 textes, 9 images, 9 audios), OCR et ASR réels mais
dépendants de l'installation de Tesseract + pack français et du téléchargement
du modèle Whisper au premier appel, et score de 1.00 calculé sur
l'entraînement (le score honnête est 0.958 en Leave-One-Out). Je les ai écrites
dans le README au lieu de les masquer.

## 20. Que feriez-vous avec plus de temps ?

Un vrai jeu de données beaucoup plus grand, une validation avec séparation
entraînement/test, davantage de mots dans les listes, une mesure d'erreur de
transcription (WER) pour l'audio comme dans l'atelier 3, et le passage au
modèle Whisper `small` pour un meilleur français. Le Leave-One-Out est déjà
fait : 0.958 contre 0.125 sans lexique.

## 21. Le sujet demande une comparaison : avez-vous testé un SVM ?

Oui — c'est même le comparateur proposé par le PDF (« Régression Logistique
ou SVM »). `train_model.py` entraîne les deux modèles avec la **même**
vectorisation et affiche une matrice de confusion pour chacun, sur
l'entraînement et en Leave-One-Out :

| Modèle | Acc. entraînement | Acc. LOO | F1 macro LOO |
|---|---|---|---|
| **Régression Logistique (retenu)** | 1.000 | **0.958** | **0.958** |
| SVM linéaire (SVC) | 1.000 | **0.958** | **0.958** |

Matrices LOO (lignes = vérité, colonnes = prédiction) :

```
Régression Logistique        SVM linéaire (SVC)
[[8 0 0]                     [[8 0 0]
 [0 8 0]                      [0 8 0]
 [0 1 7]]                     [0 1 7]]
```

Les deux modèles se trompent sur le même avis positif, prédit `neutre`. Je
garde la Régression Logistique parce qu'à score égal elle retourne des
probabilités : c'est ce qui alimente le champ `confiance` de l'API. Le SVC
par défaut donne une décision sèche ; il faudrait `probability=True` (plus
lent) pour obtenir une confiance.

## 22. Bonus : et avec un SVR ?

Le script affiche aussi une ligne `SVR` à titre de curiosité. SVR est un
modèle de **régression** : `PontSVR` code les classes 0/1/2 puis arrondit la
sortie. Il obtient 1.000 sur l'entraînement mais seulement **0.875** en LOO,
avec 3 avis positifs prédits neutres :

```
SVR — Leave-One-Out
[[8 0 0]
 [0 8 0]
 [0 3 5]]
```

C'est une limite normale d'un modèle de régression utilisé pour classer. Ce
n'est pas le comparateur demandé — le bon comparateur est le SVM de
classification (SVC) de la question 21 — mais cela montre pourquoi je n'ai
pas retenu SVR.
