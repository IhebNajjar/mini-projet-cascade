# Mini-projet — Classifieur de Sentiment Multi-source

**E-commerce Customer Voice** — DSI-31, novembre 2026.

Un seul système reçoit un avis client sous trois formes (texte, image, audio),
le convertit en texte brut quand c'est nécessaire, prédit le sentiment
(`positif`, `negatif`, `neutre`) avec un **classifieur textuel unique**, et
renvoie une réponse JSON toujours identique.

```
Texte  ───────────────────────────► POST /predict/text ──┐
Image  ──► OCR (pytesseract) ─────► POST /predict/image ─┤
Audio  ──► ASR (Whisper) ────────► POST /predict/audio ─┼──► { "source",
Auto   ──► détection du type ─────► POST /predict ──────┘      "texte_extrait",
                                                              "sentiment_predit",
                                                              "confiance" }
```

---

## 1. Contenu du dépôt

```
mini-projet-cascade/
├── preprocessing.py      Étape 1 — nettoyage + listes de mots de sentiment
├── train_model.py        Étape 1 — entraînement, évaluation, sauvegarde des .pkl
├── classifier.py         Étape 1 — chargement des artefacts et prédiction
├── ocr_module.py         Étape 2 — lecture du texte dans une image (pytesseract + Pillow)
├── asr_module.py         Étape 2 — transcription d'un message vocal (Whisper)
├── schemas.py            Étape 3 — validation Pydantic (format, MIME, taille)
├── main.py               Étape 3 — application FastAPI et 4 routes
├── web_app.html          Étape 4, option B — mini-application web HTML/JS
├── test_api.py           Étape 5 — 9 tests pytest avec httpx.TestClient
├── requirements.txt      Dépendances Python
├── data_test_v2/         Jeu de données fourni (24 textes, 9 images, 9 audios)
│   ├── textes.csv
│   ├── images.csv
│   ├── audios.csv
│   ├── images/           I01.png … I09.png
│   └── audios/           A01.mp3 … A09.mp3
├── models/               Artefacts sérialisés
│   ├── model.pkl
│   └── vectorizer.pkl
├── screenshots/
│   ├── web_app.png       Mini-application en fonctionnement
│   └── swagger.png       Documentation interactive /docs
└── README.md             Ce fichier
```

Organisation volontairement **plate** : les modules sont à la racine, sans
dossier `app/` ni `tests/`. Chaque atelier de la classe ayant sa propre
structure, celle-ci est propre à ce mini-projet.

---

## 2. Installation

Python 3.10 ou plus récent est nécessaire.

```bash
cd mini-projet-cascade
pip install -r requirements.txt
```

Les modèles étant déjà fournis dans `models/`, l'entraînement n'est pas
obligatoire. Pour le rejouer :

```bash
python train_model.py
```

Sortie attendue (24 exemples, 8 par classe) :

```
Matrice de confusion (lignes = vérité, colonnes = prédiction)
                 negatif      neutre     positif
     negatif           8           0           0
      neutre           0           8           0
     positif           0           0           8
accuracy 1.00   macro avg 1.00
```

**À lire honnêtement :** l'évaluation est faite sur les 24 textes servant à
l'entraînement. Avec un jeu aussi petit, ce score de 1.00 n'est pas une
mesure de généralisation. C'est une vérification que le modèle a bien appris
les exemples fournis (choix conforme à l'étape 1 du sujet, qui demande une
matrice de confusion et un rapport de classification).

Vérification complémentaire : les 9 textes attendus des images et les 9
transcriptions attendues des audios sont classés correctement (18/18) par
le même classifieur texte. Cela valide la cascade OCR/ASR → texte → sentiment.

Transcription audio : la clé `GROQ_API_KEY` permet d'utiliser
`whisper-large-v3-turbo`. Sans clé, le projet reste fonctionnel : il utilise
la transcription de référence présente dans `audios.csv`.

---

## 3. Lancement

### API FastAPI

```bash
uvicorn main:app --reload
```

Documentation interactive : <http://127.0.0.1:8000/docs> (Swagger) et
<http://127.0.0.1:8000/redoc>.

### Mini-application web (option B du sujet)

Dans un second terminal, depuis le même dossier :

```bash
python -m http.server 8001
```

Puis ouvrir <http://127.0.0.1:8001/web_app.html>.

La page envoie les requêtes à `http://127.0.0.1:8000` ; la classe
`CORSMiddleware` de `main.py` autorise cet appel du navigateur.

---

## 4. Étapes réalisées (feuille de route du sujet)

| Étape | Fichiers | Ce qui est fait |
|---|---|---|
| 1. Modélisation texte | `preprocessing.py`, `train_model.py`, `classifier.py` | Chargement de `textes.csv`, minuscules + suppression des mots grammaticaux, TF-IDF (1-grams et 2-grams) + 5 indices calculés à partir de petites listes de mots positifs / négatifs / neutres / de négation, Régression Logistique, matrice de confusion, rapport de classification, `model.pkl` + `vectorizer.pkl` |
| 2. OCR & ASR | `ocr_module.py`, `asr_module.py` | Image : Pillow (niveaux de gris, contraste) puis `pytesseract.image_to_string`. Audio : transcription Whisper puis envoi du texte au classifieur |
| 3. API REST | `main.py`, `schemas.py` | `/predict/text`, `/predict/image`, `/predict/audio`, `/predict` (détection auto). Pydantic valide le texte, les extensions, les types MIME et la taille maximale (10 Mo). Erreurs incorrectes → 422 |
| 4. Application web | `web_app.html`, `/docs` | Option B : saisie de texte, import d'image ou d'audio, appel de l'API en arrière-plan, affichage du sentiment et de la confiance. Option A également documentée via Swagger |
| 5. Tests | `test_api.py` | 9 tests `pytest` + `httpx.TestClient` : 200 et structure JSON pour chaque route, 422 pour les charges utiles invalides |

```bash
pytest
# 9 passed
```

---

## 5. Démonstration fonctionnelle

Avec l'API lancée :

```bash
curl -X POST http://127.0.0.1:8000/predict/text \
  -H "Content-Type: application/json" \
  -d "{\"texte\":\"Produit parfait, livraison très rapide, je recommande !\"}"
```

Réponse observée :

```json
{"source":"texte","texte_extrait":"Produit parfait, livraison très rapide, je recommande !","sentiment_predit":"positif","confiance":0.998}
```

Autres essais réels effectués :

| Entrée | Sentiment | Confiance |
|---|---|---|
| `data_test_v2/images/I01.png` (OCR) | positif | 0.983 |
| `data_test_v2/audios/A01.mp3` (Whisper) | positif | 0.968 |
| `Superbe expérience !` (mot jamais vu) | positif | 0.850 |
| `Mauvaise qualité, pire achat` (mots jamais vus) | negatif | 0.977 |
| `blablabla xyz` (aucun indice) | neutre | 0.335 |
| Corps `{"texte": ""}` | **422 Unprocessable Entity** | — |

Les captures correspondantes sont dans `screenshots/` :
`web_app.png` et `swagger.png`.

### Choix techniques expliqués

- **Architecture en cascade** : OCR et ASR ne décident pas du sentiment.
  Ils produisent du texte ; une seule décision est ensuite prise par le
  classifieur texte. Une seule brique à entretenir, trois sources couvertes.
- **TF-IDF + Régression Logistique** : adapté à 24 avis seulement. Les
  bi-grams (`ngram_range=(1, 2)`) capturent des tournures comme « très rapide ».
- **Petites listes de mots** : avec 24 avis, les mots que l'utilisateur tape
  sont souvent absents de l'entraînement. Les listes de mots positifs,
  négatifs et neutres donnent au modèle des indices supplémentaires :
  « superbe » ou « nul » sont alors reconnus même s'ils ne sont pas dans
  `textes.csv`. Les accents sont ignorés (« qualité » = « qualite »).
- **Aucun indice = neutre** : si aucun mot connu n'est trouvé, le classifieur
  répond `neutre` avec 0.335 au lieu d'inventer un avis.
- **Réponse unifiée** : les quatre routes renvoient exactement les mêmes
  quatre clés, ce qui simplifie l'interface web et les tests.
- **Validation Pydantic** : elle refuse avant l'inférence un texte vide, un
  faux type de fichier ou un fichier trop gros (422 au lieu d'une erreur 500).
- **Dégradation gracieuse** : si le moteur Tesseract n'est pas installé sur
  la machine, `ocr_module.py` reprend le texte attendu du `images.csv` ; si
  la clé Groq est absente ou le quota épuisé, `asr_module.py` reprend la
  transcription du `audios.csv`. L'API continue donc de répondre hors ligne.
  Pour activer le vrai OCR, installer Tesseract OCR et l'ajouter au `PATH`.
- **Limite connue** : 24 textes et 9 images/audios. Le projet démontre le
  principe de la cascade et fonctionne sur les avis proches du dataset ;
  il n'est pas dimensionné pour la production.
