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
├── asr_module.py         Étape 2 — transcription locale (faster-whisper)
├── schemas.py            Étape 3 — validation Pydantic (format, MIME, taille)
├── main.py               Étape 3 — application FastAPI et 4 routes
├── web_app.html          Étape 4, option B — mini-application web HTML/JS
├── test_api.py           Étape 5 — 11 tests pytest avec httpx.TestClient
├── start.bat             Raccourci Windows : lance l'API + la page web
├── test.bat              Raccourci Windows : lance les 11 tests
├── REPONSES.md             Questions probables à l'oral + réponses courtes
├── requirements.txt      Dépendances Python
├── .gitignore            Exclut __pycache__/, .pytest_cache/ et .env
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

### Raccourcis Windows (le plus simple)

- **Double-clic sur `start.bat`** : vérifie Python et les modèles, ouvre
  l'API et la page web dans deux fenêtres noires, puis ouvre les deux
  adresses dans le navigateur. Laisser les fenêtres ouvertes pendant la
  démonstration ; les fermer pour arrêter l'application.
- **Double-clic sur `test.bat`** : lance les 11 tests et affiche
  `RESULT: ALL TESTS PASSED`.

Les commandes manuelles équivalentes sont données ci-dessous.

---

## 2. Installation

Python 3.10 ou plus récent est nécessaire.

```bash
cd mini-projet-cascade
pip install -r requirements.txt
```

### Installation de Tesseract (OCR réel, obligatoire)

L'image est lue par le vrai Tesseract — **aucun retour au fichier `images.csv`**.
Sur Windows :

1. Télécharger/installer **UB-Mannheim TesseractOCR 5.4** (ou utiliser
   `winget install --id UB-Mannheim.TesseractOCR`).
2. Installer aussi le **pack de langue française** (`fra.traineddata`) dans
   `C:\Program Files\Tesseract-OCR\tessdata\`. Sans lui, la route image
   renvoie **503**.
3. Redémarrer le terminal après l'installation pour que le `PATH` soit à jour.
   `ocr_module.py` détecte aussi automatiquement
   `C:\Program Files\Tesseract-OCR\tesseract.exe` si le `PATH` n'est pas encore
   rafraîchi.

Vérification : `tesseract --list-langs` doit afficher `eng`, `fra`, `osd`.
Image illisible ou sans texte → **422** ; Tesseract absent → **503**.

### Installation de Whisper (ASR réel, obligatoire)

`asr_module.py` utilise **faster-whisper** en local : pas de clé API, pas de
`ffmpeg` système à installer sur Windows. Le modèle `base` est téléchargé
au premier appel (~100 Mo) puis mis en cache. **Faire ce premier appel avant
la démonstration**, avec une connexion qui fonctionne :

```bash
python -c "from asr_module import transcrire_audio; from pathlib import Path; print(transcrire_audio(Path('data_test_v2/audios/A01.mp3').read_bytes()))"
```

Si la transcription française est mauvaise, remplacer `"base"` par `"small"`
dans `asr_module.py` (plus lent). Aucun fallback CSV : si rien n'est entendu,
la route renvoie **422**.

Les modèles étant déjà fournis dans `models/`, l'entraînement n'est pas
obligatoire. Pour le rejouer :

```bash
python train_model.py
```

Le script affiche une **matrice de confusion par modèle** (lignes = vérité,
colonnes = prédiction), d'abord sur l'entraînement, puis en Leave-One-Out.

Modèle retenu — Régression Logistique (entraînement) :

```
                 negatif      neutre     positif
     negatif           8           0           0
      neutre           0           8           0
     positif           0           0           8
```

**Évaluation honnête (Leave-One-Out).** On retire chaque avis à tour de rôle,
on réentraîne sur les 23 autres, et on prédit l'avis retiré. Matrices LOO :

```
Régression Logistique        SVM linéaire (SVC)
[[8 0 0]                     [[8 0 0]
 [0 8 0]                      [0 8 0]
 [0 1 7]]                     [0 1 7]]
```

SVR (bonus) donne `[[8 0 0] [0 8 0] [0 3 5]]` : 3 positifs prédits neutres.

**Comparaison des modèles** demandée par le sujet (le PDF propose
« Régression Logistique ou SVM ») :

| Modèle | Acc. entraînement | Acc. LOO | F1 macro LOO |
|---|---|---|---|
| **Régression Logistique (retenu)** | 1.000 | **0.958** | **0.958** |
| SVM linéaire / SVC (comparaison demandée) | 1.000 | **0.958** | **0.958** |

Les deux modèles se trompent sur le même avis positif prédit `neutre`. La
Régression Logistique est retenue à égalité de LOO avec le SVM parce qu'elle
fournit des **probabilités**, donc le champ `confiance` de l'API. Le SVC par
défaut donne une décision sèche (`probability=True` serait plus lent).

**Bonus (hors comparaison demandée) : SVR.** Comme SVR est un modèle de
**régression**, la classe `PontSVR` code les classes en 0/1/2 puis arrondit la
valeur prédite. Résultat : 1.000 sur l'entraînement mais **0.875** en LOO
(3 avis positifs prédits neutres). Le script l'affiche aussi, pour montrer
qu'un modèle de régression est mal adapté ici — ce n'est pas le comparateur
du sujet, c'est le SVM de classification (SVC).

**23 avis sur 24 sont corrects** avec le modèle retenu ; un avis positif est
prédit `neutre`.

À titre de comparaison, le même test avec **TF-IDF seul** (sans les listes de
mots) donne **0.125**. Ce score très bas est un artefact connu d'un jeu
minuscule et parfaitement équilibré : retirer un exemple rend sa classe plus
rare et le modèle se trompe. Il montre surtout que ce sont les **listes de
mots** qui apportent la connaissance préalable et permettent de généraliser :
LOO passe de **0.125 à 0.958**. À dire soi-même à l'oral.

Le score de 1.00 affiché plus haut est mesuré sur les textes d'entraînement :
ce n'est **pas** une mesure de généralisation. C'est une vérification que le
modèle a bien appris les exemples fournis (choix conforme à l'étape 1 du
sujet, qui demande une matrice de confusion et un rapport de classification).

Vérification complémentaire : les 9 textes réellement lus dans les images par
Tesseract et les 9 transcriptions réellement produites par Whisper sont
classés correctement (18/18) par le même classifieur texte. Cela valide la
cascade OCR/ASR → texte → sentiment.

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
| 1. Modélisation texte | `preprocessing.py`, `train_model.py`, `classifier.py` | Chargement de `textes.csv`, minuscules + suppression des mots grammaticaux, TF-IDF (1-grams et 2-grams) + 5 indices calculés à partir de petites listes de mots positifs / négatifs / neutres / de négation, Régression Logistique, matrices de confusion, rapport de classification, comparaison SVM (SVC) et SVR, `model.pkl` + `vectorizer.pkl` |
| 2. OCR & ASR | `ocr_module.py`, `asr_module.py` | Image : Pillow (niveaux de gris, contraste) puis vrai Tesseract en français (`lang="fra"`) ; aucun retour au CSV. Audio : vraie transcription locale faster-whisper ; aucun retour au CSV. Le texte obtenu est envoyé au classifieur |
| 3. API REST | `main.py`, `schemas.py` | `/predict/text`, `/predict/image`, `/predict/audio`, `/predict` (détection auto). Pydantic valide le texte, les extensions, les types MIME et la taille maximale (10 Mo). Erreurs incorrectes → 422 |
| 4. Application web | `web_app.html`, `/docs` | Option B : saisie de texte, import d'image ou d'audio, appel de l'API en arrière-plan, affichage du sentiment et de la confiance. Option A également documentée via Swagger |
| 5. Tests | `test_api.py` | 11 tests `pytest` + `httpx.TestClient` : 200 et structure JSON pour chaque route, 422 pour les charges utiles invalides (texte vide, image blanche, image corrompue, mauvais format) |

```bash
pytest
# 11 passed
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
| `data_test_v2/images/I01.png` (vrai OCR Tesseract `fra`) | positif | 0.983 |
| `data_test_v2/audios/A01.mp3` (vrai Whisper local) | positif | 0.968 |
| `Superbe expérience !` (mot jamais vu) | positif | 0.850 |
| `Mauvaise qualité, pire achat` (mots jamais vus) | negatif | 0.977 |
| `blablabla xyz` (aucun indice) | neutre | 0.335 |
| Corps `{"texte": ""}` | **422 Unprocessable Entity** | — |

Les captures correspondantes sont dans `screenshots/` :
`web_app.png` et `swagger.png`.

Le fichier `REPONSES.md` reprend les 22 questions les plus probables à l'oral
avec une réponse courte et honnête à chacune, dont la comparaison avec le SVM.

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
- **Vrais modules, sans triche** : l'OCR vient de Tesseract et la transcription
  vient de Whisper. Les CSV ne servent que de vérité-terrain pour vérifier les
  résultats pendant le développement ; l'API ne les lit jamais. Une erreur
  est explicite : 503 si Tesseract manque, 422 si l'image est illisible ou
  si aucun texte/parole n'est détecté.
- **Évaluation Leave-One-Out** : 0.958 (23/24) contre 0.125 pour TF-IDF seul.
  C'est la preuve que les listes de mots, et non la mémorisation, permettent
  de généraliser. Le score 1.00 sur l'entraînement est donné à titre indicatif.
- **Comparaison SVM / SVR** : le PDF autorise « Régression Logistique ou SVM ».
  Les trois matrices et le tableau de comparaison sont produits par
  `train_model.py`. SVC arrive à égalité (0.958) mais ne fournit pas de
  probabilités ; SVR est moins bon (0.875). D'où le choix final.
- **Limite connue** : 24 textes et 9 images/audios. Le projet démontre le
  principe de la cascade et fonctionne sur les avis proches du dataset ;
  il n'est pas dimensionné pour la production. Le premier appel Whisper
  télécharge le modèle (~100 Mo) et met quelques secondes.
