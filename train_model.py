"""
Étape 1 — Modélisation et entraînement du classifieur texte.

    python train_model.py

Lit data_test_v2/textes.csv, nettoie, vectorise (TF-IDF), entraîne une
Régression Logistique, affiche la matrice de confusion et le rapport de
classification, puis sauvegarde model.pkl et vectorizer.pkl dans models/.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import VectoriseurSentiment, nettoyer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_test_v2"
MODELES_DIR = BASE_DIR / "models"


def main():
    # 1. Chargement du jeu de données fourni
    df = pd.read_csv(DATA_DIR / "textes.csv", encoding="utf-8")
    textes = [nettoyer(t) for t in df["texte"]]
    verites = df["sentiment_ref"].tolist()
    classes = sorted(set(verites))
    print(f"Textes chargés : {len(textes)}")
    print(f"Classes        : {classes}")

    # 2. Vectorisation TF-IDF
    vectorizer = VectoriseurSentiment()
    X = vectorizer.fit_transform(textes)
    print(f"Dimensions     : {X.shape[1]} caractéristiques retenues")

    # 3. Modèle de classification supervisée
    modele = LogisticRegression(max_iter=1000, random_state=0)
    modele.fit(X, verites)

    # 4. Évaluation : matrice de confusion et rapport de classification
    predictions = modele.predict(X)
    print("\nMatrice de confusion (lignes = vérité, colonnes = prédiction)")
    print(f"{'':>12}" + "".join(f"{c:>12}" for c in classes))
    for vrai, ligne in zip(classes, confusion_matrix(verites, predictions, labels=classes)):
        print(f"{vrai:>12}" + "".join(f"{v:>12}" for v in ligne))
    print("\nRapport de classification")
    print(classification_report(verites, predictions, labels=classes))

    # 5. Persistance des artefacts
    MODELES_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELES_DIR / "model.pkl", "wb") as f:
        pickle.dump(modele, f)
    with open(MODELES_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("Artefacts sauvegardés : models/model.pkl, models/vectorizer.pkl")


if __name__ == "__main__":
    main()
