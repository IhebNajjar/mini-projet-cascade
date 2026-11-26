"""
Étape 1 — Modélisation et entraînement du classifieur texte.

    python train_model.py

Lit data_test_v2/textes.csv, nettoie, vectorise (TF-IDF + lexique), entraîne
une Régression Logistique, affiche la matrice de confusion et le rapport de
classification, compare avec SVM / SVR, puis sauvegarde model.pkl et
vectorizer.pkl dans models/.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR

from preprocessing import VectoriseurSentiment, nettoyer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_test_v2"
MODELES_DIR = BASE_DIR / "models"


class PontSVR(BaseEstimator):
    """Adapte SVR (régression) aux 3 classes, uniquement pour la comparaison.

    SVR prédit un nombre : on code negatif=0, neutre=1, positif=2, puis on
    arrondit la prédiction au code le plus proche. Ce modèle n'est PAS celui
    déployé : il sert seulement à répondre à la demande de comparaison.
    """

    def __init__(self, C=1.0, kernel="linear"):
        self.C = C
        self.kernel = kernel

    def fit(self, X, y):
        self.encodeur_ = LabelEncoder().fit(y)
        self.modele_ = SVR(C=self.C, kernel=self.kernel)
        self.modele_.fit(X, self.encodeur_.transform(y))
        return self

    def predict(self, X):
        codes = np.rint(self.modele_.predict(X)).astype(int)
        codes = np.clip(codes, 0, len(self.encodeur_.classes_) - 1)
        return self.encodeur_.inverse_transform(codes)


def afficher_matrice(titre, verites, predictions, classes):
    """Affiche une matrice de confusion avec les noms des classes."""
    print(f"\n{titre}")
    print(f"{'':>12}" + "".join(f"{c:>12}" for c in classes))
    for vrai, ligne in zip(
        classes, confusion_matrix(verites, predictions, labels=classes)
    ):
        print(f"{vrai:>12}" + "".join(f"{v:>12}" for v in ligne))


def main():
    # 1. Chargement du jeu de données fourni
    df = pd.read_csv(DATA_DIR / "textes.csv", encoding="utf-8")
    textes = [nettoyer(t) for t in df["texte"]]
    verites = df["sentiment_ref"].tolist()
    classes = sorted(set(verites))
    print(f"Textes chargés : {len(textes)}")
    print(f"Classes        : {classes}")

    # 2. Vectorisation TF-IDF + indices du lexique
    vectorizer = VectoriseurSentiment()
    X = vectorizer.fit_transform(textes)
    print(f"Dimensions     : {X.shape[1]} caractéristiques retenues")

    # 3. Modèles : celui retenu, puis deux baselines de comparaison
    modeles = [
        ("Régression Logistique (retenu)", LogisticRegression(max_iter=1000, random_state=0)),
        ("SVM linéaire / SVC (comparaison)", SVC(kernel="linear", C=1.0, random_state=0)),
        ("SVR (comparaison)", PontSVR(C=1.0, kernel="linear")),
    ]

    comparaison = []
    modele_retenu = None

    for nom, estimateur in modeles:
        # 4. Entraînement et évaluation sur les données d'entraînement
        pipeline = Pipeline([
            ("vec", VectoriseurSentiment()),
            ("clf", clone(estimateur)),
        ])
        pipeline.fit(textes, verites)
        predictions = pipeline.predict(textes)
        afficher_matrice(
            f"{nom} — matrice de confusion (entraînement)", verites, predictions, classes
        )
        acc_train = accuracy_score(verites, predictions)

        # 4b. Évaluation honnête : Leave-One-Out (un avis retiré à chaque fois)
        pred_loo = cross_val_predict(
            clone(pipeline), textes, verites, cv=LeaveOneOut()
        )
        afficher_matrice(
            f"{nom} — matrice de confusion (Leave-One-Out)", verites, pred_loo, classes
        )
        print("Rapport de classification (Leave-One-Out)")
        print(classification_report(verites, pred_loo, labels=classes, zero_division=0))

        comparaison.append((
            nom,
            acc_train,
            accuracy_score(verites, pred_loo),
            f1_score(verites, pred_loo, average="macro", zero_division=0),
        ))

        if nom.startswith("Régression Logistique"):
            # On sauvegarde le modèle ajusté sur les 24 textes, sans le Pipeline :
            # classifier.py applique déjà VectoriseurSentiment avant predict().
            modele_retenu = clone(estimateur).fit(X, verites)

    # 5. Tableau de comparaison demandé par le sujet
    print("\n=== Comparaison des modèles ===")
    print(f"{'Modèle':<36}{'Acc. train':>12}{'Acc. LOO':>12}{'F1 macro LOO':>14}")
    for nom, acc_train, acc_loo, f1_loo in comparaison:
        print(f"{nom:<36}{acc_train:>12.3f}{acc_loo:>12.3f}{f1_loo:>14.3f}")
    print(
        "\nChoix : la Régression Logistique, à égalité de LOO avec le SVM, "
        "parce qu'elle fournit des probabilités (champ 'confiance')."
    )

    # 6. Persistance des artefacts du modèle retenu uniquement
    MODELES_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELES_DIR / "model.pkl", "wb") as f:
        pickle.dump(modele_retenu, f)
    with open(MODELES_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("Artefacts sauvegardés : models/model.pkl, models/vectorizer.pkl")


if __name__ == "__main__":
    main()
