"""
Génère les images des matrices de confusion et des graphiques de comparaison.

    python generer_graphiques.py

Produit :
    screenshots/confusion_matrices.png
    screenshots/comparaison_modeles.png

Les valeurs sont recalculées depuis data_test_v2/textes.csv avec le même
Leave-One-Out que train_model.py : aucune image n'est dessinée à la main.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from preprocessing import VectoriseurSentiment, nettoyer
from train_model import PontSVR

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_test_v2"
IMAGES_DIR = BASE_DIR / "screenshots"


def dessiner_matrice(fig, ax, matrice, classes, titre):
    """Dessine une matrice de confusion avec les valeurs dans les cases."""
    image = ax.imshow(matrice, cmap="Blues", vmin=0, vmax=8)
    ax.set_xticks(range(len(classes)), classes, rotation=20, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vérité")
    ax.set_title(titre, fontsize=11, fontweight="bold")
    for i in range(len(classes)):
        for j in range(len(classes)):
            valeur = matrice[i, j]
            ax.text(
                j, i, str(valeur),
                ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if valeur >= 5 else "#1f2937",
            )
    fig.colorbar(image, ax=ax, shrink=0.8)


def main():
    df = pd.read_csv(DATA_DIR / "textes.csv", encoding="utf-8")
    textes = [nettoyer(t) for t in df["texte"]]
    verites = df["sentiment_ref"].tolist()
    classes = sorted(set(verites))

    modeles = [
        ("Régression logistique", LogisticRegression(max_iter=1000, random_state=0)),
        ("SVM linéaire (SVC)", SVC(kernel="linear", C=1.0, random_state=0)),
        ("SVR (bonus)", PontSVR(C=1.0, kernel="linear")),
    ]

    # ---- Figure 1 : les 4 matrices de confusion demandées ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    resultats = []
    for ligne, (nom, estimateur) in enumerate(modeles[:2]):
        pipeline = Pipeline([
            ("vec", VectoriseurSentiment()),
            ("clf", clone(estimateur)),
        ])
        predictions_train = clone(pipeline).fit(textes, verites).predict(textes)
        predictions_loo = cross_val_predict(
            clone(pipeline), textes, verites, cv=LeaveOneOut()
        )
        dessiner_matrice(
            fig, axes[ligne, 0],
            confusion_matrix(verites, predictions_train, labels=classes),
            classes, f"{nom}\nEntraînement",
        )
        dessiner_matrice(
            fig, axes[ligne, 1],
            confusion_matrix(verites, predictions_loo, labels=classes),
            classes, f"{nom}\nLeave-One-Out",
        )
        resultats.append((
            nom,
            accuracy_score(verites, predictions_train),
            accuracy_score(verites, predictions_loo),
            f1_score(verites, predictions_loo, average="macro", zero_division=0),
        ))

    # Bonus SVR pour le deuxième graphique
    pipeline_svr = Pipeline([
        ("vec", VectoriseurSentiment()),
        ("clf", clone(modeles[2][1])),
    ])
    predictions_svr = cross_val_predict(
        pipeline_svr, textes, verites, cv=LeaveOneOut()
    )
    resultats.append((
        "SVR (bonus)", 1.0,
        accuracy_score(verites, predictions_svr),
        f1_score(verites, predictions_svr, average="macro", zero_division=0),
    ))

    fig.suptitle(
        "Matrices de confusion — 24 avis (8 par classe)",
        fontsize=16, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2 : graphiques de comparaison ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    noms = [r[0].replace(" ", "\n", 1) for r in resultats]
    acc_train = [r[1] for r in resultats]
    acc_loo = [r[2] for r in resultats]
    f1_loo = [r[3] for r in resultats]
    x = np.arange(len(noms))
    largeur = 0.26
    ax1.bar(x - largeur, acc_train, largeur, label="Acc. entraînement", color="#3b82f6")
    ax1.bar(x, acc_loo, largeur, label="Acc. Leave-One-Out", color="#10b981")
    ax1.bar(x + largeur, f1_loo, largeur, label="F1 macro LOO", color="#f59e0b")
    ax1.set_xticks(x, noms, fontsize=9)
    ax1.set_ylim(0.75, 1.03)
    ax1.set_ylabel("Score")
    ax1.set_title("Comparaison des modèles", fontweight="bold")
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(axis="y", alpha=0.3)
    for barre in ax1.patches:
        ax1.text(
            barre.get_x() + barre.get_width() / 2, barre.get_height() + 0.004,
            f"{barre.get_height():.3f}", ha="center", va="bottom", fontsize=7,
        )

    # Effet du lexique : avec vs sans les listes de mots
    pipeline_tfidf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, random_state=0)),
    ])
    pred_sans_lexique = cross_val_predict(
        pipeline_tfidf, textes, verites, cv=LeaveOneOut()
    )
    acc_sans_lexique = accuracy_score(verites, pred_sans_lexique)
    acc_avec_lexique = resultats[0][2]
    etiquettes = ["Avec lexique\n(retenu)", "Sans lexique\n(TF-IDF seul)"]
    valeurs = [acc_avec_lexique, acc_sans_lexique]
    barres = ax2.bar(etiquettes, valeurs, color=["#10b981", "#ef4444"], width=0.5)
    ax2.axhline(1 / 3, color="#6b7280", linestyle="--", linewidth=1,
                label="Hasard (3 classes)")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Accuracy Leave-One-Out")
    ax2.set_title("Effet des listes de mots", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    for barre, valeur in zip(barres, valeurs):
        ax2.text(
            barre.get_x() + barre.get_width() / 2, valeur + 0.015,
            f"{valeur:.3f}", ha="center", fontweight="bold",
        )

    fig.suptitle(
        "Résultats du Leave-One-Out (un avis retiré à chaque fois)",
        fontsize=15, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "comparaison_modeles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Images générées :")
    print("  screenshots/confusion_matrices.png")
    print("  screenshots/comparaison_modeles.png")
    print(f"  accuracy LOO avec lexique : {acc_avec_lexique:.3f}")
    print(f"  accuracy LOO sans lexique : {acc_sans_lexique:.3f}")


if __name__ == "__main__":
    main()
