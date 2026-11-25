"""Étape 1 (suite) — Chargement des artefacts et inférence de sentiment."""

import pickle
from pathlib import Path

import numpy as np

from preprocessing import (
    MOTS_NEGATIFS,
    MOTS_NEUTRES,
    MOTS_POSITIFS,
    nettoyer,
)

BASE_DIR = Path(__file__).resolve().parent
MODELES_DIR = BASE_DIR / "models"


class ClassifieurSentiment:
    """Utilise model.pkl et vectorizer.pkl sauvegardés par train_model.py."""

    def __init__(self):
        with open(MODELES_DIR / "model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(MODELES_DIR / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

    def predire(self, texte: str):
        """Renvoie (classe, confiance) pour un texte brut."""
        texte_propre = nettoyer(texte)

        # Aucun mot connu : on ne peut rien dire, on reste neutre.
        mots = set(texte_propre.split())
        connus = set(self.vectorizer.tfidf.vocabulary_)
        lexique = MOTS_POSITIFS | MOTS_NEGATIFS | MOTS_NEUTRES
        if not (mots & connus) and not (mots & lexique):
            return "neutre", 0.335

        X = self.vectorizer.transform([texte_propre])
        classe = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confiance = float(np.max(probabilities))
        return classe, round(confiance, 3)


# Une seule instance, chargée une fois au démarrage.
classifieur = ClassifieurSentiment()
