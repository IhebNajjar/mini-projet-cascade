"""Étape 1 — Nettoyage du texte et vectorisation avec indices de sentiment.

Deux sources d'informations sont envoyées au classifieur :
  1. les mots / groupes de mots TF-IDF appris sur les 24 avis ;
  2. cinq nombres simples calculés à partir de petites listes de mots
     positifs, négatifs, neutres et de négation (utile quand l'utilisateur
     écrit un mot absent des 24 avis, par exemple « superbe » ou « nul »).
"""

import re
import unicodedata

from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

# Mots purement grammaticaux : on les retire avant la vectorisation.
# Les mots de négation sont gardés et servent à calculer l'indice `negation`.
STOP_WORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "au", "aux",
    "ce", "cet", "cette", "ces", "que", "qui", "quoi", "dont", "où",
    "se", "son", "sa", "ses", "leur", "leurs", "y", "en",
    "l", "s", "qu", "c", "n", "j",
}

MOTS_POSITIFS = {
    "bon", "bonne", "excellent", "excellente", "parfait", "parfaite",
    "recommande", "recommander", "recommandation", "satisfait", "satisfaite",
    "ravi", "ravie", "content", "contente", "heureux", "heureuse",
    "magnifique", "super", "superbe", "genial", "geniale", "top", "nickel",
    "impeccable", "rapide", "express", "incroyable", "elegant", "elegante",
    "soigne", "soignee", "cadeau", "apprecie", "appreciee", "bluffe",
    "bluffee", "bravo", "merci", "etoiles", "intuitif", "intuitive",
    "fluide", "exemplaire", "resolu", "resolue", "fiable",
}

MOTS_NEGATIFS = {
    "casse", "cassee", "decu", "decue", "mauvais", "mauvaise", "arnaque",
    "pire", "nul", "nulle", "horrible", "insatisfait", "mecontent",
    "mecontente", "fragile", "fissure", "fissuree", "retard", "complique",
    "compliquee", "impossible", "erreur", "injoignable", "inadmissible",
    "inutilisable", "defectueux", "defectueuse", "probleme", "panne",
    "souci", "litige", "attente", "fuyez", "scandaleux", "ecrase",
    "ecrasee", "abime", "abimee", "manquant", "manquante", "debite",
    "incompatibilite", "echec", "defaillance", "lent", "lente", "retard",
    "remboursement", "rembourse", "rembourses",
}

# « bien reçu », « conforme », « colis » : ce sont des mots neutres,
# pas des compliments. Ils aident à ne pas transformer une confirmation
# de réception en avis positif.
MOTS_NEUTRES = {
    "conforme", "reception", "receptionnee", "recupere", "recuperee",
    "boutique", "suivi", "colis", "commande", "matin", "delais", "standard",
    "correct", "travail", "catalogue", "notice", "particulier",
    "specifications", "semaine", "prevu", "relais", "dimensions",
    "installation", "livree", "point", "statut", "modele", "paquet",
    "fiche", "cable", "photo", "heures", "indique",
}

MOTS_NEGATION = {"pas", "ne", "jamais", "non", "rien", "aucun", "aucune"}


def _sans_accent(mot: str) -> str:
    """'qualité' devient 'qualite' : les deux écritures comptent pareil."""
    return "".join(
        c for c in unicodedata.normalize("NFD", mot)
        if unicodedata.category(c) != "Mn"
    )


def nettoyer(texte: str) -> str:
    """Met en minuscules, retire la ponctuation et les mots grammaticaux."""
    texte = _sans_accent(str(texte).lower())
    texte = re.sub(r"[^a-z0-9'\-]", " ", texte)
    mots = [mot for mot in texte.split() if mot not in STOP_WORDS and len(mot) > 1]
    return " ".join(mots)


class VectoriseurSentiment(BaseEstimator, TransformerMixin):
    """TF-IDF + 5 indices de sentiment calculés à la main."""

    def __init__(self):
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)

    def fit(self, X, y=None):
        self.tfidf.fit(X)
        return self

    def transform(self, X):
        tfidf = self.tfidf.transform(X)
        indices = []
        for texte in X:
            mots = set(texte.split())
            positifs = len(mots & MOTS_POSITIFS)
            negatifs = len(mots & MOTS_NEGATIFS)
            neutres = len(mots & MOTS_NEUTRES)
            negation = int(bool(mots & MOTS_NEGATION))
            indices.append([
                positifs, negatifs, positifs - negatifs, negation, neutres,
            ])
        return hstack([tfidf, csr_matrix(indices)], format="csr")
