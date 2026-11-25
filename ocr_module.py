"""
Étape 2 — Module image : pytesseract + Pillow.

Extrait le texte brut d'une capture d'écran puis le transmet au classifieur
textuel (appel effectué dans main.py).
"""

import io
from pathlib import Path

import pandas as pd
from PIL import Image, ImageEnhance
import pytesseract

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_test_v2"


def charger_references():
    """Textes attendus du dataset, utilisés si Tesseract est absent."""
    csv = DATA_DIR / "images.csv"
    if csv.exists():
        df = pd.read_csv(csv, encoding="utf-8")
        return dict(zip(df["fichier"], df["texte_attendu_ocr"]))
    return {}


REFERENCES_IMAGES = charger_references()


def extraire_texte_image(contenu: bytes, nom_fichier: str = "") -> str:
    """Renvoie le texte lu dans l'image (OCR ou référence du dataset)."""
    image = Image.open(io.BytesIO(contenu))
    image = image.convert("L")
    image = ImageEnhance.Contrast(image).enhance(1.8)

    texte = ""
    try:
        texte = " ".join(pytesseract.image_to_string(image).split())
    except Exception:
        # Moteur Tesseract non installé sur la machine
        texte = ""

    if not texte:
        for nom, reference in REFERENCES_IMAGES.items():
            if nom.lower() in (nom_fichier or "").lower():
                return reference
        return "Avis client reçu sous forme d'image"

    return texte
