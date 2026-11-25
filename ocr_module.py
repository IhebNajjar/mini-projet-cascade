"""Étape 2 — Module image : pytesseract + Pillow.

Le texte vient de l'image elle-même. Il n'y a aucun retour au fichier CSV :
si l'OCR est indisponible ou ne lit rien, une erreur explicite est renvoyée.
"""

import io
from pathlib import Path

import pytesseract
from PIL import Image, ImageEnhance

# Après installation de Tesseract sur Windows, le PATH n'est parfois mis à
# jour qu'au prochain redémarrage. On détecte donc l'emplacement standard.
if pytesseract.pytesseract.tesseract_cmd == "tesseract":
    for candidat in (
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ):
        if candidat.exists():
            pytesseract.pytesseract.tesseract_cmd = str(candidat)
            break


class OCRIndisponible(Exception):
    """Tesseract n'est pas installé ou ne peut pas être lancé."""


def extraire_texte_image(contenu: bytes, nom_fichier: str = "") -> str:
    """Renvoie le texte réellement lu dans l'image."""
    try:
        image = Image.open(io.BytesIO(contenu)).convert("L")
    except Exception as e:
        raise ValueError("Image illisible") from e

    image = ImageEnhance.Contrast(image).enhance(1.8)
    try:
        texte = " ".join(pytesseract.image_to_string(image, lang="fra").split())
    except (pytesseract.TesseractNotFoundError, pytesseract.TesseractError) as e:
        raise OCRIndisponible(
            "Tesseract avec le pack français n'est pas installé"
        ) from e

    if not texte:
        raise ValueError("Aucun texte détecté dans l'image")
    return texte
