"""
Étape 3 — Schémas Pydantic : validation du texte, des formats,
des types MIME et de la taille des fichiers.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

TAILLE_MAX = 10 * 1024 * 1024  # 10 Mo

MIMES_IMAGE = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"}
EXTENSIONS_IMAGE = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

MIMES_AUDIO = {"audio/wav", "audio/x-wav", "audio/wave", "audio/mpeg", "audio/mp3", "audio/ogg"}
EXTENSIONS_AUDIO = {".wav", ".mp3", ".ogg", ".m4a"}


class TexteRequest(BaseModel):
    """ Corps JSON attendu par /predict/text. """
    texte: str = Field(..., min_length=2, description="Avis client à classifier")

    @model_validator(mode="after")
    def texte_non_vide(self):
        if not self.texte.strip():
            raise ValueError("Le texte ne peut pas être vide")
        self.texte = self.texte.strip()
        return self


class FichierUpload(BaseModel):
    """ Validation d'un fichier envoyé en multipart/form-data. """
    nom: str
    content_type: str
    contenu: bytes
    type_fichier: Literal["image", "audio"]

    @model_validator(mode="after")
    def verifier_fichier(self):
        extension = Path(self.nom).suffix.lower()
        mime = (self.content_type or "").lower()

        if self.type_fichier == "image":
            extensions, mimes = EXTENSIONS_IMAGE, MIMES_IMAGE
        else:
            extensions, mimes = EXTENSIONS_AUDIO, MIMES_AUDIO

        if extension not in extensions and mime not in mimes:
            raise ValueError(f"Format {self.type_fichier} non supporté ({mime}, {extension})")
        if len(self.contenu) == 0:
            raise ValueError("Le fichier envoyé est vide")
        if len(self.contenu) > TAILLE_MAX:
            raise ValueError("Le fichier dépasse la limite de 10 Mo")
        return self


class PredictionResponse(BaseModel):
    """ Réponse JSON unifiée, commune aux quatre routes. """
    source: str
    texte_extrait: str
    sentiment_predit: str
    confiance: float = Field(..., ge=0.0, le=1.0)
