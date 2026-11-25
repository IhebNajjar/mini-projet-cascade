"""Étape 2 — Module audio : transcription locale avec faster-whisper.

Le modèle est chargé une seule fois, à la première requête. Le texte vient
du fichier audio envoyé ; il n'y a aucun retour au fichier CSV.
"""

import io

from faster_whisper import WhisperModel

_model = None


def _get_model():
    global _model
    if _model is None:  # chargement unique, à la première demande
        _model = WhisperModel("base", device="cpu", compute_type="int8")
    return _model


def transcrire_audio(contenu: bytes, nom_fichier: str = "") -> str:
    """Renvoie le texte réellement entendu dans le message vocal."""
    segments, _ = _get_model().transcribe(io.BytesIO(contenu), language="fr")
    texte = " ".join(s.text.strip() for s in segments).strip()
    if not texte:
        raise ValueError("Aucune parole détectée")
    return texte
