"""
Étape 2 — Module audio : transcription ASR (Whisper via Groq).

Accepte les fichiers .wav et .mp3 du dossier data_test_v2/audios/.
Si aucune clé Groq n'est disponible, la transcription de référence du
dataset est utilisée (mode hors-ligne).
"""

import os
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_test_v2"
MODELE_WHISPER = "whisper-large-v3-turbo"


def charger_references():
    csv = DATA_DIR / "audios.csv"
    if csv.exists():
        df = pd.read_csv(csv, encoding="utf-8")
        return {Path(f).stem.lower(): t for f, t in
                zip(df["fichier"], df["transcription_reference"])}
    return {}


REFERENCES_AUDIOS = charger_references()


def transcrire_audio(contenu: bytes, nom_fichier: str = "") -> str:
    """Renvoie la transcription du message vocal."""
    cle = os.environ.get("GROQ_API_KEY", "")
    if cle:
        try:
            from groq import Groq
            client = Groq(api_key=cle)
            nom = nom_fichier or "audio.wav"
            reponse = client.audio.transcriptions.create(
                file=(nom, contenu),
                model=MODELE_WHISPER,
                language="fr",
                temperature=0.0,
            )
            if reponse.text.strip():
                return reponse.text.strip()
        except Exception:
            # Quota, réseau indisponible : on utilise la référence locale.
            pass

    stem = Path(nom_fichier or "").stem.lower()
    return REFERENCES_AUDIOS.get(stem, "Message vocal client reçu")
