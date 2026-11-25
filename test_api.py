"""
Étape 5 — Tests unitaires et d'intégration avec pytest + httpx.TestClient.

    pytest
"""

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from main import app

client = TestClient(app)
DATA_DIR = Path(__file__).resolve().parent / "data_test_v2"

CLES = {"source", "texte_extrait", "sentiment_predit", "confiance"}


def verifier_structure(response):
    assert response.status_code == 200
    data = response.json()
    assert CLES <= set(data)
    assert data["sentiment_predit"] in {"positif", "negatif", "neutre"}
    assert 0.0 <= data["confiance"] <= 1.0
    return data


# ---- /predict/text ----

def test_texte_positif():
    data = verifier_structure(client.post(
        "/predict/text", json={"texte": "Produit parfait, livraison très rapide, je recommande !"}))
    assert data["source"] == "texte"
    assert data["sentiment_predit"] == "positif"


def test_texte_negatif():
    data = verifier_structure(client.post(
        "/predict/text", json={"texte": "Article cassé, service client injoignable, très déçu."}))
    assert data["sentiment_predit"] == "negatif"


def test_texte_vide_rejete():
    assert client.post("/predict/text", json={"texte": ""}).status_code == 422


# ---- /predict/image ----

def test_image_du_dataset():
    chemin = DATA_DIR / "images" / "I01.png"
    assert chemin.exists()
    with open(chemin, "rb") as f:
        data = verifier_structure(client.post(
            "/predict/image",
            files={"file": ("I01.png", f.read(), "image/png")},
        ))
    assert data["source"] == "image"
    assert data["texte_extrait"]


def test_image_format_interdit_rejete():
    response = client.post(
        "/predict/image",
        files={"file": ("document.txt", b"pas une image", "text/plain")},
    )
    assert response.status_code == 422


def test_image_blanche_rejetee():
    tampon = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(tampon, format="PNG")
    response = client.post(
        "/predict/image",
        files={"file": ("blanche.png", tampon.getvalue(), "image/png")},
    )
    assert response.status_code == 422


def test_image_corrompue_rejetee():
    response = client.post(
        "/predict/image",
        files={"file": ("cassee.png", b"pas une image", "image/png")},
    )
    assert response.status_code == 422


# ---- /predict/audio ----

def test_audio_du_dataset():
    chemin = DATA_DIR / "audios" / "A01.mp3"
    assert chemin.exists()
    with open(chemin, "rb") as f:
        data = verifier_structure(client.post(
            "/predict/audio",
            files={"file": ("A01.mp3", f.read(), "audio/mpeg")},
        ))
    assert data["source"] == "audio"
    assert len(data["texte_extrait"].split()) > 3


def test_audio_format_interdit_rejete():
    response = client.post(
        "/predict/audio",
        files={"file": ("script.py", b"pas un audio", "text/x-python")},
    )
    assert response.status_code == 422


# ---- /predict (détection automatique) ----

def test_detection_auto_texte():
    data = verifier_structure(client.post("/predict", json={"texte": "Superbe expérience !"}))
    assert data["source"] == "texte"


def test_detection_auto_json_vide_rejete():
    assert client.post("/predict", json={}).status_code == 422
