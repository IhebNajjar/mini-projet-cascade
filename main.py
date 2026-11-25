"""
Étape 3 — Application FastAPI.

Routes :
    POST /predict/text    (corps JSON)
    POST /predict/image   (multipart/form-data)
    POST /predict/audio   (multipart/form-data)
    POST /predict         (détection automatique du type)

Documentation interactive : /docs et /redoc (Étape 4, option A).
"""

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from asr_module import transcrire_audio
from classifier import classifieur
from ocr_module import OCRIndisponible, extraire_texte_image
from schemas import FichierUpload, PredictionResponse, TexteRequest

app = FastAPI(
    title="Classifieur de Sentiment Multi-source",
    version="1.0.0",
    description="Texte, image (OCR) et audio (ASR) → un seul classifieur de sentiment.",
)

# Nécessaire pour que la page web_app.html puisse appeler l'API depuis le navigateur.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def repondre(source: str, texte: str):
    """Appelle le classifieur et construit la réponse JSON unifiée."""
    sentiment, confiance = classifieur.predire(texte)
    return PredictionResponse(
        source=source,
        texte_extrait=texte,
        sentiment_predit=sentiment,
        confiance=confiance,
    )


@app.post("/predict/text", response_model=PredictionResponse)
def predict_text(request: TexteRequest):
    """Prédiction depuis un texte brut envoyé en JSON."""
    return repondre("texte", request.texte)


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Prédiction depuis une image : OCR puis classifieur texte."""
    contenu = await file.read()
    try:
        valide = FichierUpload(
            nom=file.filename or "image.png",
            content_type=file.content_type or "",
            contenu=contenu,
            type_fichier="image",
        )
    except ValidationError as erreur:
        raise HTTPException(status_code=422, detail=str(erreur)) from erreur

    try:
        texte = extraire_texte_image(valide.contenu, valide.nom)
    except OCRIndisponible as erreur:
        raise HTTPException(status_code=503, detail=str(erreur)) from erreur
    except ValueError as erreur:
        raise HTTPException(status_code=422, detail=str(erreur)) from erreur
    return repondre("image", texte)


@app.post("/predict/audio", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    """Prédiction depuis un message vocal : ASR puis classifieur texte."""
    contenu = await file.read()
    try:
        valide = FichierUpload(
            nom=file.filename or "audio.wav",
            content_type=file.content_type or "",
            contenu=contenu,
            type_fichier="audio",
        )
    except ValidationError as erreur:
        raise HTTPException(status_code=422, detail=str(erreur)) from erreur

    try:
        texte = transcrire_audio(valide.contenu, valide.nom)
    except ValueError as erreur:
        raise HTTPException(status_code=422, detail=str(erreur)) from erreur
    return repondre("audio", texte)


@app.post("/predict", response_model=PredictionResponse)
async def predict_auto(request: Request, file: UploadFile = File(None)):
    """Détecte automatiquement le type de la charge utile."""
    content_type = request.headers.get("content-type", "").lower()

    if "application/json" in content_type:
        try:
            corps = await request.json()
            texte_demande = TexteRequest(**corps)
        except Exception as erreur:
            raise HTTPException(status_code=422, detail="JSON invalide") from erreur
        return repondre("texte", texte_demande.texte)

    if file is not None and file.filename:
        nom = file.filename.lower()
        mime = (file.content_type or "").lower()
        if nom.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")) or "image/" in mime:
            return await predict_image(file)
        if nom.endswith((".wav", ".mp3", ".ogg", ".m4a")) or "audio/" in mime:
            return await predict_audio(file)

    raise HTTPException(status_code=422, detail="Type d'entrée non reconnu")
