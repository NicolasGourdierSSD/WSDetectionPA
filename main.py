from typing import Optional
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
import numpy as np
import cv2
import time

from DetectionTension import detecterTensions
from DetectionTensionCropped import detecterTensionsCropped
from ScannerDocument import scannerDocument

import base64

app = FastAPI()

@app.get("/")
async def read_root():
    return {"/detecterPA": "Détecter la pression artérielle affichée sur un appareil de mesure de tension.",
            "/scanDocument": "Recadrer une image d'un document papier."}


@app.post("/detecterTension")
async def get_tension(file: UploadFile = File(...)): # get pression artérielle
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    tempsDebut = time.time()
    PAS, PAD = detecterTensions(image)
    tempsDeCalcul = (time.time() - tempsDebut)
    status = "Unknown" if PAS == 0 else "Ok"
    choosed = False if PAS == 0 else True
    return {"Choosed":choosed, # par défaut faux?
            "Status":status,
            "Systolic":PAS,
            "Diastolic":PAD,
            "ElapsedTime":tempsDeCalcul,
            "UsedMethods":"Unknown"}
    
@app.post("/detecterTensionRecadre")
async def get_tension_recadre(file: UploadFile = File(...)): # get pression artérielle
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    tempsDebut = time.time()
    PAS, PAD = detecterTensionsCropped(image)
    tempsDeCalcul = (time.time() - tempsDebut)
    status = "Unknown" if PAS == 0 else "Ok"
    choosed = False if PAS == 0 else True
    return {"Choosed":choosed, # par défaut faux?
            "Status":status,
            "Systolic":PAS,
            "Diastolic":PAD,
            "ElapsedTime":tempsDeCalcul,
            "UsedMethods":"Unknown"}

@app.post("/scanDocument")
async def scan_document(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    scanned = scannerDocument(image)
    
    encoded_img = cv2.imencode('.png', scanned)[1].tobytes()
    
    return Response(content = encoded_img, media_type = "image/png")