from typing import Optional
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
import numpy as np
import cv2

from DetectionTension import detecterTensions
from ScannerDocument import scannerDocument

import base64

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome from the API"}


@app.post("/detecterPA")
async def get_pa(file: UploadFile = File(...)): # get pression artérielle
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    PAS, PAD = detecterTensions(image)
    return {"PAS":PAS,"PAD":PAD}

@app.post("/scanDocument")
async def scan_document(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    scanned = scannerDocument(image)
    
    encoded_img = cv2.imencode('.png', scanned)[1].tobytes()
    
    return Response(content = encoded_img, media_type = "image/png")