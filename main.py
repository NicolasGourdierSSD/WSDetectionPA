from typing import Optional
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel
import numpy as np
import cv2

from DetectionTension import detecterTensions

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome from the API"}


@app.post("/get")
async def get_pa(file: UploadFile = File(...)): # get pression artérielle
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    width, height, _ = image.shape
    PAS, PAD = detecterTensions(image)
    return {"PAS":PAS,"PAD":PAD}
