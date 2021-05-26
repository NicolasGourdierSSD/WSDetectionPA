from typing import Optional
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel
import cv2 # opencv
import numpy as np

from PIL import Image

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome from the API"}

@app.post("/get")
async def get_pa(file: UploadFile = File(...)): # get pression artérielle
    image = np.array(Image.open(file.file))
    width, height, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {"Width":width,"Height":height}