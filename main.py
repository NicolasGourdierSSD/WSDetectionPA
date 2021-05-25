from typing import Optional
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel
import cv2 # opencv
import numpy as np

from PIL import Image

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None

@app.get("/")
async def read_root():
    return {"message": "Welcome from the API"}

@app.post("/get")
async def get_pa(file: UploadFile = File(...)): # get pression artérielle
    image = np.array(Image.open(file.file))
    width, height, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {"Width":width,"Height":height}


# class Analyzer(BaseModel):
#     filename: str
#     img_dimensions: str
#     encoded_img: str

# @app.post("/analyze", response_model=Analyzer)
# async def analyze_route(file: UploadFile = File(...)):
#     contents = await file.read()
#     nparr = np.fromstring(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     img_dimensions = str(img.shape)
#     return_img = processImage(img)

#     # line that fixed it
#     _, encoded_img = cv2.imencode('.PNG', return_img)

#     encoded_img = base64.b64encode(encoded_img)

#     return{
#         'filename': file.filename,
#         'dimensions': img_dimensions,
#         'encoded_img': endcoded_img,
#     }