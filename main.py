from typing import Union
from fastapi import FastAPI, UploadFile, HTTPException
from PIL import Image
from caption_extractor import CachedPipeline, UnsupportedModelException
from evaluation import evaluate_on_data


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/uploadimage/")
def create_upload_image(image: UploadFile, reference_caption: str):
    #TODO: check that it's an image, check size is up to 1024x1024
    return {"filename": image.filename}

@app.get("/evaluate/{model}")
async def evaluate(model: str):
    try:
        pipeline = CachedPipeline.get(model)
        return evaluate_on_data(pipeline)
    except UnsupportedModelException as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/image2caption/{model}")
async def image2caption(image: UploadFile, model: str):
    try:
        pipeline = CachedPipeline.get(model)
        converted_image = Image.open(image.file)
        return {"caption": pipeline(converted_image)}
    except UnsupportedModelException as e:
        raise HTTPException(status_code=404, detail=str(e))