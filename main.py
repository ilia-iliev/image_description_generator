from typing import Union
from fastapi import FastAPI, UploadFile
from PIL import Image
from caption_extractor import CachedPipeline, model_shortener


app = FastAPI()
model = None

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/uploadimage/")
def create_upload_image(image: UploadFile):
    #TODO: check that it's an image, check size is up to 1024x1024
    return {"filename": image.filename}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/image2caption/{model}")
async def image2caption(image: UploadFile, model: str):
    full_name = model_shortener.get(model)
    pipeline = CachedPipeline.get(full_name)
    converted_image = Image.open(image.file)
    return {"caption": pipeline(converted_image)}
