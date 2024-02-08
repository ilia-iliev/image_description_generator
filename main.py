from fastapi import FastAPI, UploadFile, HTTPException
from PIL import Image
from caption_extractor import CachedPipeline, UnsupportedModelException
from evaluation import evaluate_on_data, EvalMetrics


app = FastAPI()

@app.get("/evaluate/{model}")
async def evaluate(model: str) -> EvalMetrics:
    """
    Trigger an evaluation pipeline of the specified model on the validation dataset
    Parameters:
    - model (str): The name of the model to evaluate.

    Returns:
    - EvalMetrics: An dictionary-like object containing the evaluation metrics for the model.

    Raises:
    - HTTPException: If the specified model is not supported.
    """
    try:
        pipeline = CachedPipeline.get(model)
        return evaluate_on_data(pipeline)
    except UnsupportedModelException as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/image2caption/{model}")
async def image2caption(image: UploadFile, model: str):
    """
    Takes an uploaded image file and a model name as input. Uses a cached pipeline 
    to convert the image into a caption using the specified model.

    Args:
        image (UploadFile): The uploaded image file.
        model (str): The name of the model to use for image-to-caption conversion.

    Returns:
        dict: The caption generated from the image.

    Raises:
        HTTPException: If the specified model is not supported.
    """
    try:
        pipeline = CachedPipeline.get(model)
        converted_image = Image.open(image.file)
        caption = pipeline(converted_image)
        return {"caption": caption}
    except UnsupportedModelException as e:
        raise HTTPException(status_code=404, detail=str(e))