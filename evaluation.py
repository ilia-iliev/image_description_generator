from evaluate import load
from datasets import load_dataset
import time
from pydantic import BaseModel, validator


wer = load("wer")
rouge = load("rouge")

ds = load_dataset("Ilia-Iliev/example_captioner")
VALIDATION_DATASET = ds['validation']


class EvalMetrics(BaseModel):
    wer: float
    rouge1: float
    rouge2: float
    rougeL: float
    rougeLsum: float
    avg_inference: float

    @validator('*')
    def round_float(cls, v):
        if isinstance(v, float):
            return round(v, 3)
        return v

def evaluate_metrics(preds, references):
    """
    Compute evaluation metrics for predicted captions.

    Args:
        preds (list): A list of predicted captions.
        references (list): A list of reference captions.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    res = rouge.compute(predictions=preds, references=references)
    res['wer'] = wer.compute(predictions=preds, references=references)
    return res


def evaluate_on_data(model) -> EvalMetrics:
    """
    Evaluates the performance of a model on a dataset.

    Args:
        model: An instance of a model that can generate predictions for input images.

    Returns:
        A dictionary containing the evaluation metrics
    """

    start = time.time()
    preds=model(VALIDATION_DATASET['image'])
    avg_inference = (time.time()-start) / len(preds)

    evaluation_metrics = evaluate_metrics(preds, VALIDATION_DATASET['text'])
    evaluation_metrics['avg_inference'] = avg_inference
    return EvalMetrics(**evaluation_metrics)