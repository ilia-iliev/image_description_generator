from evaluate import load
from datasets import load_dataset
import time
from pydantic import BaseModel, validator


wer = load("wer")
rouge = load("rouge")

class EvalMetrics(BaseModel):
    wer: float
    rouge1: float
    rouge2: float
    rougeL: float
    rougeLsum: float
    avg_inference: float

    @validator('*', pre=True)
    def round_float(cls, v):
        if isinstance(v, float):
            return round(v, 3)
        return v

def evaluate_metrics(preds, references):
    res = {}
    res['wer'] = wer.compute(predictions=preds, references=references)
    res.update(rouge.compute(predictions=preds, references=references))
    return res


def evaluate_on_data(model):
    ds = load_dataset("data/ilia_captioner")

    start = time.time()
    preds=model(ds['validation']['image'])
    avg_inference = (time.time()-start) / len(ds)

    evaluation_metrics = evaluate_metrics(preds, ds['validation']['text'])
    evaluation_metrics['avg_inference'] = avg_inference
    return EvalMetrics(**evaluation_metrics)