from evaluate import load
from datasets import load_dataset
import time


wer = load("wer")
rouge = load("rouge")

def evaluate_metrics(preds, references):
    res = {}
    res['wer'] = wer.compute(predictions=preds, references=references)
    res.update(rouge.compute(predictions=preds, references=references))
    return res


def evaluate_on_data(model):
    ds = load_dataset("data/ilia_captioner")

    start = time.time()
    preds=model(ds['validation']['image'])
    avg_inference_time = (time.time()-start) / len(ds)

    evaluation_metrics = evaluate_metrics(preds, ds['validation']['text'])
    evaluation_metrics['avg_inference_time'] = avg_inference_time
    return evaluation_metrics