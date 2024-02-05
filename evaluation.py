from evaluate import load
from datasets import load_dataset

wer = load("wer")
rouge = load("rouge")

def evaluate_metrics(preds, references):
    res = {}
    res['wer'] = wer.compute(predictions=preds, references=references)
    res.update(rouge.compute(predictions=preds, references=references))
    return res


def evaluate_on_data(model):
    ds = load_dataset("data/ilia_captioner")

    preds=model(ds['validation']['image'])

    return evaluate_metrics(preds, ds['validation']['text'])