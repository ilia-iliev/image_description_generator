from transformers import pipeline, AutoProcessor, Trainer
from data.loader import load_ds
from evaluation import compute_metrics

DS_NAME = "lambdalabs/pokemon-blip-captions"

model_shortener = {"blip":"Salesforce/blip-image-captioning-base",
                    "git": "microsoft/git-base"}


def one_sample_postprocess(self, model_outputs) -> str:
        # Postprocessing function for ImageToTextPipeline on per-sample basis. Batch support is removed in favour of
        # string output
        return self.tokenizer.decode(model_outputs.squeeze(0), skip_special_tokens=True)


class CachedPipeline:
    """
    A class representing a cached pipeline for image-to-text transformation.

    Attributes:
        _cache (object): The cached pipeline object.

    Methods:
        get(expected_model_name): Retrieves the cached pipeline object or creates a new one if it doesn't exist or the expected pipeline name has changed.
    """
    _cache = None
    
    @classmethod
    def get(cls, expected_model_name):
        if cls._cache is None or cls._cache.model.name_or_path != expected_model_name:
            # loads pipeline that does all the wrangling around image-to-text transformation
            pip = pipeline("image-to-text", expected_model_name)
            pip.postprocess = one_sample_postprocess.__get__(pip, pipeline) # monkey-patch the pipeline for cleaner output
            cls._cache = pip

        return cls._cache
    
if __name__ == '__main__':
    pipeline = model_shortener.get("blip")
    ds = load_ds(DS_NAME)
    ds = ds["train"].train_test_split(test_size=3)
    preds = pipeline(ds['test']['image'])

    #wer.compute(predictions=[x[0]['generated_text'] for x in preds], references=ds['train']['text'])