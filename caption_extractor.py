from transformers import pipeline

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
        full_name = model_shortener.get(expected_model_name)
        if cls._cache is None or cls._cache.model.name_or_path != full_name:
            # loads pipeline that does all the wrangling around image-to-text transformation
            pip = pipeline("image-to-text", full_name)
            pip.postprocess = one_sample_postprocess.__get__(pip, pipeline) # monkey-patch the pipeline for cleaner output
            cls._cache = pip

        return cls._cache
