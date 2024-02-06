from transformers import pipeline
from transformers.pipelines import ImageToTextPipeline

model_mapper = {"blip-base":"Salesforce/blip-image-captioning-base",
                "blip-large":"Salesforce/blip-image-captioning-large",
                "git-base": "microsoft/git-base",
                "git-large": "microsoft/git-large-coco",
                "blip2": "Salesforce/blip2-opt-2.7b",
                "vit": "nlpconnect/vit-gpt2-image-captioning"}

generate_kwargs = {"min_length": 32,
                   "max_length": 64,
                   "repetition_penalty":1.5}


def one_sample_postprocess(self, model_outputs) -> str:
        # Postprocessing function for ImageToTextPipeline on per-sample basis. Batch support is removed in favour of
        # string output
        return self.tokenizer.decode(model_outputs.squeeze(0), skip_special_tokens=True)


class UnsupportedModelException(Exception):
    pass


class ModifiedPipeline:
      def __init__(self, model_name: str):
            try:
                 full_name = model_mapper[model_name]
            except KeyError:
                raise UnsupportedModelException(f'Not supported model {model_name}, please choose from {model_mapper.keys()}')
            self.model_name = model_name
            pip = pipeline("image-to-text", full_name)
            pip.postprocess = one_sample_postprocess.__get__(pip, pipeline) # monkey-patch the pipeline for cleaner output
            self.image2text_pipeline: ImageToTextPipeline = pip

      def __call__(self, *args, **kwargs):
            if 'generate_kwargs' not in kwargs:
                kwargs['generate_kwargs'] = generate_kwargs
            return self.image2text_pipeline(*args, **kwargs)
      
      def postprocess(self, model_outputs):
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
        if cls._cache is None or cls._cache.model_name != expected_model_name:
            modified = ModifiedPipeline(expected_model_name)
            # pip.postprocess = one_sample_postprocess.__get__(pip, pipeline) # monkey-patch the pipeline for cleaner output
            cls._cache = modified

        return cls._cache
