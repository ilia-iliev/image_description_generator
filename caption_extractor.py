from transformers import pipeline
from transformers.pipelines import ImageToTextPipeline

model_mapper = {"blip-base":"Salesforce/blip-image-captioning-base",
                "blip-large":"Salesforce/blip-image-captioning-large",
                "git-base": "microsoft/git-base",
                "git-large": "microsoft/git-large-coco"}

# arguments that will be passed to generation phase of the pipeline. Currently, those force the model to
# provide longer captions which seems to make it more descriptive
generate_kwargs = {"min_length": 32,
                   "max_length": 64,
                   "repetition_penalty":1.5}


def one_sample_postprocess(self, model_outputs) -> str:
        """
        Postprocessing function for ImageToTextPipeline on per-sample basis. Batch inference is no longer supported but output
        is a single string instead of list of strings. This is okay since this batch inference is not expected for this PoC"""
        return self.tokenizer.decode(model_outputs.squeeze(0), skip_special_tokens=True)


class UnsupportedModelException(Exception):
    pass


class ModifiedPipeline:
    """
    A wrapper around the ImageToTextPipeline from the Hugging Face Transformers library.
    Provides a modified version of the pipeline with additional functionalities for image-to-text transformation.
    """

    def __init__(self, model_name: str):
        """
        Initializes an instance of the ModifiedPipeline class with the specified model name.

        Args:
            model_name (str): The name of the model to use for image-to-text transformation.

        Raises:
            UnsupportedModelException: If the specified model name is not supported.
        """
        self.model_name = model_name
        self.image2text_pipeline = self._get_pipeline(model_name)

    def __call__(self, *args, **kwargs):
        """
        Provides a callable interface to the pipeline. Accepts image paths or PIL images as input and returns the generated captions.
        """
        if 'generate_kwargs' not in kwargs:
            kwargs['generate_kwargs'] = generate_kwargs
        return self.image2text_pipeline(*args, **kwargs)

    @staticmethod
    def _get_pipeline(model_name: str) -> ImageToTextPipeline:
        """
        Gets the ImageToTextPipeline for the specified model name.

        Args:
            model_name (str): The name of the model.

        Returns:
            ImageToTextPipeline
        """
        try:
            full_name = model_mapper[model_name]
        except KeyError:
            raise UnsupportedModelException(f'Not supported model {model_name}, please choose from {model_mapper.keys()}')
        pip = pipeline("image-to-text", full_name)
        pip.postprocess = one_sample_postprocess.__get__(pip, pipeline)  # monkey-patch the pipeline for cleaner output
        return pip


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
            cls._cache = modified
        return cls._cache
