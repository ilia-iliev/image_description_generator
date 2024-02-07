Proof of concept service which, given an image, generates an image description


```
curl -X 'POST' \
'http://localhost:8000/image2caption/blip' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@example.jpeg;type=image/jpeg'
  ```


Next Steps:
- Fine-tuned model:
  1. Decide on minimum satisfactory performance - metric and resources/speed of inference
  2. Generate captions for images that are more similar to in-house expectations. For instance: (https://huggingface.co/datasets/Ilia-Iliev/example_captioner)
  3. Fine-tune model using new dataset
  4. Evaluate. Make sure metrics are improving at no large performance cost.
  5. Iterate on points 2 to 4 points above, if necessary
- An endpoint for adding training images could be added and integrated for the step above
- Alternatively, output could be modified by an additional prompt to another LLM to generate tags, add colors to the description, etc
- Experiment with blip2. This requires a GPU and more RAM. 
- Implement a server queue for expensive operations
- Add some safety guards to server. I.e. - don't allow users to trigger model evaluation or downloading models from HuggingFace



Things tried that did not give satisfactory results:
1. User-prompt was provided to the language model (instead of the default start-of-sequence token). For example "the colors in the image are". However, this seemed to worsen results.
2. Blip2 could not be loaded in memory: https://huggingface.co/Salesforce/blip2-opt-2.7b. It should be noted, that it might be possible to load some version of blip2 under this setup but this would require special customization that could be pursued but is generally not a good idea for PoC
3. Doc strings were (mostly) auto-generated using Codium: https://www.codium.ai/
4. This project is my introduction to FastAPI, so apologies if there is a glaring issue with the API


Additional Notes:
- All experiments were run on an old machine with Intel i7-6500U CPU @ 2.50GHz and 8 GB RAM, without CUDA support. 
- A relatively cheap setup with GPU and more RAM can unlock blip2 which might be useful for giving more prompting control to the user
- Metrics are placeholders that could be replaced if needed. A more suitable metric could be found for the task at hand since ROUGE and WER do have issues.