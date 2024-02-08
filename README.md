Proof of concept service which, given an image, generates an image description. To run, make sure all requirements are installed. Start a server using:

```
uvicorn main:app --reload
```

To generate caption on example.jpeg, call the server:
```
curl -X 'POST' \
'http://localhost:8000/image2caption/blip-base' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@example.jpeg;type=image/jpeg'
  ```

To trigger an evaluation pipeline, call the server:

```
curl http://localhost:8000/evaluate/blip-base
```



Next Steps:
- Decide on metric. It's very difficult to make improvements without measuring them on pre-decided metric. The exact metric is tied to the exact task at hand - for example ROUGE-1 and ROUGE-2 are decent if we are looking for search keywords but have issues for more "creative" descriptions
- Parameter tuning - more permutations of the dictionary passed during the generation phase - `caption_extractor.generare_kwarg` could be tried. For example, a hyperparameter search could identify the best permutation of arguments. Results are expected to be model-specific and could slow down inference. The evaluation pipeline could be used for this step
- Fine-tune model:
  1. Decide on minimum satisfactory performance - metric and resources/speed of inference
  2. Generate captions for images that are more similar to in-house expectations. For instance: (https://huggingface.co/datasets/Ilia-Iliev/example_captioner)
  3. Fine-tune model using new dataset
  4. Evaluate using evaluate endpoint. Make sure metrics are improving at no large performance cost.
  5. Iterate on points 2 to 4 points above, if necessary
- An endpoint for adding training images could be added and integrated for the step above
- Alternatively, output could be modified by an additional prompt to another LLM to generate tags, add colors to the description, etc
- Experiment with blip2. This requires a GPU and/or more RAM. 
- Implement a server queue for expensive operations
- Add some safety guards to server. I.e. - don't allow users to trigger model evaluation or downloading models from HuggingFace. Alternatively, those could be manually triggered instead through the API.
- The service could be integrated with something like MLFlow for easier experiment result tracking
- Some unit tests and assertions are always a good idea. Could be auto-generated using something like Codium, but I decided that unit tests for PoC would be confusing. However, if the codebase grows, those would be very welcome.



Things tried that did not give satisfactory results:
1. User-prompt was provided to start of the decoding step (instead of the default start-of-sequence token). For example "the colors in the image are" or "hashtags: #". However, this seemed to always worsen results instead of guiding output, probably because this use case is not how image captioning models were trained
2. Blip2 could not be loaded in memory: https://huggingface.co/Salesforce/blip2-opt-2.7b. That seems promising since blip2 allows the aforementioned user-provided prompt since it does have an LLM in addition to the image-grounded encoder. It should be noted, that it might be possible to load some version of blip2 under this hardware. However, this would require special customization which is generally not a good idea for PoC. 
3. Doc strings were (mostly) auto-generated using Codium: https://www.codium.ai/
4. This project is my introduction to FastAPI, so apologies if there is a glaring issue with the service


Additional Notes:
- All experiments were run on an old machine with Intel i7-6500U CPU @ 2.50GHz and 8 GB RAM, without CUDA support and Python3.10

Here is table of metrics on the validation set, albeit only 5 images:
| | WER| ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-L-SUM | AVG INFERENCE TIME|
| ---------- | ----- | ----- | ----- | ----- | ----- | ------------ |
| blip-base  | 2.566 | 0.114 | 0.01  | 0.114 | 0.114 | 7.49 s
| blip-large | 2.509 | 0.161 | 0.0   | 0.113 | 0.114 | 10.603 s
| git-base   | 2.075 | 0.109 | 0.0   | 0.081 | 0.081 | 29.922 s
| git-large  | 2.509 | 0.119 | 0.009 | 0.111 | 0.109 | 105.956 s
