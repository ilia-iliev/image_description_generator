Proof of concept service which, given an image, generates an image description


```
curl -X 'POST' \
'http://localhost:8000/image2caption/blip' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@example.jpg;type=image/jpeg'
  ```


Next Steps:
1. Better modelling:
  1a. Generate captions for images that are more similar to in-house expectations. Generate image from prompt, redact prompt (if necessary) so that it matches. Save for next steps
  1b. Fine-tune small model(s) using new dataset
  1c. Fine-tune larger model(s)
  1d. Evaluate on metrics
  1e. Evaluate on performance
  1f. Decide on minimum satisfactory performance. Choose the highest metric model.
2. Often the prompts used for generation do not describe the image and objects in it correctly. Why? What use case are we trying to solve?
3. Mixture-of-experts models
4. Return server is busy when doing expensive operations
5. Add dataset to huggingface