Proof of concept service which, given an image, generates an image description


```
curl -X 'POST' \
'http://localhost:8000/image2caption/blip' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@example.jpg;type=image/jpeg'
  ```