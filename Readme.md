# Introduction
The purpose of this repository is to demo a simplistic MLops setup
The goal is to show how you can train a model, build a runtime for inferenece and add the necesarry monitoring to have a baseline to imporve upon.


# How to run the project

As the project is publishes on GHCR, the simplest way is to run:

`docker run -it --rm -p 8000:8000 ghcr.io/bok11/amazon-review-runtime:latest`

Once the container is running you can use the API interactively by visiting http://localhost:8000/docs or curl the endpoint directly like this:

```
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "review": "string"
}'
```

The APi will then responds with a json message like this, predicting the rating between 1-5 based on the text provided:
```
{
  "predicted_rating": 1
}
```