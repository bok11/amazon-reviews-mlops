import onnxruntime as rt
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, generate_latest
import time


app = FastAPI()

REQUEST_COUNTER = Counter(
    "app_requests_total",  # Metric name
    "Total number of requests to the app",  # Metric description
    ["route", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "http_server_duration_seconds",
    "End-to-end HTTP request latency",
    ["route", "status_code"],
    buckets=(
        0.025,
        0.05,
        0.1,
        0.2,
        0.3,
        0.5,
        1.0,
        2.5,
        5.0,
    ),  # we define buckets in seconds, with a higher resolution for lower latencies as our target is 300ms
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    REQUEST_COUNTER.labels(request.url.path, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.url.path, str(response.status_code)).observe(
        duration
    )
    return response


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")


# we use a class to define the input data model
# this will be used by FastAPI to validate the input JSON
# and to generate the OpenAPI schema and trying out the API in the docs
class TextInput(BaseModel):
    review: str


# load the model into memory at startup
@app.on_event("startup")
async def startup_event():
    global sess, input_name, label_name, prob_name
    onnx_path = "model.onnx"
    sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name  # predicted class label
    prob_name = sess.get_outputs()[1].name  # class probabilities (dense)


# Predict the rating, and return it as JSON
@app.post("/predict")
async def root(input: TextInput):
    input_data = np.array([input.review], dtype=object)
    pred_label, _ = sess.run([label_name, prob_name], {input_name: input_data})
    return {"predicted_rating": int(pred_label[0])}
