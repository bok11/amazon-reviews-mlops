import onnxruntime as rt
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, Request


app = FastAPI()


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
