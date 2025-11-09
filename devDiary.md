# First steps
I open my task, and orient myself.
I am asked to create a end-to-end machine learning system, to do predictions on the included dataset.
My immidiate thoughts are that i need to go through some high level steps:
- Explore the dataset, and create a validation set from the start.
- Based on my exploration, decide on a simple model to use as baseline.
- Build a runtime component to host the inference model

I also identify the non-functional requirement:
- The runtime must respond to queries in less than 300ms with a p99

After getting an overview i open a jupyter notebook for my exploration phase.

(see exploration.ipynb)

# Build baseline model
After seeing the data and making the split, I decide to use a simple linear regression model as my baseline. It will be quick to train and easy to implement in an inference engine.
I will use review stars as my target, and use vectorize the text using TF-IDF. This is a simple way to get started, but lacks awareness of semantics and word relationships.
It is a simple approach to assign value to each word based on frequency, and is ideal to use for a simple baseline we can beat in the future.

My goal is to have a model that can take the words, and save it to an ONNX model for consumtion and to prepare seperate artefact tracks for the inference software and the model development in the future.

(see train.py)


# Building a basic runtime
Now that i can succesfully train a model and save it, it is time to build an inference runtime.
Here i will keep it simple, and create a docker container with the ONNX model and host it as a rest API.

I use FastAPI and uvicorn to build a python based inference machine.

(see runtime/main.py)

# Adding monitoring
As we now have a basic endpoint to call, we want to get some visibility int performance and ensure we can measure if we meet our requirement of latency.
Therefore i add the basic endpoints of a counter so we can follow how much usage the endpoint have, and a histogram to show performance

(see runtime/main.py)


# Building tests
As we now have a basic concept i need to write a test ensuring future changes does not deteriorate the work done so far.
As we are building a docker container, the approach will be to run the container locally and then run a test in python.
For the test i will use the entirety of the dataset, and run concurrent connections to simulate some load, and make sure i use data that resembles real world data.
