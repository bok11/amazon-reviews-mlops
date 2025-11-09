# Use the official Python image as a base image
FROM python:3.13-slim

# which model to cupy
ARG MODEL="output/model.onnx"

# Set the working directory in the container
WORKDIR /app

#### Needed for ONNX Runtime to work, we put this first as it wont change
#### and by an early layer we dont need to rebuild it if we change something below
# install locale and libgomp1
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Generate en_US.UTF-8 and set as default
RUN sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
 && locale-gen en_US.UTF-8

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8
#########################################


# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./runtime .

COPY ${MODEL} ./model.onnx

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
