FROM ubuntu:latest

# FROM python:3.6

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3-pip \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py \
    && rm -rf /root/.cache

RUN pip install --upgrade pip

COPY requirements.txt /opt/program/requirements.txt

RUN pip3 install --no-cache-dir -r /opt/program/requirements.txt

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering
# our standard output stream, which means that logs can be delivered to the
# user quickly.
# PYTHONDONTWRITEBYTECODE keeps Python from writing the .pyc files which are
# unnecessary in this case.
# We also update PATH so that the train and serve programs are found when the
# container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENV MODEL_PATH="/opt/ml/model"

# Set up the program in the image
COPY model /opt/program
WORKDIR /opt/program

