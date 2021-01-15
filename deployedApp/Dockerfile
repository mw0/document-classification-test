FROM ubuntu:latest

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3-pip \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py \
    && rm -rf /root/.cache

RUN pip install --upgrade pip

COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip install -r requirements.txt

RUN rm -rf /root/.cache

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
RUN echo "PATH: $PATH"

RUN echo ""
# Set up the program in the image
COPY Classifier /opt/program/Classifier
RUN echo "Done copying Classifier to /opt/program/."
COPY serve /opt/program/
RUN echo "Done copying serve to /opt/program/."
WORKDIR /opt/program
RUN echo "WORKDIR set to $WORKDIR."
