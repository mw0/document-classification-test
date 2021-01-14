FROM ubuntu:latest

MAINTAINER speedy

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3 \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get install python3-pip

RUN pip install --upgrade pip
COPY requirements.txt /tmp
RUN pip install -r requirements.txt

# RUN pip install numpy scipy==1.1.0 scikit-learn==0.22.2 pandas numpy datetime re math os pickle warnings time pyarrow boto3 logging nltk xlrd flask gevent gunicorn && \
# RUN pip install numpy scipy==1.1.0 scikit-learn==0.22.2 boto3 logging flask gunicorn && \
RUN rm -rf /root/.cache

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY Classifier /opt/code
WORKDIR /opt/code

