FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

MAINTAINER Mariano Cabezas <mariano.cabezas@sydney.edu.au>

COPY base.py /src/base.py
COPY models.py /src/models.py
COPY utils.py /src/utils.py
COPY criteria.py /src/criteria.py
COPY test_models.py /src/inference.py
COPY requirements.txt /src/requirements.txt
COPY ModelWeights /model

RUN pip install -r /src/requirements.txt && rm /workspace -rf && mkdir /workspace

WORKDIR /workspace