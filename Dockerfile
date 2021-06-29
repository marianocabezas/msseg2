FROM nvcr.io/nvidia/pytorch:20.02-py3

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