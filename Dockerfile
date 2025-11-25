FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && \
   rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

COPY . /app/

WORKDIR /app

RUN pip install -r requirements.txt