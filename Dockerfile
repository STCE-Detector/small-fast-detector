# TODO: Not working yet, must be fixed if client wants to use it in a containerized way

FROM python:3.8-slim
LABEL authors="ejbejaranos"
RUN apt update \
    && apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0

COPY ./ultralytics /app/src
COPY ./inference_tools /app/inference_tools
#WORKDIR /app/inference_tools

RUN pip install cmake
RUN pip install -r /app/inference_tools/inference_requirements.txt

#CMD ["python", "infer.py"]