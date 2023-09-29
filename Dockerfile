FROM python:3.8-slim
LABEL authors="ejbejaranos"
RUN apt update \
    && apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0
WORKDIR /app
COPY ultralytics ultralytics

COPY inference_tools/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY inference_tools/images images
COPY inference_tools/models models
COPY inference_tools/export.py export.py
COPY inference_tools/infer.py infer.py


CMD ["python", "infer.py"]