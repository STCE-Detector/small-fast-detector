# TODO: Not working yet, must be fixed if client wants to use it in a containerized way

FROM python:3.8-slim
LABEL authors="ejbejaranos"
RUN apt update \
    && apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0
WORKDIR /app
COPY ultralytics ultralytics

COPY inference_tools/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY inference_tools/ .


#CMD ["python", "infer.py"]