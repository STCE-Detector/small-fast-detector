# README TO RUN INFERENCE 🚀
This README provides instructions on how to run inference using a Docker image. Follow the steps below to build and run the Docker image:
## 👨🏽‍💻 Prerequisites 
- Docker installed on your system.

## 👻 Steps
1. Build the Docker image using the following command:
```bash
docker build -t onnx .
```
2. Run the Docker image using the following command:
```
sudo docker run -it onnx
```
This command starts a new container from the onnx image and opens an interactive terminal session (-it option).
4. Remember to put your set of images in some directory called images into the inference_tools directory.
