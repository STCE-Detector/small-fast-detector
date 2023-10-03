# README TO RUN INFERENCE 🚀
This README provides instructions on how to run inference and export using a Docker image. Follow the steps below to build and run the Docker image:
## 👨🏽‍💻 Prerequisites 
- Docker installed on your system.

## 👻 Steps
### 1. Build the Docker image using the following command:
```bash
docker build -t detector_onnx .
```
This command builds a Docker image from the Dockerfile in the current directory and tags the image as detector_onnx.
If you are running the Docker image in Apple M1/M2 processor, you should use the following command:
```bash
docker build -t detector_onnx . --platform linux/amd64
```
The image is build using the requirements.txt file located inside /inference_tools folder. This file matches the specified software 
requirements made by the client.
### 2. Run the Docker image using the following command:
``` bash
sudo docker run -it -v absolute_path_to/inference_tools/models:/app/models -v absolute_path_to/inference_tools/outputs:/app/outputs detector_onnx
```
Notice that two volumes are mounted in the container to store the outputs and the models produced by both inference 
(infer.py) and exportation (export.py). The first volume is mounted in the /app/models directory of the container and 
the second volume is mounted in the /app/outputs directory of the container. The absolute paths to the models and outputs 
directories in your system should be provided in the command.

This command starts a new container from the onnx image and opens an interactive terminal session (-it option).
### 3. Run export.py to export the model to ONNX format (Optional):
Inside the interective terminal session, run the following command:
``` bash
python export.py
```
This command exports the /inference_tools/models/custom_best.pt model to /inference_tools/models/custom_best.onnx.
This step is optional because the model is already exported to ONNX format and stored in the models directory. 
Nonetheless, for a proper inference, the model should have been exported to ONNX format using the same software version
as the one used to run inference. This is because the ONNX format is not backward compatible.
### 4. Run infer.py to run inference on a set of images:
Inside the interective terminal session, run the following command:
``` bash
python infer.py
```
This command runs inference on the images located in the /inference_tools/images/test directory and stores the outputs 
in the outputs directory. The outputs are: original images with bounding boxes, a text file with detection results for 
each image in YOLO format and a csv with inference time for each image.