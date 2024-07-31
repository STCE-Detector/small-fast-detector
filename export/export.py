import json
import os

from nms.export_nms import torch2onnx_nms
from ultralytics import YOLO
import subprocess


# Load config.json
with open("./export_config.json", "r") as f:
    config = json.load(f)
print("Loaded config: ", config)

# Export model to onnx with NMS included as a node and then convert to TensorRT engine
if config["include_nms"]:
    # Export model to onnx with NMS included as a node
    torch2onnx_nms(
        weights=config["model_path"],
        output=config["output"],
        version="yolov8",
        imgsz=config["imgsz"],
        batch=config["batch"],
        max_boxes=config["max_boxes"],
        iou_thres=config["iou_thres"],
        conf_thres=config["conf_thres"],
        opset_version=config["opset"],
        slim=config["simplify"],
    )

    # Convert onnx to TensorRT engine
    original_model_path = config["model_path"]
    base_path, filename = os.path.split(original_model_path)
    name, ext = os.path.splitext(filename)
    onnx_path = os.path.join(base_path, f"{name}_nms.onnx")
    engine_path = os.path.join(base_path, f"{name}_nms.engine")
    command = f'''
        /usr/src/tensorrt/bin/trtexec --onnx="{onnx_path}" --saveEngine="{engine_path}" --fp16
        '''
    subprocess.run(command, shell=True, executable='/bin/bash')

# Export model to onnx or engine directly using ultralytics
else:
    print("🚀 Initializing model...")
    # Initialize and set up model
    # Load model
    model = YOLO(model=config["model_path"], task="detect")
    model.export(
        format=config["format"],
        imgsz=config["img_size"],
        opset=config["opset"],
        simplify=config["simplify"],
        dynamic=False,
        batch=1,
    )


