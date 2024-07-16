import json
import os

from nms.export_nms import nms_export
from ultralytics import YOLO
import subprocess


# Load config.json
with open("./export_config.json", "r") as f:
    config = json.load(f)
print("Loaded config: ", config)

# TODO: if comfig["include_nms"]:
# EXPORT USING NMS MODIFICAITONS
if config["include_nms"]:
    nms_export(config)
    original_model_path = config["model_path"]
    base_path, filename = os.path.split(original_model_path)
    name, ext = os.path.splitext(filename)
    onnx_path = os.path.join(base_path, f"{name}_nms.onnx")
    engine_path = os.path.join(base_path, f"{name}_nms.engine")
    command = f'''
        /usr/src/tensorrt/bin/trtexec --onnx="{onnx_path}" --saveEngine="{engine_path}" --fp16
        '''

    subprocess.run(command, shell=True, executable='/bin/bash')

else:
    print("🚀 Initializing model...")
    # Initialize and set up model
    # Load model
    model = YOLO(model=config["model_path"], task="detect")
    model.export(
        format=config["format"],
        imgsz=config["img_size"],
        opset=config["opset_version"],
        simplify=config["simplify_onnx"],
        dynamic=False,
        batch=1,
    )


