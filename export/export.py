import json
from ultralytics import YOLO

# Load config.json
with open("./export_config.json", "r") as f:
    config = json.load(f)
print("Loaded config: ", config)

# TODO: if comfig["include_nms"]:
# EXPORT USING NMS MODIFICAITONS

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


