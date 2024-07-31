import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
from ultralytics import YOLO


# Load config.json
with open("./eval_config.json", "r") as f:
    config = json.load(f)
print("Loaded config: ", config)

# If nms_wrapper is True, run the nms_wrapper_eval.py script
if config['nms_wrapper']:
    from evaluation.nms_wrapper_eval import yolo_wrapper_validation
    yolo_wrapper_validation(config)

# If nms_wrapper is False, run the default YOLO model using ultralytics
else:
    #  START OF EVALUATION
    print("🚀...WELCOME TO EVALUATION DETECTOR MODEL...")

    print("🚀...Initializing model...")
    model = YOLO(config["model_path"], task='detect')

    print("🚀...INFERENCE MODE...🚀")
    print("📦...GETTING PREDICTIONS...📦")
    metrics = model.val(
        data=config["input_data_dir"],
        imgsz=config["img_size"],
        batch=config["batch"],
        device=config["device"],
        iou=config["iou"],
        half=config["half"],
        save=True,
        save_json=True,
        plots=True,
        save_txt=False,      # Text files
        save_conf=False,     # Save confidences
        # Save results to project/name relative to script directory or absolute path
        project=config["output_dir"],
        name=time.strftime("%Y%m%d-%H%M%S") if config["name"] is None else config["name"],
    )
