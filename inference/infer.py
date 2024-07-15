import json
import time

from inference_utils import inference_time_csv_writer
from ultralytics import YOLO
# For more info visit: https://docs.ultralytics.com/modes/predict/


# Load config.json
with open("./inference_config.json", "r") as f:
    config = json.load(f)
print("Loaded config: ", config)

# TODO: if config["nms_wrapper"]

# Load model
model = YOLO(config['model_path'], task='detect')

# Inference
results = model(
    source=config['input_data_dir'],    # Input data directory
    conf=config['conf'],                # Confidence threshold
    iou=config['iou'],                  # NMS IoU threshold
    imgsz=config['img_size'],           # Inference size (pixels)
    half=config['half'],                         # Use FP16 half-precision inference
    device=config['device'],            # Device to use for inference
    save=True,                          # Images
    save_txt=True,                      # Text files
    save_conf=True,                     # Save confidences
    # Save results to project/name relative to script directory or absolute path
    project=config["output_dir"],
    name=time.strftime("%Y%m%d-%H%M%S") if config["name"] is None else config["name"],
)

# Save inference time to csv
inference_time_csv_writer(results, config["output_dir"], model.predictor.save_dir.name)
