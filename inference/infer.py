import json
import time

from inference_utils import inference_time_csv_writer
from ultralytics import YOLO


# Load config.json
with open("./inference_config.json", "r") as f:
    config = json.load(f)
print("Loaded config: ", config)

if config["nms_wrapper"]:
    from inference.nms_wrapper_infer import nms_wrapper_infer
    nms_wrapper_infer(config)
else:
    # Load model
    model = YOLO(config['model_path'], task='detect')
    results = model(
        source=config["input_data_dir"],  # Input image
        conf=config['conf'],  # Confidence threshold
        iou=config['iou'],  # NMS IoU threshold
        imgsz=config['img_size'],  # Inference size (pixels)
        half=config['half'],  # Use FP16 half-precision inference
        device=config['device'],  # Device to use for inference
        save=True,  # Save images
        save_txt=True,  # Save text files
        save_conf=True,  # Save confidences
        project=config["output_dir"],  # Save results to project/name
        name=time.strftime("%Y%m%d-%H%M%S") if config["name"] is None else config["name"],
    )

    # Save inference time to csv
    inference_time_csv_writer(results, config["output_dir"], model.predictor.save_dir.name)
