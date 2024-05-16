import json
import os

import numpy as np
from tqdm import tqdm

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from ultralytics import YOLO


def main(config):
    # Data paths
    sequences_dir = config["source_sequence_dir"]
    sequence_paths = [os.path.join(sequences_dir, name) for name in os.listdir(sequences_dir) if
                      os.path.isdir(os.path.join(sequences_dir, name))]

    # Output path
    output_root_dir = "./outputs/detections/" + config["name"] + "/" + sequences_dir.split('/')[-1]

    # TODO: use tensorrt weights
    # Generate detections for each sequence
    model = YOLO(config["source_weights_path"])

    # Infer on each sequence
    for sequence_path in sequence_paths:
        for img in tqdm(os.listdir((sequence_path + '/img1'))):
            if img.endswith(".jpg") or img.endswith(".png"):

                # Infer on image
                result = model.predict(
                    sequence_path + "/img1/" + img,
                    verbose=False,
                    conf=config["conf_threshold"],
                    iou=config["iou_threshold"],
                    imgsz=config["img_size"],
                    device=config["device"],
                    max_det=config["max_det"],
                    agnostic_nms=config["agnostic_nms"],
                    classes=[0],
                )[0]

                # Post-process detections
                mot_data = np.column_stack((
                    result.boxes.cls.cpu().numpy(),
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                ))

                # Save detections
                output_sequence_path = output_root_dir + "/" + sequence_path.split('/')[-1]
                os.makedirs(output_sequence_path, exist_ok=True)
                txt_path = output_sequence_path + f"/{img.split('.')[0]}.txt"

                np.savetxt(txt_path, mot_data, fmt='%.6f')


if __name__ == "__main__":
    with open("./cfg/detect.json", "r") as f:
        config = json.load(f)
    main(config)
