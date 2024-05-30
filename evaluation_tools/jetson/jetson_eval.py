import yaml
import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
from pathlib import Path
import torch

from evaluation_tools.jetson.coco_eval import COCOeval
from tracker.jetson.model.model import Yolov8
from ultralytics.utils import ops

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def evaluate(cocoGt_file, cocoDt_file):
    """Evaluate predictions using COCO metrics."""
    cocoGt = COCO(cocoGt_file)
    cocoDt = cocoGt.loadRes(cocoDt_file)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def custom_evaluate(cocoGt_file, cocoDt_file):
    """Custom evaluation with different area ranges."""
    anno = COCO(cocoGt_file)
    pred = anno.loadRes(cocoDt_file)
    eval = COCOeval(anno, pred, 'bbox')

    # Set Custom Area Ranges
    eval.params.areaRng = [[0 ** 2, 1e5 ** 2],
                           [0 ** 2, 16 ** 2],
                           [16 ** 2, 32 ** 2],
                           [32 ** 2, 96 ** 2],
                           [96 ** 2, 1e5 ** 2]]
    eval.params.areaRngLbl = ['all', 'tiny', 'small', 'medium', 'large']
    eval.params.maxDets = [3, 30, 300]

    eval.evaluate()
    eval.accumulate()
    eval.summarize()


def load_yaml(file_path):
    """Load a YAML configuration file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def load_images_from_folder(folder):
    """Load all images from a specified folder."""
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if any(img_path.endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
            img = np.array(cv2.imread(img_path)).astype(np.float32)
            if img is not None:
                images.append((img_path, img))
    return images


def pred_to_json(results, filename, class_map):
    """Serialize YOLO predictions to COCO json format."""
    jdict = []
    stem = Path(filename).stem
    image_id = int(stem) if stem.isnumeric() else stem

    for result in results:
        boxes = result.boxes
        bboxes = boxes.xyxy.cpu().numpy().astype(np.float32)  # Bounding boxes in xyxy format
        scores = boxes.conf.cpu().numpy().astype(np.float32)  # Confidence scores
        class_ids = boxes.cls.cpu().numpy().astype(np.float32)  # Class IDs

        # Convert xyxy to xywh
        bboxes_xywh = ops.xyxy2xywh(bboxes)
        bboxes_xywh[:, :2] -= bboxes_xywh[:, 2:] / 2  # Convert xy center to top-left corner

        for bbox, score, class_id in zip(bboxes_xywh, scores, class_ids):
            jdict.append(
                {
                    "image_id": image_id,
                    "category_id": int(class_id),  # Set category_id to the integer class ID
                    "bbox": [round(float(x), 3) for x in bbox],  # Ensure conversion to float
                    "score": round(float(score), 5),
                }
            )
    return jdict



def main(config_path, model_config):
    """Main function to generate predictions and save them in COCO format."""
    config = load_yaml(config_path)

    dataset_path = config['path']
    val_images_path = os.path.join(dataset_path, config['val'])
    category_map = {int(k): v for k, v in config['names'].items()}

    # Initialize YOLO model
    device = torch.device(model_config['device'], 0)
    yolov8 = Yolov8({
        'source_weights_path': model_config['source_weights_path'],
        'device': device
    }, labels=config['names'])

    # Load images
    images = load_images_from_folder(val_images_path)

    coco_results = {
        'annotations': [],
        'images': [],
        'categories': [{'id': int(k), 'name': v} for k, v in category_map.items()]
    }

    for img_id, (img_path, img) in tqdm(enumerate(images), total=len(images)):
        results = yolov8.predict(img)

        # Append image info
        coco_results['images'].append({
            'id': img_id,
            'file_name': os.path.basename(img_path),
            'width': int(img.shape[1]),
            'height': int(img.shape[0])
        })

        # Convert results to COCO format
        annotations = pred_to_json(results, img_path, category_map)
        coco_results['annotations'].extend(annotations)

    # Save full results to a JSON file
    full_output_file = 'full_coco_results.json'
    with open(full_output_file, 'w') as f:
        json.dump(coco_results, f, indent=4, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    # Save only annotations to a JSON file
    annotations_output_file = 'coco_results.json'
    with open(annotations_output_file, 'w') as f:
        json.dump(coco_results['annotations'], f, indent=4, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    print(f'Results saved to {full_output_file} and {annotations_output_file}')
    ground_truth_file = '../data/client_test/annotations/instances_val2017.json'
    detection_file = annotations_output_file

    # Call the custom_evaluate function
    custom_evaluate(ground_truth_file, detection_file)


if __name__ == "__main__":
    config_path = '../data/client_test/data.yaml'
    model_config = {
        "source_weights_path": "../detectors/model_v2.engine",
        "device": "cuda"
    }
    main(config_path, model_config)
