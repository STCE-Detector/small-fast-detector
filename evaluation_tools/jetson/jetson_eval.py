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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


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
            img = cv2.imread(img_path)
            if img is not None:
                images.append((img_path, img))
    return images


def pred_to_json(predn, filename, class_map):
    """Serialize YOLO predictions to COCO json format."""
    jdict = []
    stem = Path(filename).stem
    image_id = int(stem) if stem.isnumeric() else stem
    box = ops.xyxy2xywh(predn[:, :4])  # Convert xyxy to xywh
    box[:, :2] -= box[:, 2:] / 2  # Convert xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
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
    device = torch.device(model_config['device'], 2)
    yolov8 = Yolov8({
        'source_weights_path': model_config['source_weights_path'],
        'device': device
    }, labels=config['names'])

    # Load images
    images = load_images_from_folder(val_images_path)

    coco_results = {
        'annotations': [],
        'images': [],
        'categories': [{'id': k, 'name': v} for k, v in category_map.items()]
    }

    for img_id, (img_path, img) in tqdm(enumerate(images), total=len(images)):
        results = yolov8.predict(img)

        # Append image info
        coco_results['images'].append({
            'id': img_id,
            'file_name': os.path.basename(img_path),
            'width': img.shape[1],
            'height': img.shape[0]
        })

        # Convert results to COCO format
        for result in results:
            predn = result.boxes
            annotations = pred_to_json(predn, img_path, category_map)
            coco_results['annotations'].extend(annotations)

    # Save results to a JSON file
    output_file = 'coco_results.json'
    with open(output_file, 'w') as f:
        json.dump(coco_results, f, indent=4)

    print(f'Results saved to {output_file}')
    ground_truth_file = '../data/client_test/annotations/instances_val2017.json'
    detection_file = 'coco_results.json'

    # Call the custom_evaluate function
    custom_evaluate(ground_truth_file, detection_file)


if __name__ == "__main__":
    config_path = '../data/client_test/data.yaml'
    model_config = {
        "source_weights_path": "../detectors/8sp2_150e_64b.engine",
        "device": "cuda"
    }
    main(config_path, model_config)
