import os
import json
import yaml
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO
from tracker.jetson.model.model import Yolov8
from ultralytics.utils import ops
from ultralytics.utils.cocoeval import COCOeval
from ultralytics.utils.metrics import ConfusionMatrix

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def evaluate(coco_gt_file, coco_dt_file):
    """Evaluate predictions using COCO metrics."""
    coco_gt = COCO(coco_gt_file)
    coco_dt = coco_gt.loadRes(coco_dt_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def custom_evaluate(coco_gt_file, coco_dt_file):
    """Custom evaluation with different area ranges."""
    anno = COCO(coco_gt_file)
    pred = anno.loadRes(coco_dt_file)
    eval = COCOeval(anno, pred, 'bbox')

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
            img = np.array(cv2.imread(img_path))
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
        bboxes = boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
        scores = boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = boxes.cls.cpu().numpy()  # Class IDs

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

def initialize_model(model_config, labels):
    """Initialize the YOLO model."""
    device = torch.device(model_config['device'], 0)
    return Yolov8({
        'source_weights_path': model_config['model_path'],
        'device': device
    }, labels=labels)

def process_images(yolov8, images, category_map, gt_detections, confusion_matrix):
    """Process images and generate predictions."""
    df_detections_gt = pd.DataFrame(gt_detections['annotations'])
    coco_results = {
        'annotations': [],
        'images': [],
        'categories': [{'id': int(k), 'name': v} for k, v in category_map.items()]
    }

    for img_id, (img_path, img) in tqdm(enumerate(images), total=len(images)):
        results = yolov8.predict(img)

        coco_results['images'].append({
            'id': img_id,
            'file_name': os.path.basename(img_path),
            'width': int(img.shape[1]),
            'height': int(img.shape[0])
        })

        detections = results[0].boxes.data.to("cpu")
        image_id = int(os.path.basename(img_path).split(".")[0])
        img_gt_det = df_detections_gt[df_detections_gt['image_id'] == image_id]
        img_gt_cls = torch.from_numpy(img_gt_det['category_id'].values)
        img_bboxes_gt = torch.from_numpy(np.array(img_gt_det['bbox'].tolist()))
        img_bboxes_gt = torch.cat((img_bboxes_gt[:, :2], img_bboxes_gt[:, :2] + img_bboxes_gt[:, 2:]), dim=1)
        confusion_matrix.process_batch(detections, img_bboxes_gt, img_gt_cls)

        annotations = pred_to_json(results, img_path, category_map)
        coco_results['annotations'].extend(annotations)

    return coco_results

def save_results(coco_results, full_output_file, annotations_output_file):
    """Save results to JSON files."""
    with open(full_output_file, 'w') as f:
        json.dump(coco_results, f, indent=4, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    with open(annotations_output_file, 'w') as f:
        json.dump(coco_results['annotations'], f, indent=4, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    print(f'Results saved to {full_output_file} and {annotations_output_file}')

def wrapper_run(config):
    """Main function to generate predictions and save them in COCO format."""
    config_path = load_yaml(config["input_data_dir"])

    path = config["input_data_dir"]
    val_images_path = f"{path[:path.rfind('/')]}/{config_path['val']}"
    category_map = {int(k): v for k, v in config_path['names'].items()}
    ground_truth_file = f"{path[:path.rfind('/')]}/{'annotations/instances_val2017.json'}"
    with open(ground_truth_file, 'r') as f:
        gt_detections = json.load(f)

    yolov8 = initialize_model(config, config_path['names'])
    confusion_matrix = ConfusionMatrix(nc=6, conf=0.3, iou_thres=0.3)
    images = load_images_from_folder(val_images_path)

    coco_results = process_images(yolov8, images, category_map, gt_detections, confusion_matrix)

    for normalize in [False, 'gt', 'pred']:
        confusion_matrix.plot(normalize=normalize, save_dir="./")

    save_results(coco_results, 'full_coco_results.json', 'coco_results.json')
    custom_evaluate(ground_truth_file, 'coco_results.json')