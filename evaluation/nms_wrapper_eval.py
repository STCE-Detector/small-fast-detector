import os
import json
import time

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

    # Extract COCOEval results and save them
    #TODO: use function and pass save_dir as argument
    extract_cocoeval_metrics(eval, save_dir)

    # Extract Confusion Matrix results and save them
    stats = {}
    # Precision
    stats['metrics/AP(T)'] = eval.stats[1]
    stats['metrics/AP(S)'] = eval.stats[2]
    stats['metrics/AP(M)'] = eval.stats[3]
    stats['metrics/AP(L)'] = eval.stats[4]
    # Recall by IoU
    stats['metrics/AR(50:95)'] = eval.stats[25]
    stats['metrics/AR(75)'] = eval.stats[30]
    stats['metrics/AR(50)'] = eval.stats[31]
    # Recall by Area
    stats['metrics/AR(T)'] = eval.stats[26]
    stats['metrics/AR(S)'] = eval.stats[27]
    stats['metrics/AR(M)'] = eval.stats[28]
    stats['metrics/AR(L)'] = eval.stats[29]
    # Save results to file
    # TODO: save_dir is inferred from coco_dt_file path
    results_file = save_dir / "evaluation_results.json"
    with results_file.open("w") as file:
        json.dump(stats, file)


def extract_cocoeval_metrics(eval, save_dir):
    """Extracts metrics from COCOeval object and saves them to a DataFrame."""

    # Function to append metrics
    def append_metrics(metrics, metric_type, iou, area, max_dets, value):
        metrics.append({
            'Metric Type': metric_type,
            'IoU': iou,
            'Area': area,
            'Max Detections': max_dets,
            'Value': value
        })

    # Initialize a list to store the metrics
    metrics_ = []

    # Extract metrics for bbox/segm evaluation
    iou_types = ['0.50:0.95', '0.50', '0.75']
    areas = eval.params.areaRngLbl
    max_dets = eval.params.maxDets

    # Extract AP metrics (indices 0-14: 3 IoUs * 5 areas)
    for i, iou in enumerate(iou_types):
        for j, area in enumerate(areas):
            idx = i * len(areas) + j
            append_metrics(metrics_, 'AP', iou, area, max_dets[-1], eval.stats[idx])

    # Extract AR metrics (indices 15-17: 3 maxDets for 'all' area)
    num_ap_metrics = len(iou_types) * len(areas)  # Total number of AP metrics

    # Iterate over max_dets to append AR metrics
    for i, md in enumerate(max_dets):
        for j, area in enumerate(areas):
            idx = num_ap_metrics + j + i * len(areas)  # Adjust index calculation for AR
            append_metrics(metrics_, 'AR', '0.50:0.95', area, md, eval.stats[idx])

    # Append AR metrics for 0.75 and 0.50 IoU
    for i, iou in enumerate(['0.75', '0.50']):
        append_metrics(metrics_, 'AR', iou, 'all', '300', eval.stats[idx + i + 1])

    # Convert to DataFrame
    df_metrics = pd.DataFrame(metrics_)

    # Save to file
    df_metrics.to_csv(save_dir / "cocoeval_results.csv", index=False)


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
        'model_path': model_config['model_path'],
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

def yolo_wrapper_validation(config):
    """Main function to generate predictions and save them in COCO format."""
    config_path = load_yaml(config["input_data_dir"])
    name = time.strftime("%Y%m%d-%H%M%S") if config["name"] is None else config["name"]
    output_dir = config['output_dir'] + "/" + name
    os.makedirs(output_dir, exist_ok=True)
    path = config["input_data_dir"]
    val_images_path = f"{path[:path.rfind('/')]}/{config_path['val']}"
    category_map = {int(k): v for k, v in config_path['names'].items()}
    ground_truth_file = f"{path[:path.rfind('/')]}/{'annotations/instances_val2017.json'}"
    with open(ground_truth_file, 'r') as f:
        gt_detections = json.load(f)

    yolov8 = initialize_model(config, config_path['names'])
    confusion_matrix = ConfusionMatrix(nc=6, conf=0.3, iou_thres=0.3)
    images = load_images_from_folder(val_images_path)

    results = process_images(yolov8, images, category_map, gt_detections, confusion_matrix)

    for normalize in [False, 'gt', 'pred']:
        confusion_matrix.plot(normalize=normalize, save_dir=output_dir)

    save_results(results, output_dir + '/full_coco_results.json', output_dir + '/coco_results.json')
    custom_evaluate(ground_truth_file, output_dir + '/coco_results.json')