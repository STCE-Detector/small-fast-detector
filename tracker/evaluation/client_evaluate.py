import os
import time
import json
import torch
import numpy as np

from tqdm import tqdm

from ultralytics.utils.metrics import ConfusionMatrix


def eval_sequence(video_root, detection_cm):
    # Read gt
    gt_path = video_root + '/gt/auto_gt.txt'
    gt = np.loadtxt(gt_path, delimiter=',')

    # Read predictions
    pred_path = video_root + '/gt/auto_gt.txt'
    pred = np.loadtxt(pred_path, delimiter=',')

    # TODO: filter by class in predictions

    # Iterate over frames
    sequence_name = video_root.split('/')[-1]
    unique_frames = list(set(gt[:, 0]))
    for frame_id in tqdm(unique_frames, desc=f'Evaluating {sequence_name}', unit=' frames'):
        # Get GT and predictions for this frame
        gt_frame = gt[gt[:, 0] == frame_id]
        pred_frame = pred[pred[:, 0] == frame_id]

        # DETECTION EVALUATION
        # Preprocess GT
        gt_tlwh = gt_frame[:, 2:6]
        gt_cls = torch.from_numpy(np.zeros(gt_frame.shape[0]))
        gt_xyxy = gt_tlwh.copy()
        gt_xyxy[:, 2:] += gt_xyxy[:, :2]
        gt_xyxy = torch.from_numpy(gt_xyxy)

        # Preprocess predictions
        pred_tlwh = pred_frame[:, 2:6]
        pred_xyxy = pred_tlwh.copy()
        pred_xyxy[:, 2:] += pred_xyxy[:, :2]
        pred_detections = np.zeros((pred_frame.shape[0], 6))
        pred_detections[:, :4] = pred_xyxy
        # TODO: maybe read scores from file if available
        pred_detections[:, 4] = np.ones(pred_frame.shape[0])    # Set score to 1
        pred_detections[:, 5] = np.zeros(pred_frame.shape[0])     # Set class to 1
        pred_detections = torch.from_numpy(pred_detections)

        detection_cm.process_batch(pred_detections, gt_xyxy, gt_cls)


if __name__ == '__main__':
    # Read config
    with open('cfg/eval.json') as f:
        config = json.load(f)

    # Create output directory
    time_str = time.strftime("%Y%m%d-%H%M%S")
    output_dir = config['output_dir'] + '/client_eval/' + config['name'] + '/' + time_str
    config['output_dir'] = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the Confusion Matrix for tracking evaluation
    confusion_matrix = ConfusionMatrix(
        nc=1,
        conf=config['tracking']['confidence_threshold'],
        iou_thres=config['tracking']['iou_threshold']
    )

    # Read all sequences
    #video_root = config['data_dir'] + 'MOT17-09'
    data_dir = config['data_dir']
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    # Iterate over sequences
    for folder in folders[2:3]:
        video_root = data_dir + folder
        eval_sequence(video_root, confusion_matrix)

    # Aggregate confusion matrix
    for normalize in [False, 'gt', 'pred']:
        confusion_matrix.plot(normalize=normalize, save_dir=config['output_dir'])
