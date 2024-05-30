import os
import time
import json
import torch
import numpy as np

from tqdm import tqdm

from tracker.ar_tools.ar_confusion_matrix import ARConfusionMatrix


def eval_sequence(video_root, ar_cm, config):
    # Read gt
    gt_path = video_root + '/gt/auto_gt.txt'
    gt = np.loadtxt(gt_path, delimiter=',')

    # Read predictions
    pred_path = video_root + '/gt/auto_gt.txt'
    pred = np.loadtxt(pred_path, delimiter=',')

    # Filter predictions by classes
    # TODO: currently only pedestrians is supported
    pred = pred[pred[:, 7] == 1]

    # TEST ONLY
    #pred[11, -1] = 3

    # Iterate over frames
    sequence_name = video_root.split('/')[-1]
    unique_frames = list(set(gt[:, 0]))
    for frame_id in tqdm(unique_frames, desc=f'Evaluating {sequence_name}', unit=' frames'):
        # Get GT and predictions for this frame
        gt_frame = gt[gt[:, 0] == frame_id]
        pred_frame = pred[pred[:, 0] == frame_id]

        ##############################
        # DETECTION EVALUATION
        # Preprocess GT
        gt_tlwh = gt_frame[:, 2:6]
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
        ##############################

        ##############################
        # BEHAVIOR EVALUATION
        # Preprocess GT
        gt_behaviors = gt_frame[:, -4:]
        gt_behaviors = torch.from_numpy(gt_behaviors)

        # Preprocess predictions
        pred_behaviors = pred_frame[:, -4:]
        pred_behaviors = torch.from_numpy(pred_behaviors)

        # Gathering does now distinguish between different groups, uncomment if needed back to bool
        if not config['action_recognition']['discriminate_groups']:
            gt_behaviors[:, -1] = (gt_behaviors[:, -1] > 0).bool()
            pred_behaviors[:, -1] = (pred_behaviors[:, -1] > 0).bool()

        # Merge the first 5 columns of pred_detections with pred_behaviors
        pred_behaviors = torch.cat((pred_detections[:, :5], pred_behaviors), dim=1)

        # Compute confusion matrix
        if ar_cm is not None:
            ar_cm.process_batch(pred_behaviors, gt_xyxy, gt_behaviors)


if __name__ == '__main__':
    # Read config
    with open('cfg/eval.json') as f:
        config = json.load(f)

    # Create output directory
    time_str = time.strftime("%Y%m%d-%H%M%S")
    output_dir = config['output_dir'] + '/eval/' + config['name'] + '/' + time_str
    config['output_dir'] = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize Confusion Matrix for detection and behavior evaluation
    ar_confusion_matrix = ARConfusionMatrix(
        nc=4,
        conf=config['action_recognition']['confidence_threshold'],
        iou_thres=config['action_recognition']['iou_threshold']
    )

    # Read all sequences
    data_dir = config['data_dir']
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    # Iterate over sequences
    for folder in folders:
        video_root = data_dir + folder
        eval_sequence(video_root, ar_confusion_matrix, config)

    # Aggregate confusion matrices
    ar_confusion_matrix.save_results(config['output_dir'])

