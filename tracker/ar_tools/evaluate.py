import os
import time
import json
import torch
import numpy as np

from tracker.ar_tools.confusion_matrix import ConfusionMatrix, ARConfusionMatrix


def eval_sequence(video_root, config):
    # Read gt
    gt_path = video_root + '/gt/auto_gt.txt'
    gt = np.loadtxt(gt_path, delimiter=',')

    # Read predictions
    pred_path = video_root + '/gt/auto_gt_mod.txt'
    pred = np.loadtxt(pred_path, delimiter=',')

    # Initialize Confusion Matrix for detection and behavior evaluation
    tracking_eval_flag = config['tracking']['enable']
    ar_eval_flag = config['action_recognition']['enable']

    if tracking_eval_flag:
        confusion_matrix = ConfusionMatrix(
            nc=1,
            conf=config['tracking']['confidence_threshold'],
            iou_thres=config['tracking']['iou_threshold']
        )
    if ar_eval_flag:
        ar_confusion_matrix = ARConfusionMatrix(
            nc=4,
            conf=config['action_recognition']['confidence_threshold'],
            iou_thres=config['action_recognition']['iou_threshold']
        )

    # Iterate over frames
    for frame_id in list(set(gt[:, 0])):
        # Get GT and predictions for this frame
        gt_frame = gt[gt[:, 0] == frame_id]
        pred_frame = pred[pred[:, 0] == frame_id]

        ##############################
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

        # Compute confusion matrix
        if tracking_eval_flag:
            confusion_matrix.process_batch(pred_detections, gt_xyxy, gt_cls)
        ##############################

        ##############################
        # BEHAVIOR EVALUATION
        # Preprocess GT
        gt_behaviors = gt_frame[:, 6:]
        gt_behaviors = torch.from_numpy(gt_behaviors)

        # Preprocess predictions
        pred_behaviors = pred_frame[:, 6:]
        pred_behaviors = torch.from_numpy(pred_behaviors)

        # TODO: currently Gathering does not distinguish between different groups, now is just boolean
        gt_behaviors[:, -1] = (gt_behaviors[:, -1] > 0).long()
        pred_behaviors[:, -1] = (pred_behaviors[:, -1] > 0).long()

        # Merge the first 5 columns of pred_detections with pred_behaviors
        pred_behaviors = torch.cat((pred_detections[:, :5], pred_behaviors), dim=1)

        # Compute confusion matrix
        if ar_eval_flag:
            ar_confusion_matrix.process_batch(pred_behaviors, gt_xyxy, gt_behaviors)

    # Aggregate confusion matrices
    if tracking_eval_flag:
        for normalize in [False, 'gt', 'pred']:
            confusion_matrix.plot(normalize=normalize, save_dir=config['output_dir'])

    if ar_eval_flag:
        ar_confusion_matrix.save_results(config['output_dir'])

    return None


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

    # Read all sequences
    video_root = config['data_dir'] + 'MOT17-09'

    # Iterate over sequences

    # Evaluate sequence
    eval_sequence(video_root, config)
    # Save results

