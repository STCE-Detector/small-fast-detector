import configparser
import os
import time
import json
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.utils.ops import clip_boxes


def load_dataframe(df_path, seq_name, frame_shape):
    column_names = ['frame', 'id', 'xl', 'yt', 'w', 'h', 'x/conf', 'y/class', 'z/vis']

    if not os.path.exists(df_path):
        print(f'No dataframe found for sequence {seq_name}, skipping')
        return None
    try:
        df = pd.read_csv(df_path, header=None)
        df.columns = column_names[:len(df.columns)]
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=column_names)

    df[['frame', 'id', 'y/class']] = df[['frame', 'id', 'y/class']].astype(int)

    df['xr'] = df['xl'] + df['w']
    df['yb'] = df['yt'] + df['h']

    df['yt'] = df['yt'].clip(lower=0, upper=frame_shape[0])
    df['yb'] = df['yb'].clip(lower=0, upper=frame_shape[0])
    df['xl'] = df['xl'].clip(lower=0, upper=frame_shape[1])
    df['xr'] = df['xr'].clip(lower=0, upper=frame_shape[1])

    return df


def eval_sequence(video_root, config, detection_cm):
    sequence_name = video_root.split('/')[-1]

    # Read frame shape
    seq_config = configparser.ConfigParser()
    seq_config.read(video_root + '/seqinfo.ini')
    frame_width = int(seq_config['Sequence']['imWidth'])
    frame_height = int(seq_config['Sequence']['imHeight'])
    frame_shape = (frame_height, frame_width)

    # Read gt
    gt_path = video_root + '/gt/gt.txt'
    gt_df = load_dataframe(gt_path, sequence_name, frame_shape)
    if gt_df is None:
        return
    # TODO: Hardcoded for now
    gt_df = gt_df[gt_df['y/class'] == 1]
    gt_df['y/class'] = gt_df['y/class'].map({1: 0})

    # Read predictions
    pred_path = config['pred_dir'] + 'data/' + sequence_name + '.txt'
    pred_df = load_dataframe(pred_path, sequence_name, frame_shape)
    if pred_df is None:
        return

    # Filter predictions by classes
    classes = config['tracking']['classes']
    # TODO: Hardcoded for now
    pred_df = pred_df[pred_df['y/class'] == 0]

    # Iterate over frames
    unique_frames = pred_df['frame'].unique()
    for frame_id in tqdm(unique_frames, desc=f'Evaluating {sequence_name}', unit=' frames'):
        # Filter by frame
        gt_frame = gt_df[gt_df['frame'] == frame_id]
        pred_frame = pred_df[pred_df['frame'] == frame_id]

        # Extract data
        gt_xyxy = torch.from_numpy(gt_frame[['xl', 'yt', 'xr', 'yb']].to_numpy())
        gt_cls = torch.from_numpy(gt_frame['y/class'].to_numpy())
        preds = torch.from_numpy(pred_frame[['xl', 'yt', 'xr', 'yb', 'x/conf', 'y/class']].to_numpy())

        detection_cm.process_batch(preds, gt_xyxy, gt_cls)


if __name__ == '__main__':
    # Read config
    with open('cfg/eval.json') as f:
        config = json.load(f)

    # Initialize the Confusion Matrix for tracking evaluation
    confusion_matrix = ConfusionMatrix(
        nc=1,
        conf=config['tracking']['confidence_threshold'],
        iou_thres=config['tracking']['iou_threshold']
    )

    # Read all sequences
    data_dir = config['data_dir']
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    # Iterate over sequences
    for folder in folders:
        video_root = data_dir + folder
        eval_sequence(video_root, config, confusion_matrix)

    # Aggregate confusion matrix
    for normalize in [False, 'gt', 'pred']:
        confusion_matrix.plot(
            normalize=normalize,
            save_dir=config['pred_dir'],
            names=[i for i in range(len(config['tracking']['classes']))]
        )
