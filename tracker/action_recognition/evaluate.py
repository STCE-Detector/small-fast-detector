import configparser
import os
import time
import json

import numpy as np
import torch
import pandas as pd

from tqdm import tqdm

from tracker.action_recognition.ar_confusion_matrix import ARConfusionMatrix
from ultralytics.utils.ops import clip_boxes


class AREvaluator:
    def __init__(self, config):
        self.config = config

        # Results flags
        self.save_results = config['action_recognition']['save_results']
        self.print_results = config['action_recognition']['print_results']

        # Create output directory
        output_sufix = time.strftime("%Y%m%d-%H%M%S") if config['name'] is None else config['name']
        self.output_dir = config['pred_dir'] + 'eval/' + output_sufix + '/'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Save config
        with open(self.output_dir + 'config.json', 'w') as f:
            json.dump(config, f, indent=4)

        # Read active behavior classes
        self.active_behaviors = config['action_recognition']['active_behaviors']

        # Initialize confusion matrices
        self.ar_confusion_matrix = ARConfusionMatrix(
            nc=len(self.active_behaviors),
            class_names=self.active_behaviors,
            conf=config['action_recognition']['confidence_threshold'],
            iou_thres=config['action_recognition']['iou_threshold'],
        )

        # Read all sequences
        self.data_dir = config['data_dir']
        self.sequences = [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))]

        # DataFrame column names
        self.behavior_columns = ['SS', 'SR', 'FA', 'G', 'OB']
        self.df_columns = ['frame', 'id', 'xl', 'yt', 'w', 'h', 'x/conf', 'y/class', 'z/vis'] + self.behavior_columns

        # Frame shape
        self.frame_shape = None

        # Smoothing and padding parameters
        self.smoothing_window = config['action_recognition']['smoothing_window']
        self.initial_pad = config['action_recognition']['initial_pad']
        self.final_pad = config['action_recognition']['final_pad']
        self.should_smooth_or_pad = self.smoothing_window > 0 or self.initial_pad > 0 or self.final_pad > 0

    def evaluate(self):
        # Evaluate all sequences
        for seq in self.sequences:
            video_path = os.path.join(self.data_dir, seq)
            self.evaluate_sequence(video_path)

        # Aggregate results and save
        return self.ar_confusion_matrix.save_results(self.output_dir, save=self.save_results, print_results=self.print_results)

    def load_dataframe(self, df_path, seq_name):
        """
        Load a dataframe from a txt file. If the file does not exist, it prints a message and returns None.
        args:
            df_path: str, path to the txt file
            seq_name: str, sequence name
        returns:
            df: pd.DataFrame, dataframe with the loaded data
        """
        if not os.path.exists(df_path):
            if self.print_results:
                print(f'No dataframe found for sequence {seq_name}, skipping')
            return None

        try:
            df = pd.read_csv(df_path, header=None)
            df.columns = self.df_columns[:len(df.columns)]
            self.behavior_columns = df.columns[9:]
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=self.df_columns)

        # Cast columns to correct types
        df[self.behavior_columns] = df[self.behavior_columns].astype(int)
        df[['frame', 'id', 'y/class']] = df[['frame', 'id', 'y/class']].astype(int)

        # Compute xr, yb columns
        df['xr'] = df['xl'] + df['w']
        df['yb'] = df['yt'] + df['h']

        # Clip boxes to frame shape
        df['yt'] = df['yt'].clip(lower=0, upper=self.frame_shape[0])
        df['yb'] = df['yb'].clip(lower=0, upper=self.frame_shape[0])
        df['xl'] = df['xl'].clip(lower=0, upper=self.frame_shape[1])
        df['xr'] = df['xr'].clip(lower=0, upper=self.frame_shape[1])

        # Gathering discrimination: if not discriminating groups, convert G column to 1 if any number is bigger than 0
        if not self.config['action_recognition']['discriminate_groups']:
            df['G'] = (df['G'] > 0).astype(int)

        return df

    def smooth_n_pad_predictions(self, pred_df, smoothing_window=0, initial_pad=0, final_pad=0):
        """
        Smooth predictions by forward filling and backward filling. Since limit_area is not specified, it also performs
        a forward fill and backward fill for the first and last frames of each track (padding).
        """
        pred_df = pred_df.sort_values(by=['id', 'frame']).reset_index(drop=True)

        def smooth_column(group, col):

            # If the column is entirely 0, return the group as is
            if group[col].sum() == 0:
                return group

            # Replace 0s with NaNs for forward and backward filling
            group[col] = group[col].replace(0, np.nan)

            # Smooth inner values
            if smoothing_window > 0:
                group[col] = group[col].ffill(limit=smoothing_window, limit_area='inside')

            # Initial padding
            if initial_pad > 0:
                group[col] = group[col].bfill(limit=initial_pad)

            # Final padding
            # TODO: is it necessary to pad the final values?
            if final_pad > 0:
                group[col] = group[col].ffill(limit=final_pad)

            # Replace NaNs with 0s
            group[col] = group[col].fillna(0)
            return group

        for flag in self.active_behaviors:
            pred_df = pred_df.groupby('id').apply(lambda group: smooth_column(group, flag)).reset_index(drop=True)

        return pred_df

    def preprocess_predictions(self, pred_df):
        # Filter by class
        pred_df = pred_df[pred_df['y/class'] == 0]

        # Smooth predictions
        if self.should_smooth_or_pad:
            pred_df = self.smooth_n_pad_predictions(
                pred_df,
                smoothing_window=self.smoothing_window,
                initial_pad=self.initial_pad,
                final_pad=self.final_pad
            )

        return pred_df

    def evaluate_sequence(self, video_path):
        seq_name = video_path.split('/')[-1]

        # Read frame shape
        config = configparser.ConfigParser()
        config.read(video_path + '/seqinfo.ini')
        frame_width = int(config['Sequence']['imWidth'])
        frame_height = int(config['Sequence']['imHeight'])
        self.frame_shape = (frame_height, frame_width)

        # Load ground truth
        gt_path = os.path.join(video_path, 'gt', 'manual_gt.txt')
        gt_df = self.load_dataframe(gt_path, seq_name)
        if gt_df is None:
            return

        # Load predictions
        pred_path = os.path.join(self.config["pred_dir"], 'data', f'{seq_name}.txt')
        pred_df = self.load_dataframe(pred_path, seq_name)
        if pred_df is None:
            return

        # Preprocess predictions
        pred_df = self.preprocess_predictions(pred_df)

        # Iterate over frames
        unique_frames = pred_df['frame'].unique()
        for frame_id in tqdm(unique_frames, desc=f'Evaluating {seq_name}', unit=' frames', disable=not self.print_results):
            # Filter by frame
            gt_frame = gt_df[gt_df['frame'] == frame_id]
            pred_frame = pred_df[pred_df['frame'] == frame_id]

            # Extract data
            gt_xyxy = torch.from_numpy(gt_frame[['xl', 'yt', 'xr', 'yb']].to_numpy())
            gt_behaviors = torch.from_numpy(gt_frame[self.active_behaviors].to_numpy())
            preds = torch.from_numpy(pred_frame[['xl', 'yt', 'xr', 'yb', 'x/conf'] + self.active_behaviors].to_numpy())

            self.ar_confusion_matrix.process_batch(preds, gt_xyxy, gt_behaviors)


if __name__ == '__main__':
    # Read config
    with open('cfg/eval.json') as f:
        config = json.load(f)

    # Initialize evaluator
    evaluator = AREvaluator(config)
    metrics_df = evaluator.evaluate()


