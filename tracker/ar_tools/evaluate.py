import os
import time
import json
import torch
import pandas as pd

from tqdm import tqdm

from tracker.ar_tools.ar_confusion_matrix import ARConfusionMatrix


class AREvaluator:
    def __init__(self, config):
        self.config = config

        # Create output directory
        time_str = time.strftime("%Y%m%d-%H%M%S")
        self.output_dir = config['output_dir'] + '/eval/' + config['name'] + '/' + time_str
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Initialize confusion matrices
        self.ar_confusion_matrix = ARConfusionMatrix(
            nc=4,
            conf=config['action_recognition']['confidence_threshold'],
            iou_thres=config['action_recognition']['iou_threshold']
        )

        # Read all sequences
        self.data_dir = config['data_dir']
        self.sequences = [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))]

        # DataFrame column names
        self.behavior_columns = ['SS', 'SR', 'FA', 'G']
        self.df_columns = ['frame', 'id', 'xl', 'yt', 'w', 'h', 'x/conf', 'y/class', 'z/vis'] + self.behavior_columns

    def evaluate(self):
        # Evaluate all sequences
        for seq in self.sequences:
            video_path = os.path.join(self.data_dir, seq)
            self.evaluate_sequence(video_path)

        # Aggregate results and save
        self.ar_confusion_matrix.save_results(self.output_dir)

    def load_dataframe(self, df_path, seq_name):
        if not os.path.exists(df_path):
            print(f'No dataframe found for sequence {seq_name}, skipping')
            return None

        df = pd.read_csv(df_path, header=None)
        df.columns = self.df_columns

        # Cast columns to correct types
        df[self.behavior_columns] = df[self.behavior_columns].astype(int)
        df[['frame', 'id', 'y/class']] = df[['frame', 'id', 'y/class']].astype(int)

        # Compute xr, yb columns
        df['xr'] = df['xl'] + df['w']
        df['yb'] = df['yt'] + df['h']
        return df

    def smooth_predictions(self, pred_df, window=1):
        """
        Smooth predictions by forward filling and backward filling. Since limit_area is not specified, it also performs
        a forward fill and backward fill for the first and last frames of each track (padding).
        """
        pred_df = pred_df.sort_values(by=['id', 'frame']).reset_index(drop=True)

        def smooth_column(group, col):
            # Forward fill, backward fill
            group[col] = group[col].replace(0, pd.NA)

            # Smooth
            # TODO: for SS column, bfill could have bigger limit to avoid initial gap
            group[col] = group[col].ffill(limit=window).bfill(limit=window * 2 if col == 'SS' else window)
            group[col] = group[col].fillna(0)
            return group

        for flag in self.behavior_columns:
            pred_df = pred_df.groupby('id').apply(lambda group: smooth_column(group, flag)).reset_index(drop=True)

        return pred_df

    def preprocess_predictions(self, pred_df, smoothing_window=60):
        # Filter by class
        pred_df = pred_df[pred_df['y/class'] == 0]  # TODO: currently only pedestrian class is supported

        # Smooth predictions
        if smoothing_window > 0:
            pred_df = self.smooth_predictions(pred_df, window=smoothing_window)

        return pred_df

    def evaluate_sequence(self, video_path):
        seq_name = video_path.split('/')[-1]

        # Load ground truth
        gt_path = os.path.join(video_path, 'gt', 'manual_gt.txt')
        gt_df = self.load_dataframe(gt_path, seq_name)
        if gt_df is None:
            return

        # Load predictions
        pred_path = os.path.join(self.config["pred_dir"], 'data', f'{seq_name}.txt')
        pred_df = self.load_dataframe(pred_path, seq_name)

        # Preprocess predictions
        pred_df = self.preprocess_predictions(pred_df, smoothing_window=self.config['action_recognition']['smoothing_window'])

        # Iterate over frames
        unique_frames = pred_df['frame'].unique()
        for frame_id in tqdm(unique_frames, desc=f'Evaluating {seq_name}', unit=' frames'):
            # Filter by frame
            gt_frame = gt_df[gt_df['frame'] == frame_id]
            pred_frame = pred_df[pred_df['frame'] == frame_id]

            # Gathering discrimination
            if not self.config['action_recognition']['discriminate_groups']:
                gt_frame['G'] = gt_frame['G'].astype(bool).astype(int)
                pred_frame['G'] = pred_frame['G'].astype(bool).astype(int)

            # Extract data
            gt_xyxy = torch.from_numpy(gt_frame[['xl', 'yt', 'xr', 'yb']].to_numpy())
            gt_behaviors = torch.from_numpy(gt_frame[self.behavior_columns].to_numpy())
            preds = torch.from_numpy(pred_frame[['xl', 'yt', 'xr', 'yb', 'x/conf'] + self.behavior_columns].to_numpy())

            self.ar_confusion_matrix.process_batch(preds, gt_xyxy, gt_behaviors)


if __name__ == '__main__':
    # Read config
    with open('cfg/eval.json') as f:
        config = json.load(f)

    # Initialize evaluator
    evaluator = AREvaluator(config)
    evaluator.evaluate()


