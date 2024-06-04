import os
import pandas as pd
from tqdm import tqdm


def add_actions(sequence_path):

    gt_columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'x/conf', 'y/class', 'z/vis']
    gt_df = pd.read_csv(os.path.join(sequence_path, 'gt', 'gt.txt'), names=gt_columns, usecols=[i for i in range(len(gt_columns))])

    intervals_columns = ['id', 'start', 'end', 'SS', 'SR', 'FA', 'G']
    intervals_df = pd.read_csv(os.path.join(sequence_path, 'gt', 'actions.csv'), names=intervals_columns, usecols=[i for i in range(len(intervals_columns))], header=0)
    # Correct visualization tool's 0-based indexing
    intervals_df['start'] = intervals_df['start'] + 1
    intervals_df['end'] = intervals_df['end'] + 1

    expanded_intervals_df = intervals_df.copy()
    expanded_intervals_df['frame'] = [list(range(start, end + 1)) for start, end in zip(expanded_intervals_df['start'], expanded_intervals_df['end'])]
    expanded_intervals_df = expanded_intervals_df.explode('frame')
    expanded_intervals_df.drop(columns=['start', 'end'], inplace=True)
    expanded_intervals_df = expanded_intervals_df[['frame', 'id', 'SS', 'SR', 'FA', 'G']]
    # expanded_intervals_df[(expanded_intervals_df['id']==4) & (expanded_intervals_df['SS']==1)]['frame'].max()

    manual_gt = pd.merge(gt_df, expanded_intervals_df, how='left', on=['frame', 'id'])
    manual_gt.fillna(0, inplace=True)
    manual_gt['SS'] = manual_gt['SS'].astype(int)
    manual_gt['SR'] = manual_gt['SR'].astype(int)
    manual_gt['FA'] = manual_gt['FA'].astype(int)
    manual_gt['G'] = manual_gt['G'].astype(int)
    #manual_gt[(manual_gt['id']==86) & (manual_gt['SS']==1)]['frame'].min()

    manual_gt.to_csv(os.path.join(sequence_path, 'gt', 'manual_gt.txt'), header=False, index=False, sep=',')


if __name__ == "__main__":

    dataset_root = './../evaluation/TrackEval/data/gt/mot_challenge/MOTHupba-train'

    #sequence = 'MOT17-04'
    #sequence_path = os.path.join(dataset_root, sequence)
    #add_actions(sequence_path)

    # Iterate over all sequences
    for sequence in tqdm(os.listdir(dataset_root), desc='Evaluating sequences', unit=' sequences'):
        sequence_path = os.path.join(dataset_root, sequence)
        add_actions(sequence_path)

