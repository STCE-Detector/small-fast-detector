import os
import pandas as pd
from tqdm import tqdm

from tracker.ar_tools.auto_labelling import get_intervals


def update_sequential_df(sequential_df, intervals_df):
    behavior_columns = ['SS', 'SR', 'FA', 'G']

    # Avoid modifying the original dataframes
    expanded_intervals_df = intervals_df.copy()
    manual_gt = sequential_df.copy()

    def generate_range(row):
        return list(range(row['start_frame'], row['end_frame'] + 1))
    expanded_intervals_df['frame'] = expanded_intervals_df.apply(generate_range, axis=1)

    expanded_intervals_df = expanded_intervals_df.explode('frame')
    expanded_intervals_df.drop(columns=['start_frame', 'end_frame'], inplace=True)
    expanded_intervals_df = expanded_intervals_df[['frame', 'id'] + behavior_columns]

    # Delete identical rows
    expanded_intervals_df = expanded_intervals_df.drop_duplicates()
    # Select the first fully duplicated row
    assert expanded_intervals_df[expanded_intervals_df.duplicated(keep=False)].empty == True, f'Identical rows found'

    # Check if there are rows with the more than one flag for the same frame and id
    problematic_rows = expanded_intervals_df[(expanded_intervals_df['SS'] + expanded_intervals_df['SR'] + expanded_intervals_df['FA']) > 1]
    assert problematic_rows.empty, f'More than one flag for per single interval'

    # Merge flags for the same frame and id combination
    expanded_intervals_df = expanded_intervals_df.groupby(['frame', 'id']).max().reset_index()

    # Check if there are rows with an incorrect flag value
    problematic_rows = expanded_intervals_df[(expanded_intervals_df['SS'] > 1) | (expanded_intervals_df['SR'] > 1) | (expanded_intervals_df['FA'] > 1)]
    assert problematic_rows.empty, f'Incorrect flag value'

    # Merge the expanded intervals with the sequential dataframe
    if sequential_df.columns.isin(behavior_columns).sum() == 4:
        manual_gt.update(expanded_intervals_df, overwrite=True)
    else:
        manual_gt = pd.merge(manual_gt, expanded_intervals_df, how='left', on=['frame', 'id'])

    manual_gt.fillna(0, inplace=True)
    manual_gt[behavior_columns] = manual_gt[behavior_columns].astype(int)
    manual_gt.drop_duplicates(inplace=True)

    # Check if last frame of every id in the manual_gt is the same as the last frame of the sequential_df
    assert manual_gt.groupby('id')['frame'].max().equals(sequential_df.groupby('id')['frame'].max()), f'Last frame of every id in the manual_gt is not the same as the last frame of the sequential_df'

    # Check if first frame of every id in the manual_gt is the same as the last frame of the sequential_df
    assert manual_gt.groupby('id')['frame'].min().equals(sequential_df.groupby('id')['frame'].min()), f'First frame of every id in the manual_gt is not the same as the first frame of the sequential_df'

    # Check if the manual_gt has the same number of frames as the sequential_df
    assert manual_gt['frame'].nunique() == sequential_df['frame'].nunique(), f'Number of frames in the manual_gt is not the same as the sequential_df'

    # Check if the manual_gt has the same number of ids as the sequential_df
    assert manual_gt['id'].nunique() == sequential_df['id'].nunique(), f'Number of ids in the manual_gt is not the same as the sequential_df'

    # Check if the manual_gt has the same number of rows as the sequential_df
    assert manual_gt.shape[0] == sequential_df.shape[0], f'Number of rows in the manual_gt is not the same as the sequential_df'

    # Extract intervals with the same behavior into a dataframe
    # TODO: error of 1 frame for some end_frame values
    """new_intervals_df = pd.DataFrame()
    for col in behavior_columns:
        behaviour_df = get_intervals(manual_gt, col)
        new_intervals_df = pd.concat([new_intervals_df, behaviour_df])

    new_intervals_df.fillna(0, inplace=True)
    new_intervals_df.drop_duplicates(inplace=True)
    new_intervals_df = new_intervals_df.astype(int)
    new_intervals_df = new_intervals_df[['id', 'start_frame', 'end_frame'] + behavior_columns]"""
    return manual_gt


def add_actions(sequence_path):
    gt_columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'x/conf', 'y/class', 'z/vis']
    gt_df = pd.read_csv(os.path.join(sequence_path, 'gt', 'gt.txt'), names=gt_columns, usecols=[i for i in range(len(gt_columns))])

    # Check if the actions file exists
    actions_path = os.path.join(sequence_path, 'gt', 'actions.csv')
    if not os.path.exists(actions_path):
        print(f'\nActions file not found for {sequence_path}')
        return

    intervals_columns = ['id', 'start_frame', 'end_frame', 'SS', 'SR', 'FA', 'G']
    intervals_df = pd.read_csv(actions_path, names=intervals_columns, usecols=[i for i in range(len(intervals_columns))], header=0)
    # Correct visualization tool's 0-based indexing
    intervals_df['start_frame'] = intervals_df['start_frame'] + 1
    intervals_df['end_frame'] = intervals_df['end_frame'] + 1
    # Drop duplicates
    intervals_df.drop_duplicates(inplace=True)

    # Update the gt with the actions
    manual_gt = update_sequential_df(gt_df, intervals_df)

    manual_gt.to_csv(os.path.join(sequence_path, 'gt', 'manual_gt.txt'), header=False, index=False, sep=',')


if __name__ == "__main__":

    dataset_root = './../evaluation/TrackEval/data/gt/mot_challenge/MOTHupba-train'

    # Get all sequences
    sequences = [sequence for sequence in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, sequence))]
    # sequences = ['dancetrack0079']

    # Iterate over all sequences
    for sequence in tqdm(sequences, desc='Evaluating sequences', unit=' sequences'):
        sequence_path = os.path.join(dataset_root, sequence)
        add_actions(sequence_path)

