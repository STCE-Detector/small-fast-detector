import json
import numpy as np
import pandas as pd
import configparser
from itertools import combinations
import networkx as nx


def seq_info(info_path):
    """
    Extracts sequence information from the info file
    :param info_path: path to the info file
    :return: frame_width, frame_height, video_fps
    """
    config = configparser.ConfigParser()
    config.read(info_path)
    frame_width = int(config['Sequence']['imWidth'])
    frame_height = int(config['Sequence']['imHeight'])
    video_fps = int(config['Sequence']['frameRate'])
    return frame_width, frame_height, video_fps


def minimal_distance_to_bbox(P, bbox):
    """
    Compute the minimal distance between a point P and a bounding box
    :param P: sensor point coordinates
    :param bbox: bounding box coordinates in the form (x_center, y_center, w, h)
    """
    x_center, y_center, w, h = bbox

    x_min = x_center - w / 2
    x_max = x_center + w / 2
    y_min = y_center - h / 2
    y_max = y_center + h / 2

    # Closest point on the bounding box to P
    closest_x = max(x_min, min(P[0], x_max))
    closest_y = max(y_min, min(P[1], y_max))

    # Compute the Euclidean distance between P and the closest point
    distance = np.sqrt((P[0] - closest_x) ** 2 + (P[1] - closest_y) ** 2)
    return distance


def recognize_gather(df, frame_id, area_threshold, distance_threshold, min_people):
    """
    Recognize gatherings in a frame
    :param df: dataframe with the bounding boxes
    :param frame_id: frame to analyze
    :param area_threshold: threshold for the area similarity
    :param distance_threshold: threshold for the distance
    :param min_people: minimum number of people in a gathering
    :return: dataframe with the gatherings labeled
    """
    frame_df = df[df['frame'] == frame_id].copy()
    frame_df.reset_index(inplace=True, drop=True)

    pairs = []
    for i, j in combinations(range(len(frame_df)), 2):
        # Check similarity of areas
        # TODO: order numerator and denominator
        if area_threshold <= (frame_df.loc[i, 'area'] / frame_df.loc[j, 'area']) <= (1 / area_threshold):
            # Compute euclidean distance
            d = np.sqrt(np.sum((frame_df.loc[i, ['xc', 'yc']] - frame_df.loc[j, ['xc', 'yc']]) ** 2))
            # Get mean area
            a = (frame_df.loc[i, 'area'] + frame_df.loc[j, 'area'])
            # Normalize distance
            norm_d = d / np.sqrt(a)
            # Check distance ¿speed?
            if norm_d <= distance_threshold:
                pairs.append([i, j])

    # Get independent chains
    g = nx.Graph()
    g.add_edges_from(pairs)
    # Find connected components
    independent_chains = list(nx.connected_components(g))
    # Filter out chains having less than 3 elements
    valid_chains = [chain for chain in independent_chains if len(chain) > (min_people - 1)]

    # Assing group tags to the corresponding bboxes
    for i, chain in enumerate(valid_chains):
        frame_df.loc[list(chain), 'G'] = i + 1

    # TODO: review this merge
    df = pd.merge(df, frame_df[['frame', 'id', 'G']], on=['frame', 'id'], how='left', suffixes=('', '_dup'))
    if 'G_dup' in df.columns:
        df['G'] = df['G'].combine_first(df['G_dup'])
        df.drop(columns=['G_dup'], inplace=True)
    return df


def get_motion_descriptors(df, id, w, dt, l):
    """
    Get motion descriptors for a given id
    :param df: dataframe with the bounding boxes
    :param id: id to analyze
    :param w: weight for the projection of the speed
    :param dt: window frames
    :param l: list's length
    :return: dataframe with the motion descriptors
    """
    # Select ID, should be paralelized using group by
    id_df = df[df['id'] == id].copy()

    # Compute projected instant diferentials
    id_df.loc[:, 'dx'] = (id_df['xc'] - id_df['xc'].shift(1)) * w[0]
    id_df.loc[:, 'dy'] = (id_df['yc'] - id_df['yc'].shift(1)) * w[1]

    # Compute area-normalized projected instant speed
    id_df.loc[:, 'dv'] = np.sqrt(id_df['dx'] ** 2 + id_df['dy'] ** 2) / (id_df['area'])

    # Compute interval projected differences
    id_df.loc[:, 'Dx'] = (id_df['xc'] - id_df['xc'].shift(dt)) * w[0]
    id_df.loc[:, 'Dy'] = (id_df['yc'] - id_df['yc'].shift(dt)) * w[1]

    # Compute area-normalized projected speed
    id_df.loc[:, 'V'] = np.sqrt(id_df['Dx'] ** 2 + id_df['Dy'] ** 2) / (dt + id_df['area'])

    # Average the speed according to last L*dt observations
    id_df.loc[:, 'aV'] = id_df['V'].rolling(window=dt * l, min_periods=1).mean()

    # Compute average and instant direction of movement # TODO: which one should we use?
    id_df.loc[:, 'idir'] = np.sign(id_df['dy'])
    id_df.loc[:, 'aidir'] = id_df['idir'].rolling(window=dt * l, min_periods=1).mean()
    id_df.loc[:, 'dir'] = np.sign(id_df['Dy'])
    id_df.loc[:, 'adir'] = id_df['dir'].rolling(window=dt * l, min_periods=1).mean()

    new_df = pd.merge(df, id_df[['frame', 'id', 'dv', 'aV', 'aidir', 'adir']], on=['frame', 'id'], how='left', suffixes=('', '_dup'))
    if 'dv_dup' in new_df.columns:
        new_df['dv'] = new_df['dv'].combine_first(new_df['dv_dup'])
        new_df['aV'] = new_df['aV'].combine_first(new_df['aV_dup'])
        new_df['aidir'] = new_df['aidir'].combine_first(new_df['aidir_dup'])
        new_df['adir'] = new_df['adir'].combine_first(new_df['adir_dup'])
        new_df.drop(columns=['dv_dup', 'aV_dup', 'aidir_dup', 'adir_dup'], inplace=True)
    return new_df


def get_intervals(df, flag):
    new_df = df.copy()
    # Sort the DataFrame by object_id and frame
    new_df = new_df.sort_values(by=['id', 'frame'])

    # Create a group identifier that changes when the flag changes
    new_df['group'] = (new_df[flag] != new_df.groupby('id')[flag].shift()).cumsum()

    # Aggregate to get the start and end frames for each interval
    intervals = new_df.groupby(['id', flag, 'group']).agg(start_frame=('frame', 'min'),
                                                      end_frame=('frame', 'max')).reset_index()

    # Drop the group column as it is no longer needed
    intervals = intervals.drop(columns=['group'])

    # Only positive intervals
    return intervals[intervals[flag]==True]


def label_sequence(video_root, config):
    """
    Label a sequence with the given configuration
    :param video_root: path to the sequence
    :param config: configuration dictionary
    """
    gt_path = video_root + '/gt/gt.txt'
    info_path = video_root + '/seqinfo.ini'

    # Extract sequence info
    frame_width, frame_height, video_fps = seq_info(info_path)

    # Read ground truth
    original_columns = ['frame', 'id', 'xl', 'yt', 'w', 'h']
    df = pd.read_csv(gt_path, names=original_columns, usecols=[i for i in range(len(original_columns))])

    # Print some information
    video_name = video_root.split('/')[-1]
    print(f'Video: {video_name}')
    print(f'Frame width: {frame_width}')
    print(f'Frame height: {frame_height}')
    print(f'Video FPS: {video_fps}')
    print(f'Number of frames: {df["frame"].nunique()}')
    print(f'Number of objects: {df["id"].nunique()}')

    # Generic bbox transformations
    df['area'] = df['w'] * df['h']
    df['xc'] = df['xl'] + df['w'] / 2
    df['yc'] = df['yt'] + df['h'] / 2

    ##############################
    # NEAR SENSOR
    # Define interest point and trigger radius
    interest_point = np.array([frame_width // 2, frame_height])
    trigger_radius = frame_height / config['fast_approach']['distance_threshold']

    # Compute minimal distance to interest point
    def near_sensor(row):
        bbox = (row['xc'], row['yc'], row['w'], row['h'])
        d = minimal_distance_to_bbox(interest_point, bbox)
        return True if d <= trigger_radius else False

    df['near_sensor'] = df.apply(near_sensor, axis=1)
    ##############################

    ##############################
    # RECOGNIZE GATHERING
    # Iterate over frames
    for frame_id in df['frame'].unique():
        df = recognize_gather(df, frame_id, config['gather']['area_threshold'],
                              config['gather']['distance_threshold'], config['gather']['min_people'])
    ##############################

    ##############################
    # COMPUTE MOTION DESCRIPTORS
    w = np.array(config["speed_projection"])
    dt = 30  # window frames
    l = 5  # list's length

    # Iterate over ids
    for id in df['id'].unique():
        df = get_motion_descriptors(df, id, w, dt, l)
    ##############################

    ##############################
    # FLAGGING
    df['SS'] = df['aV'] < 0.001
    df['SR'] = df['dv'] > 0.025
    df['FA'] = (df['aidir'] > 0) & df['near_sensor']

    # For each column get its intervals
    behaviour_columns = ['SS', 'SR', 'FA', 'G']
    intervals_df = pd.DataFrame()
    for col in behaviour_columns:
        behaviour_df = get_intervals(df, col)
        intervals_df = pd.concat([intervals_df, behaviour_df])
    ##############################

    # Save results gt with flags
    final_columns = original_columns + behaviour_columns
    df['G'].fillna(False, inplace=True)
    df[behaviour_columns] = df[behaviour_columns].astype(int)
    df.to_csv(video_root + '/gt/auto_gt.txt', index=False, columns=final_columns, header=False)
    # TODO: Save intervals


if __name__ == '__main__':
    # Read config
    with open("./cfg.json", "r") as f:
        config = json.load(f)

    # Read all sequences
    video_root = '/Users/inaki-eab/Desktop/small-fast-detector/tracker/evaluation/TrackEval/data/gt/mot_challenge/MOTHupba-train/MOT17-09'

    # For each sequence run function label_sequence
    label_sequence(video_root, config)
