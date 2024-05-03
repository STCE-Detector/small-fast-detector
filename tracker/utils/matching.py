from typing import Tuple

import numpy as np
import scipy
from scipy.spatial.distance import cdist

from tracker.utils.preprocessing import chi2inv95
from ultralytics.utils.metrics import bbox_ioa

try:
    import lap  # for linear_assignment

    assert lap.__version__  # verify package is not directory
except (ImportError, AssertionError, AttributeError):
    from ultralytics.utils.checks import check_requirements

    check_requirements('lapx>=0.5.2')  # update to lap package from https://github.com/rathaROG/lapx
    import lap


def indices_to_matches(cost_matrix: np.ndarray, indices: np.ndarray, thresh: float) -> Tuple[np.ndarray, tuple, tuple]:
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = matched_cost <= thresh

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))
    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh, use_lap=True):
    """
    Perform linear assignment using scipy or lap.lapjv.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments.
        thresh (float): Threshold for considering an assignment valid.
        use_lap (bool, optional): Whether to use lap.lapjv. Defaults to True.

    Returns:
        (tuple): Tuple containing matched indices, unmatched indices from 'a', and unmatched indices from 'b'.
    """

    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if use_lap:
        # https://github.com/gatagat/lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        x, y = scipy.optimize.linear_sum_assignment(cost_matrix)  # row x, col y
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
        if len(matches) == 0:
            unmatched_a = list(np.arange(cost_matrix.shape[0]))
            unmatched_b = list(np.arange(cost_matrix.shape[1]))
        else:
            unmatched_a = list(set(np.arange(cost_matrix.shape[0])) - set(matches[:, 0]))
            unmatched_b = list(set(np.arange(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def iou_distance(atracks, btracks, b=0.0, type='iou'):
    """
    Compute cost based on Intersection over Union (IoU) between tracks or variations of IoU.

    Args:
        atracks (list[STrack] | list[np.ndarray]): List of tracks 'a' or bounding boxes.
        btracks (list[STrack] | list[np.ndarray]): List of tracks 'b' or bounding boxes.
        b (float, optional): Buffer for detection bounding boxes. Defaults to 0.
        type (str, optional): Type of IoU distance to compute. Defaults to 'iou'.

    Returns:
        (np.ndarray): Cost matrix computed based on IoU.
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) \
            or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    atlbrs = np.ascontiguousarray(atlbrs, dtype=np.float32)
    btlbrs = np.ascontiguousarray(btlbrs, dtype=np.float32)

    if b > 0 and len(btlbrs) > 0:
        widths = btlbrs[:, 2] - btlbrs[:, 0]
        heights = btlbrs[:, 3] - btlbrs[:, 1]
        buffer_widths = widths * b
        buffer_heights = heights * b
        buffered_boxes = np.empty_like(btlbrs)
        buffered_boxes[:, 0] = btlbrs[:, 0] - buffer_widths
        buffered_boxes[:, 1] = btlbrs[:, 1] - buffer_heights
        buffered_boxes[:, 2] = btlbrs[:, 2] + buffer_widths
        buffered_boxes[:, 3] = btlbrs[:, 3] + buffer_heights
        btlbrs = buffered_boxes

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if len(atlbrs) and len(btlbrs):
        ious = bbox_bbsi(atlbrs, btlbrs, type=type)

        # Penalize the cost matrix for different classes
        class_ids_a = np.array([track.class_ids for track in atracks])
        class_ids_b = np.array([track.class_ids for track in btracks])
        class_mask = (class_ids_a[:, None] != class_ids_b[None, :]).astype(np.float32)
        ious *= (1 - class_mask)  # Element-wise multiplication to zero out IoUs for different classes

    return 1 - ious  # cost matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    Compute distance between tracks and detections based on embeddings.

    Args:
        tracks (list[STrack]): List of tracks.
        detections (list[BaseTrack]): List of detections.
        metric (str, optional): Metric for distance computation. Defaults to 'cosine'.

    Returns:
        (np.ndarray): Cost matrix computed based on embeddings.
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Normalized features
    return cost_matrix


def fuse_score(cost_matrix, detections):
    """
    Fuses cost matrix with detection scores to produce a single similarity matrix.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments.
        detections (list[BaseTrack]): List of detections with scores.

    Returns:
        (np.ndarray): Fused similarity matrix.
    """

    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # fuse_cost


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def gate(cost_matrix, emb_cost):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    if cost_matrix.size == 0:
        return cost_matrix

    index = emb_cost > 0.3
    cost_matrix[index] = 1

    return cost_matrix


def bbox_bbsi(box1, box2, type='iou', eps=1e-7):
    """
    Calculate the Bounding Box Similarity Index (BBSI) between two sets of bounding boxes.
    Args:
        box1 (np.ndarray): The first set of bounding boxes.
        box2 (np.ndarray): The second set of bounding boxes.
        type (str, optional): The type of similarity index to calculate. Defaults to 'iou'.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    h_intersection = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0)
    w_intersection = (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)

    # Calculate the intersection area
    intersection = h_intersection * w_intersection

    # Calculate the union area
    box1_height = b1_x2 - b1_x1
    box2_height = b2_x2 - b2_x1
    box1_width = b1_y2 - b1_y1
    box2_width = b2_y2 - b2_y1

    box1_area = box1_height * box1_width
    box2_area = box2_height * box2_width

    if type == 'iou_1way':
        return intersection / (box1_area + eps)

    if type == 'iou_2way':
        return intersection / (box2_area + eps)

    union = (box2_area + box1_area[:, None] - intersection + eps)

    # Calculate the IoU
    iou = intersection / union

    if type == 'iou':
        return iou

    if type == 'hmiou':
        # TODO: can be used in conjunction with other metrics
        w_union = np.maximum(b1_y2[:, None], b2_y2) - np.minimum(b1_y1[:, None], b2_y1)
        return iou * (w_intersection / w_union)

    # Calculate the DIoU
    centerx1 = (b1_x1 + b1_x2) / 2.0
    centery1 = (b1_y1 + b1_y2) / 2.0
    centerx2 = (b2_x1 + b2_x2) / 2.0
    centery2 = (b2_y1 + b2_y2) / 2.0
    inner_diag = np.abs(centerx1[:, None] - centerx2) + np.abs(centery1[:, None] - centery2)

    xxc1 = np.minimum(b1_x1[:, None], b2_x1)
    yyc1 = np.minimum(b1_y1[:, None], b2_y1)
    xxc2 = np.maximum(b1_x2[:, None], b2_x2)
    yyc2 = np.maximum(b1_y2[:, None], b2_y2)
    outer_diag = np.abs(xxc2 - xxc1) + np.abs(yyc2 - yyc1)

    diou = iou - (inner_diag / outer_diag)

    if type == 'diou':
        return diou

    # Calculate the BBSI
    delta_w = np.abs(box2_width - box1_width[:, None])
    sw = w_intersection / np.abs(w_intersection + delta_w + eps)

    delta_h = np.abs(box2_height - box1_height[:, None])
    sh = h_intersection / np.abs(h_intersection + delta_h + eps)

    bbsi = diou + sh + sw

    # Normalize the BBSI
    n_bbsi = (bbsi) / 3.0

    if type == 'bbsi':
        return n_bbsi
