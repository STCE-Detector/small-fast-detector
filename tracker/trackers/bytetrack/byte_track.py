import os
from collections import deque

import numpy as np
import torch

from ultralytics.utils.downloads import safe_download
from tracker.trackers.bytetrack.basetrack import BaseTrack, TrackState
from tracker.utils import matching
from tracker.utils.gmc import GMC
from tracker.utils.kalman_filter import KalmanFilterXYAH
from tracker.utils.preprocessing import extract_image_patches
from tracker.utils.slm import load_model
from .detections import Detections


class STrack(BaseTrack):
    """
    Single object tracking representation that uses Kalman filtering for state estimation.

    This class is responsible for storing all the information regarding individual tracklets and performs state updates
    and predictions based on Kalman filter.

    Attributes:
        shared_kalman (KalmanFilterXYAH): Shared Kalman filter that is used across all STrack instances for prediction.
        _tlwh (np.ndarray): Private attribute to store top-left corner coordinates and width and height of bounding box.
        kalman_filter (KalmanFilterXYAH): Instance of Kalman filter used for this particular object track.
        mean (np.ndarray): Mean state estimate vector.
        covariance (np.ndarray): Covariance of state estimate.
        is_activated (bool): Boolean flag indicating if the track has been activated.
        score (float): Confidence score of the track.
        tracklet_len (int): Length of the tracklet.
        cls (any): Class label for the object.
        idx (int): Index or identifier for the object.
        frame_id (int): Current frame ID.
        start_frame (int): Frame where the object was first detected.

    Methods:
        predict(): Predict the next state of the object using Kalman filter.
        multi_predict(stracks): Predict the next states for multiple tracks.
        multi_gmc(stracks, H): Update multiple track states using a homography matrix.
        activate(kalman_filter, frame_id): Activate a new tracklet.
        re_activate(new_track, frame_id, new_id): Reactivate a previously lost tracklet.
        update(new_track, frame_id): Update the state of a matched track.
        convert_coords(tlwh): Convert bounding box to x-y-angle-height format.
        tlwh_to_xyah(tlwh): Convert tlwh bounding box to xyah format.
        tlbr_to_tlwh(tlbr): Convert tlbr bounding box to tlwh format.
        tlwh_to_tlbr(tlwh): Convert tlwh bounding box to tlbr format.
    """

    shared_kalman = KalmanFilterXYAH()

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50, speed_projection=None):
        """Initialize new STrack instance."""
        self._tlwh = np.asarray(self.tlbr_to_tlwh(tlwh[:-1]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = tlwh[-1]

        # SMILE FEATURES
        self.features = deque([], maxlen=feat_history)
        self.class_ids = -1
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(cls, score)
        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.alpha = 0.9

        # Action Recognition initialization
        self.speed_projection = speed_projection
        self.norm_speed = deque([], maxlen=150)
        self.prev_state = None
        self.SS = False
        self.SR = False
        self.FA = False
        self.G = False
        self.OB = False

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            # Exponential moving average
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, class_ids, score):
        """Updates the class ID and score for the track."""
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if class_ids == c[0]:
                    c[1] += score
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.class_ids = c[0]
            if not found:
                self.cls_hist.append([class_ids, score])
                self.class_ids = class_ids
        else:
            self.cls_hist.append([class_ids, score])
            self.class_ids = class_ids

    def predict(self):
        """Predicts mean and covariance using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            # TODO: mean_state[6] = 0?
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """Perform multi-object predictive tracking using Kalman filter for given stracks."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                # TODO: mean_state[6] = 0?
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """Update state tracks positions and covariances using a homography matrix."""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        # Action Recognition
        self.prev_state = self.mean

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivates a previously lost track with a new detection."""
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.convert_coords(new_track.tlwh), new_track.score)
        self.tracklet_len = 0

        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx
        # Action Recognition
        self.prev_state = self.mean

    def update(self, new_track, frame_id):
        """
        Update the state of a matched track.

        Args:
            new_track (STrack): The new track containing updated information.
            frame_id (int): The ID of the current frame.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.convert_coords(new_tlwh), new_track.score)
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx

        # Action Recognition
        if self.prev_state is not None:
            # Compute speed
            speed = np.linalg.norm((self.mean[:2] - self.prev_state[:2])*self.speed_projection)
            # Normalize speed by current area
            speed /= np.sqrt(new_track.tlwh[2] * new_track.tlwh[3])
            # Append speed to buffer
            self.norm_speed.append(speed)
        else:
            self.norm_speed.append(0)

        self.prev_state = self.mean

    def convert_coords(self, tlwh):
        """Convert a bounding box's top-left-width-height format to its x-y-angle-height equivalent."""
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self):
        """Get current position in bounding box format (top left x, top left y, width, height)."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format (min x, min y, max x, max y), i.e., (top left, bottom right)."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format (center x, center y, aspect ratio, height), where the aspect ratio is width /
        height.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """Converts top-left bottom-right format to top-left width height format."""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        """Converts tlwh bounding box format to tlbr format."""
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        """Return a string representation of the BYTETracker object with start and end frames and track ID."""
        return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'


class ByteTrack:
    """
    SMILETracker: A tracking algorithm built on top of YOLOv8 for object detection and tracking with REID.

    The class is responsible for initializing, updating, and managing the tracks for detected objects in a video
    sequence. It maintains the state of tracked, lost, and removed tracks over frames, utilizes Kalman filtering for
    predicting the new object locations, and performs data association. Reid is used for matching tracks to detections.

    Attributes:
        tracked_stracks (list[STrack]): List of successfully activated tracks.
        lost_stracks (list[STrack]): List of lost tracks.
        removed_stracks (list[STrack]): List of removed tracks.
        frame_id (int): The current frame ID.
        args (namespace): Command-line arguments.
        max_time_lost (int): The maximum frames for a track to be considered as 'lost'.
        kalman_filter (object): Kalman Filter object.

    Methods:
        update(results, img=None): Updates object tracker with new detections.
        get_kalmanfilter(): Returns a Kalman filter object for tracking bounding boxes.
        init_track(dets, scores, cls, img=None): Initialize object tracking with detections.
        get_dists(tracks, detections): Calculates the distance between tracks and detections.
        multi_predict(tracks): Predicts the location of tracks.
        reset_id(): Resets the ID counter of STrack.
        joint_stracks(tlista, tlistb): Combines two lists of stracks.
        sub_stracks(tlista, tlistb): Filters out the stracks present in the second list from the first list.
        remove_duplicate_stracks(stracksa, stracksb): Removes duplicate stracks based on IOU.
    """

    def __init__(self, config, video_info):
        """Initialize a YOLOv8 object to track objects with given arguments and frame rate."""

        self.active_tracks = []     # Only for csv writing
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.reset_id()

        self.frame_id = 0
        self.args = config["tracker_args"]

        self.track_high_thresh = self.args["track_high_thresh"]
        self.track_low_thresh = self.args["track_low_thresh"]
        self.new_track_thresh = self.args["new_track_thresh"]
        self.first_match_thresh = self.args["first_match_thresh"]
        self.second_match_thresh = self.args["second_match_thresh"]
        self.new_match_thresh = self.args["new_match_thresh"]
        self.first_buffer = self.args["first_buffer"]
        self.second_buffer = self.args["second_buffer"]
        self.new_buffer = self.args["new_buffer"]
        self.first_fuse = self.args["first_fuse"]
        self.second_fuse = self.args["second_fuse"]
        self.new_fuse = self.args["new_fuse"]

        self.iou_type_dict = {
            0: 'iou',
            1: 'iou_1way',
            2: 'iou_2way',
            3: 'diou',
            4: 'bbsi',
            5: 'hmiou',
        }
        self.first_iou = self.iou_type_dict[self.args["first_iou"]]
        self.second_iou = self.iou_type_dict[self.args["second_iou"]]
        self.new_iou = self.iou_type_dict[self.args["new_iou"]]

        self.buffer_size = np.int8(video_info.fps / 30.0 * self.args["track_buffer"])
        self.max_time_lost = self.buffer_size
        self.cw_thresh = self.args["cw_thresh"]   # 0 to deactivate
        self.nk_flag = self.args["nk_flag"]
        self.nk_alpha = self.args["nk_alpha"]
        self.kalman_filter = self.get_kalmanfilter()

        # ReID module
        self.with_reid = self.args["with_reid"]
        self.device = self.args["device"]
        self.proximity_thresh = self.args["proximity_thresh"]
        self.appearance_thresh = self.args["appearance_thresh"]

        if self.with_reid:
            if self.args["reid_default"]:
                self.weight_path = self.args["weight_path"]
                # check if self.weight_path is exists if not asset
                if not os.path.exists(self.weight_path):
                    safe_download('https://drive.google.com/file/d/1RDuVo7jYBkyBR4ngnBaVQUtHL8nAaGaL/view',
                                  self.weight_path)
                self.encoder = load_model(self.weight_path)
                if self.device == 'cuda':
                    self.encoder = self.encoder.to(torch.device('cuda'))
                elif self.device == 'mps':
                    self.encoder = self.encoder.to(torch.device('mps'))
                else:
                    self.encoder = self.encoder.to(torch.device('cpu'))
                self.encoder = self.encoder.eval()
            else:
                self.with_reid = False

        # GMC module
        self.gmc = GMC(method=self.args["gmc_method"])   # TODO: review the GMC module, bouncing boxes

        # Action Recognition
        # TODO: solve when config is ConfigParser
        if hasattr(config, "keys"):
            if "action_recognition" in config:
                self.speed_projection = np.array(config["action_recognition"]["speed_projection"])
            else:
                # TODO: use best known values
                self.speed_projection = np.array([1, 1])
        else:
            if "action_recognition" in config.config:
                self.speed_projection = np.array(config["action_recognition"]["speed_projection"])
            else:
                # TODO: use best known values
                self.speed_projection = np.array([1, 1])

    def update(self, results, img=None):
        """Updates object tracker with new detections and returns tracked object bounding boxes."""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.confidence
        bboxes = results.xyxy
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)   # Add index to bboxes
        cls = results.class_id
        # TODO: features = results.features?

        remain_inds = scores >= self.track_high_thresh   # Select indices with high scores for first association
        inds_low = scores >= self.track_low_thresh
        inds_high = scores < self.track_high_thresh
        inds_second = inds_low & inds_high   # Select indices with low scores for second association

        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = cls[remain_inds]
        cls_second = cls[inds_second]
        features_keep = None


        # STEP 1: Feature extraction and create embedding
        if self.with_reid:
            # TODO: batch inference? Maybe at fixed size?
            patches_det = extract_image_patches(img, dets)
            features = torch.zeros((len(patches_det), 128), dtype=torch.float32)

            for time in range(len(patches_det)):
                patches_det[time] = torch.tensor(patches_det[time]).to(self.device)
                features[time, :] = self.encoder.inference_forward_fast(patches_det[time].type(torch.float32))

            features_keep = features.cpu().detach().numpy()

        # Initialize high score detections
        detections = self.init_track(dets, scores_keep, cls_keep, features_keep, img)
        # Differentiate between unconfirmed (previously lost) and tracked stracks (have been activated in last frame)
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)


        # STEP 2: First association, with high score detection boxes
        # Join lost stracks to tracked stracks for the first association
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        self.multi_predict(strack_pool)

        # Fix camera motion
        if hasattr(self, 'gmc') and img is not None:
            warp = self.gmc.apply(img, dets)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        # Compute distance matrix
        dists = self.get_dists(strack_pool, detections,
                               conf_fuse=self.first_fuse,
                               reid=True,
                               buffer=self.first_buffer,
                               iou_type=self.first_iou)
        # Perform data association
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.first_match_thresh)

        # Update matched stracks with matched detections
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)


        # STEP 3: Second association, with low score detection boxes association the untrack to the low score detections
        # Initialize low score detections
        detections_second = self.init_track(dets_second, scores_second, cls_second, img=img)

        # Select unmatched tracks from the first association
        r_tracked_stracks = [strack_pool[i] for i in u_track if (strack_pool[i].state == TrackState.Tracked)]

        # Compute distance matrix only based on IoU
        dists = self.get_dists(r_tracked_stracks, detections_second,
                               conf_fuse=self.second_fuse,
                               reid=False,
                               buffer=self.second_buffer,
                               iou_type=self.second_iou)
        # Perform data association only based on IoU distance and greater than 0.5
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=self.second_match_thresh)

        # Update matched stracks with matched detections
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Deal with unmatched tracks and mark them as lost
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Deal with unmatched detections from the first association and unconfirmed tracks (usually tracks with only one frame)
        #TODO: is this necessary? Can't we just activate the unconfirmed tracks in their first frame?
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections,
                               conf_fuse=self.new_fuse,
                               reid=False,
                               buffer=self.new_buffer,
                               iou_type=self.new_iou)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.new_match_thresh)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        # Unconfirmed tracks that are not matched with any detections are removed
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)


        # STEP 4: Init new stracks: high confidence detections that are unmatched in the first association and with unconfirmed tracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)


        # STEP 5: Update state
        # Deal with lost tracks, remove if tracked for too long
        for track in self.lost_stracks:
            if (self.frame_id - track.end_frame) > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Merge
        self.tracked_stracks = [t for t in self.tracked_stracks if (t.state == TrackState.Tracked)]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]  # clip remove stracks to 1000 maximum
        # Filter activated tracks
        self.active_tracks = [track for track in self.tracked_stracks if track.is_activated]


        # STEP 6: Process detection information of activated tracks so that it can be used for further visualization
        # Initialize empty arrays for detections attributes
        xyxy = []
        class_ids = []
        tracker_ids = []
        confidences = []

        # Prepare data for Detections and tracks in a single loop
        for track in self.active_tracks:
            xyxy.append(track.tlbr)
            class_ids.append(int(track.class_ids))
            tracker_ids.append(int(track.track_id))
            confidences.append(track.score)

        # Convert lists to NumPy arrays
        xyxy = np.array(xyxy, dtype=np.float32) if xyxy else np.empty((0, 4), dtype=np.float32)
        class_ids = np.array(class_ids, dtype=int) if class_ids else np.empty(0, dtype=int)
        tracker_ids = np.array(tracker_ids, dtype=int) if tracker_ids else np.empty(0, dtype=int)
        confidences = np.array(confidences, dtype=np.float32) if confidences else np.empty(0, dtype=np.float32)

        # Create Detections object
        detections = Detections(xyxy=xyxy, class_id=class_ids, tracker_id=tracker_ids, confidence=confidences)

        return detections, self.active_tracks

    def get_kalmanfilter(self):
        """Returns a Kalman filter object for tracking bounding boxes."""
        return KalmanFilterXYAH(self.cw_thresh, self.nk_flag, self.nk_alpha)

    def init_track(self, dets, scores, cls, features=None, img=None):
        """Initialize object tracking with detections and scores using STrack algorithm."""
        if len(dets) > 0:
            """Detections."""
            if self.with_reid and features is not None:
                detections = [STrack(xyxy,
                                     s, c, f,
                                     speed_projection=self.speed_projection)
                              for (xyxy, s, c, f) in zip(dets, scores, cls, features)]
            else:
                detections = [STrack(xyxy,
                                     s, c,
                                     speed_projection=self.speed_projection)
                              for (xyxy, s, c) in zip(dets, scores, cls)]
        else:
            detections = []

        return detections

    def get_dists(self, tracks, detections, conf_fuse=True, reid=False, buffer=0, iou_type='iou'):
        """Get distances between tracks and detections using IoU and (optionally) ReID embeddings."""

        dists = matching.iou_distance(tracks, detections, b=buffer, type=iou_type)
        # TODO: set it in config
        if conf_fuse:
            # Originally only used with MOT20 dataset
            dists = matching.fuse_score(dists, detections)

        if self.with_reid and (self.encoder is not None) and reid:
            dists_mask = (dists > self.proximity_thresh)
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[dists_mask] = 1.0
            # TODO: should be sum?
            dists = np.minimum(dists, emb_dists)

            # TODO: SMILE TRACK IMPLEMENTATION
            # emb_dists = matching.embedding_distance(tracks, detections)
            # emb_dists = matching.fuse_motion(self.kalman_filter, emb_dists, tracks, detections)
            # if emb_dists.size != 0:
            #    # This gate is the same as appearance threshold but with fixed value 0.3
            #    dists = matching.gate(dists, emb_dists)

        return dists

    def multi_predict(self, tracks):
        """Returns the predicted tracks using the YOLOv8 network."""
        STrack.multi_predict(tracks)

    def reset_id(self):
        """Resets the ID counter of STrack."""
        STrack.reset_id()

    def reset(self):
        """Reset tracker."""
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()
        self.gmc.reset_params()

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """Combine two lists of stracks into a single one."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """DEPRECATED CODE in https://github.com/ultralytics/ultralytics/pull/1890/
        stracks = {t.track_id: t for t in tlista}
        for t in tlistb:
            tid = t.track_id
            if stracks.get(tid, 0):
                del stracks[tid]
        return list(stracks.values())
        """
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """Remove duplicate stracks with non-maximum IOU distance."""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
