from itertools import combinations

import cv2
import networkx as nx
import numpy as np

from shapely.geometry import Point, Polygon


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.4
font_thickness = 1
text_color = (255, 255, 255)
background_color = (100, 100, 100)
text_padding = 7


class ActionRecognizer:
    """
    Class that recognizes actions in a frame. Spatial and temporal information is used to recognize actions such as
    gatherings, standing still, fast approach, suddenly running, and overstepping boundaries.
    """
    def __init__(self, config, video_info):
        """
        Initializes the ActionRecognizer object.
        Args:
            config (dict): dictionary containing the configuration parameters for the action recognition.
            video_info (VideoInfo): object containing information about the video.
        """
        self.ar_enabled = config["enabled"]
        self.video_info = video_info
        # Gathering parameters
        self.g_enabled = config["gather"]["enabled"]
        self.g_min_people = config["gather"]["min_people"]
        self.g_distance_threshold = config["gather"]["distance_threshold"]
        self.g_area_threshold = config["gather"]["area_threshold"]
        self.g_speed_threshold = config["gather"]["speed_threshold"]
        self.g_last_n = config["gather"]["last_n"]
        self.g_alpha = config["gather"]["alpha"]
        # Standing still parameters
        self.ss_enabled = config["stand_still"]["enabled"]
        self.ss_speed_threshold = config["stand_still"]["speed_threshold"]
        self.ss_last_n = config["stand_still"]["last_n"]
        self.ss_alpha = config["stand_still"]["alpha"]
        # Fast approach parameters
        self.fa_enabled = config["fast_approach"]["enabled"]
        self.fa_draw = config["fast_approach"]["draw"]
        self.fa_distance_threshold = config["fast_approach"]["distance_threshold"]
        self.interest_point = np.array([self.video_info.resolution_wh[0]//2, self.video_info.resolution_wh[1]]) # bottom center of the frame
        self.trigger_radius = self.interest_point[1]/self.fa_distance_threshold
        # Suddenly running parameters
        self.sr_enabled = config["suddenly_run"]["enabled"]
        self.sr_speed_threshold = config["suddenly_run"]["speed_threshold"]
        self.sr_last_n = config["suddenly_run"]["last_n"]
        self.sr_alpha = config["suddenly_run"]["alpha"]
        # Overstep boundary parameters
        self.osb_enabled = config["overstep_boundary"]["enabled"]
        self.osb_draw = config["overstep_boundary"]["draw"]
        self.osb_line = config["overstep_boundary"]["line"]  # Coords: [xl, yl, xr, yr]
        self.osb_region = self.init_region(osb_line=self.osb_line,
                                           osb_direction=config["overstep_boundary"]["region"])

    def recognize_frame(self, tracks):
        """
        Recognizes actions in a frame.
        Args:
            tracks (list): list of detections in the frame (sv.STrack objects).
        Returns:
            ar_results (dict): dictionary containing the results of the action recognition.
        """
        if not self.ar_enabled:
            return None

        for track in tracks:
            track.G = 0
            track.SS = False
            track.FA = False
            track.SR = False
            track.OB = False

        group_results = {}
        individual_results = {}
        ar_results = {}

        if self.g_enabled:
            group_results["gather"] = self.recognize_gather(tracks)
        if self.ss_enabled:
            individual_results["SS"] = self.recognize_stand_still(tracks)
        if self.fa_enabled:
            individual_results["FA"] = self.recognize_fast_approach(tracks)
        if self.sr_enabled:
            individual_results["SR"] = self.recognize_suddenly_run(tracks)
        if self.osb_enabled:
            individual_results["OB"] = self.recognize_overstep_boundary(tracks)

        ar_results['individual'] = self.merge_individual_actions(individual_results)
        ar_results['group'] = group_results if len(group_results) > 0 else None

        return ar_results

    @staticmethod
    def merge_individual_actions(individual_results):
        """
        Merges the results of the individual actions recognition into a single dictionary.
        Args:
            individual_results (dict): dictionary containing the results of the individual actions recognition.
        Returns:
            merged_results (dict): dictionary containing the merged results of the individual actions recognition.
        """
        merged_results = {}
        for action, results in individual_results.items():
            if results is not None:
                for track_id, bbox in results.items():
                    if track_id not in merged_results:
                        merged_results[track_id] = {'bbox': bbox, 'actions': []}
                    merged_results[track_id]['actions'].append(action)
        return merged_results if len(merged_results) > 0 else None

    def annotate(self, frame, ar_results):
        """
        Annotates the frame with the results of the action recognition.
        Args:
            frame (np.array): frame to be annotated.
            ar_results (dict): dictionary containing the results of the action recognition.
        Returns:
            frame (np.array): annotated frame.
        """
        if not self.ar_enabled:
            return frame

        if self.fa_draw and self.fa_enabled:
            cv2.circle(frame, tuple(self.interest_point), int(self.trigger_radius), (0, 255, 0), thickness=2)

        if self.osb_draw and self.osb_enabled:
            cv2.line(frame, self.osb_line[:2], self.osb_line[2:], (0, 255, 0), thickness=2)
        if self.g_enabled and ar_results["group"] is not None:
            frame = self.annotate_gather(frame, ar_results["group"]["gather"])
        if ar_results["individual"] is not None:
            frame = self.annotate_individual_actions(frame, ar_results["individual"])
        return frame

    @staticmethod
    def annotate_individual_actions(frame, results):
        """
        Annotates the frame with the results of the individual actions recognition. Draws a bounding box around each
        detection and places text in the top right corner of the bounding box.
        Args:
            frame (np.array): frame to be annotated.
            results (dict): dictionary containing the results of the individual actions recognition.
        Returns:
            frame (np.array): annotated frame.
        """
        for track_id, data in results.items():
            x1, y1, x2, y2 = data['bbox'].astype(int)

            text = ','.join(data['actions'])
            text_width, text_height = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=font_scale,
                thickness=font_thickness,
            )[0]

            # Text must be top right corner of the bounding box of the crowd but outside the frame
            text_x = x2 - text_padding - text_width
            text_y = y1 + text_padding + text_height

            text_background_x2 = x2
            text_background_x1 = x2 - 2 * text_padding - text_width

            text_background_y1 = y1
            text_background_y2 = y1 + 2 * text_padding + text_height

            cv2.rectangle(
                img=frame,
                pt1=(text_background_x1, text_background_y1),
                pt2=(text_background_x2, text_background_y2),
                color=background_color,
                thickness=cv2.FILLED,
            )
            cv2.putText(
                img=frame,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=font_scale,
                color=text_color,
                thickness=font_thickness,
                lineType=cv2.LINE_AA,
            )
        return frame

    def recognize_gather(self, tracks):
        """
        Recognizes gatherings in a frame by computing the normalized euclidean distance between all pairs of detections
        and conditioning pairs to have similar areas. Then, it finds the independent chains in the graph and computes
        the bounding box of each crowd.
        Args:
            tracks (list): list of detections in the frame (sv.STrack objects).
        Returns:
            results (dict): dictionary containing the results of the action recognition.
        """
        if not tracks:
            return None

        pairs = []
        # Iterate over all pairs of detections
        for i, j in combinations(range(len(tracks)), 2):
            det1 = tracks[int(i)]
            det2 = tracks[int(j)]

            # If both detections are people proceed to computation
            if (det1.class_ids == 0) and (det2.class_ids == 0):
                distance, a1, a2 = self.compute_ned(det1, det2)
                # If the distance between the detections is less than the threshold and the areas are similar
                if distance <= self.g_distance_threshold and min(a1, a2) / (max(a1, a2) + 1e-9) <= self.g_area_threshold:
                    pixel_s1 = self.get_motion_descriptors(det1, last_n=self.g_last_n, alpha=self.g_alpha)
                    pixel_s2 = self.get_motion_descriptors(det2, last_n=self.g_last_n, alpha=self.g_alpha)
                    # If the average speed of both detections is less than the threshold
                    if pixel_s1 < self.g_speed_threshold and pixel_s2 < self.g_speed_threshold:
                        pairs.append([i, j])

        # Find independent chains in the graph
        crowds = self.get_independent_chains(pairs, self.g_min_people)

        # For each crowd, compute the bounding box and store it in the results dictionary
        if len(crowds) > 0:
            results = {}
            for k, crowd in enumerate(crowds):
                crowd_box = self.compute_crowd_box(tracks, crowd)
                results[k] = crowd_box
                for i in crowd:
                    tracks[i].G = k+1
            return results
        else:
            return None

    @staticmethod
    def compute_ned(det1, det2):
        """
        Computes the normalized euclidean distance between two detections using the center of mass of the bounding
        boxes, and the mean area of the bounding boxes as normalization factor.
        Args:
            det1 (sv.STrack): first detection.
            det2 (sv.STrack): second detection.
        Returns:
            normalized_distance (float): normalized euclidean distance between the two detections.
            a1 (float): area of the first bounding box.
            a2 (float): area of the second bounding box.
        """
        # Euclidean distance between center of masses
        distance = np.sqrt(np.sum((det1.mean[0:2] - det2.mean[0:2]) ** 2))
        # Mean area of the bounding boxes
        a1 = det1.tlwh[2] * det1.tlwh[3]
        a2 = det2.tlwh[2] * det2.tlwh[3]
        mean_area = (a1 + a2) / 2
        return distance/np.sqrt(mean_area), a1, a2

    @staticmethod
    def get_independent_chains(pairs, min_people=3):
        """
        Finds the independent chains in a graph, where each node is a detection and each edge is a pair of detections
        linked by the distance threshold.
        Args:
            pairs (list): list of pairs of detections.
            min_people (int): minimum number of people to consider a chain.
        Returns:
            valid_chains (list): list of lists of detections, where each list is an independent chain.
        """
        # Initialize graph
        g = nx.Graph()
        g.add_edges_from(pairs)
        # Find connected components
        independent_chains = list(nx.connected_components(g))
        # Filter out chains having less than 3 elements
        valid_chains = [chain for chain in independent_chains if len(chain) > (min_people-1)]
        return valid_chains

    @staticmethod
    def compute_crowd_box(tracks, crowd):
        """
        Computes the bounding box of a crowd of people.
        Args:
            tracks (list): list of detections in the frame (sv.STrack objects).
            crowd (list): list of detections that form a crowd.
        Returns:
            crowd_box (list): list containing the coordinates of the bounding box of the crowd.
        """
        # Get the coordinates of the bounding boxes of the detections in the crowd
        crowd_boxes = [tracks[i].tlbr for i in crowd]
        # Compute the bounding box of the crowd with margin for individual actions
        crowd_box = [min([box[0] for box in crowd_boxes])-text_padding*4,
                     min([box[1] for box in crowd_boxes])-text_padding*4,
                     max([box[2] for box in crowd_boxes])+text_padding*4,
                     max([box[3] for box in crowd_boxes])+text_padding*4]
        return np.array(crowd_box)

    def recognize_stand_still(self, tracks):
        """
        Recognizes people standing still in a frame by computing the average speed of each detection and comparing it
        to a threshold.
        Args:
            tracks (list): list of detections in the frame (sv.STrack objects).
        Returns:
            results (dict): dictionary containing the results of the action recognition.
        """
        if not tracks:
            return None
        ss_results = {}
        for track in tracks:
            if track.class_ids == 0:
                pixel_s = self.get_motion_descriptors(track, last_n=self.ss_last_n, alpha=self.ss_alpha)
                if pixel_s < self.ss_speed_threshold:
                    ss_results[track.track_id] = track.tlbr
                    track.SS = True
        return ss_results if len(ss_results.keys()) > 0 else None

    @staticmethod
    def get_motion_descriptors(track, alpha=0, last_n=150):
        """
        Computes the average speed and direction of a detection.
        Args:
            track (sv.STrack): detection to compute the motion descriptors.
            alpha (float): smoothing factor for the exponential moving average, a value close to 1 gives more weight to
                the most recent state, a value close to 0 provides a smoother result. If None, no smoothing is applied.
            last_n (int): number of states to consider for the computation.
        Returns:
            avg_speed (float): average speed of the detection.
        """
        # track.states[-1] is most recent state, track.states[0] is the oldest state
        states = np.array(track.norm_speed)
        if len(states) == 0:
            return 0

        # Get last_n states
        states = states[-last_n:]
        # Initialize EMA with the first state
        smoothed_speed = states[0]
        # Compute EMA
        for speed in states[1:]:
            smoothed_speed = alpha * speed + (1 - alpha) * smoothed_speed
        return smoothed_speed

    def recognize_fast_approach(self, tracks):
        """
        Recognizes fast approaches in a frame by computing the closest distance between each detection and the interest
        point, and comparing it to a threshold.
        Args:
            tracks (list): list of detections in the frame (sv.STrack objects).
        Returns:
            results (dict): dictionary containing the results of the action recognition.
        """
        valid_classes = [0, 1, 2]   # Person, Car, Truck
        fa_results = {}
        for track in tracks:
            if track.class_ids in valid_classes and track.frame_id > 1:
                # Check if any point of the bounding box is inside the trigger radius
                dx = max(abs(track.mean[0] - self.interest_point[0]) - track.mean[2] / 2, 0)
                dy = max(abs(track.mean[1] - self.interest_point[1]) - track.mean[3] / 2, 0)
                if np.sqrt(dx ** 2 + dy ** 2) <= self.trigger_radius:
                    fa_results[track.track_id] = track.tlbr
                    track.FA = True
        return fa_results if len(fa_results.keys()) > 0 else None

    def recognize_suddenly_run(self, tracks):
        """
        Recognizes people running suddenly in a frame by computing the average speed of each detection and comparing it
        to a threshold.
        Args:
            tracks (list): list of detections in the frame (sv.STrack objects).
        Returns:
            results (dict): dictionary containing the results of the action recognition.
        """
        sr_results = {}
        for track in tracks:
            if track.class_ids == 0 and track.frame_id > 1:
                pixel_s = self.get_motion_descriptors(track, last_n=self.sr_last_n, alpha=self.sr_alpha)
                if pixel_s > self.sr_speed_threshold:
                    sr_results[track.track_id] = track.tlbr
                    track.SR = True
        return sr_results if len(sr_results.keys()) > 0 else None

    @staticmethod
    def annotate_gather(frame, crowd_results):
        """
        Annotates the frame with the results of the gathering recognition. Draws a bounding box around each crowd. Text
        is placed in the bottom right corner of the bounding box.
        Args:
            frame (np.array): frame to be annotated.
            crowd_results (dict): dictionary containing the results of the gathering recognition.
        Returns:
            frame (np.array): annotated frame.
        """
        if crowd_results is not None:
            for idx, bbox in crowd_results.items():
                x1, y1, x2, y2 = bbox.astype(int)

                cv2.rectangle(
                    img=frame,
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    color=background_color,
                    thickness=2,
                )

                text = f"G #{idx+1}"
                text_width, text_height = cv2.getTextSize(
                    text=text,
                    fontFace=font,
                    fontScale=font_scale,
                    thickness=font_thickness,
                )[0]

                text_x = x2 - text_padding - text_width
                text_y = y2 + text_padding + text_height

                text_background_x2 = x2
                text_background_x1 = x2 - 2 * text_padding - text_width

                text_background_y1 = y2
                text_background_y2 = y2 + 2 * text_padding + text_height

                cv2.rectangle(
                    img=frame,
                    pt1=(text_background_x1, text_background_y1),
                    pt2=(text_background_x2, text_background_y2),
                    color=background_color,
                    thickness=cv2.FILLED,
                )
                cv2.putText(
                    img=frame,
                    text=text,
                    org=(text_x, text_y),
                    fontFace=font,
                    fontScale=font_scale,
                    color=text_color,
                    thickness=font_thickness,
                    lineType=cv2.LINE_AA,
                )

        return frame

    def recognize_overstep_boundary(self, tracks):
        """
        Recognizes people overstepping the boundary line in a frame by checking if the bounding box of each detection
        crosses the boundary line.
        Args:
            tracks (list): list of detections in the frame (sv.STrack objects).
        Returns:
            osb_results (dict): dictionary containing the results of the action recognition.
        """
        osb_results = {}
        for track in tracks:
            # Check if the bbox crosses the boundary line
            if track.class_ids == 0:
                # TODO: use CM or Lower Center or something else?
                if self.osb_region.contains(Point(track.mean[:2])):
                    osb_results[track.track_id] = track.tlbr
                    track.OB = True
        return osb_results if len(osb_results.keys()) > 0 else None

    def init_region(self, osb_line, osb_direction):
        """
        Initializes the region of interest for the overstep boundary action recognition.
        Args:
            osb_line (list): list containing the coordinates of the boundary line.
            osb_direction (str): direction of the boundary line.
        Returns:
            region (Polygon): region of interest for the overstep boundary action recognition.
        """
        # Check line coords follows left to right direction [xl, yl, xr, yr]
        assert osb_line[0] <= osb_line[2], "Line coords must follow left to right direction"

        frame_width, frame_height = self.video_info.resolution_wh

        # Create region of interest, defined as [X_tl, X_tr, X_bl, X_br]
        if osb_direction == "bottom":
            X_tl = np.array([0, osb_line[1]])
            X_tr = np.array([frame_width, osb_line[3]])
            X_bl = np.array([0, frame_height])
            X_br = np.array([frame_width, frame_height])

        elif osb_direction == "top":
            X_tl = np.array([0, 0])
            X_tr = np.array([frame_width, 0])
            X_bl = np.array([0, osb_line[1]])
            X_br = np.array([frame_width, osb_line[3]])

        elif osb_direction == "left":
            X_tl = np.array([0, 0])
            X_tr = np.array([osb_line[0], 0])
            X_bl = np.array([0, frame_height])
            X_br = np.array([osb_line[2], frame_height])

        elif osb_direction == "right":
            X_tl = np.array([osb_line[0], 0])
            X_tr = np.array([frame_width, 0])
            X_bl = np.array([osb_line[2], frame_height])
            X_br = np.array([frame_width, frame_height])

        return Polygon([X_tl, X_tr, X_br, X_bl])

    @staticmethod
    def annotate_overstep_boundary(frame, osb_results):
        """
        Annotates the frame with the results of the overstep boundary recognition. Draws a bounding box around each
        detection that oversteps the boundary line. Text is placed in the top right corner of the bounding box.
        Args:
            frame (np.array): frame to be annotated.
            osb_results (dict): dictionary containing the results of the overstep boundary recognition.
        Returns:
            frame (np.array): annotated frame.
        """
        if osb_results is not None:
            for idx, bbox in osb_results.items():
                x1, y1, x2, y2 = bbox.astype(int)

                cv2.rectangle(
                    img=frame,
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    color=(0, 0, 255),
                    thickness=2,
                )

                text = "OSB"
                text_width, text_height = cv2.getTextSize(
                    text=text,
                    fontFace=font,
                    fontScale=font_scale,
                    thickness=font_thickness,
                )[0]

                text_x = max(x1, min(x2 - text_width - 3 * text_padding, frame.shape[1] - text_width - 3 * text_padding))
                text_y = max(y1 + text_height + 3 * text_padding, text_height + 3 * text_padding)

                text_background_x1 = text_x - text_padding
                text_background_y1 = text_y - text_height - 2 * text_padding
                text_background_x2 = text_x + text_width + text_padding
                text_background_y2 = text_y + text_padding

                cv2.rectangle(
                    img=frame,
                    pt1=(text_background_x1, text_background_y1),
                    pt2=(text_background_x2, text_background_y2),
                    color=background_color,
                    thickness=cv2.FILLED,
                )
                cv2.putText(
                    img=frame,
                    text=text,
                    org=(text_x, text_y - text_padding),
                    fontFace=font,
                    fontScale=font_scale,
                    color=text_color,
                    thickness=font_thickness,
                    lineType=cv2.LINE_AA,
                )

        return frame
