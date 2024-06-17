import json
import os
import numpy as np
import supervision as sv
from tqdm import tqdm
import configparser

import tracker.trackers as trackers
from tracker.action_recognition import ActionRecognizer


class SequenceProcessor:
    """
    Process a sequence of detections using a tracker and save the results to a txt file.
    """

    def __init__(self, config, sequence_path, experiment_id=None):
        super().__init__()
        self.config = config

        self.detections_path = sequence_path

        self.dataset = sequence_path.split('/')[-2]
        suffix = "" if experiment_id is None else f"_{experiment_id}"
        self.experiment_name = config["name"] + "_" + sequence_path.split('/')[-3] + suffix
        self.sequence_name = sequence_path.split('/')[-1]
        self.output_dir = "./outputs/tracks/" + self.dataset + '/' + self.experiment_name + '/data'
        os.makedirs(self.output_dir, exist_ok=True)
        self.txt_path = self.output_dir + f"/{self.sequence_name}.txt"

        self.dataset_root = config["source_gt_dir"]
        self.dataset_sequence = self.dataset_root + '/' + self.sequence_name
        self.sequence_info = self.get_sequence_info()

        self.device = config["device"]

        self.tracker = getattr(trackers, config["tracker_name"])(config["tracker_args"], self.sequence_info)
        self.action_recognizer = ActionRecognizer(config["action_recognition"], self.sequence_info)

        self.data_dict = {
            "frame_id": [],
            "tracker_id": [],
            "class_id": [],
            "xl": [],
            "yt": [],
            "w": [],
            "h": [],
            "conf": [],
            "SS": [],
            "SR": [],
            "FA": [],
            "G": [],
        }

    def process_sequence(self, print_bar=False):
        for i, txt_file in enumerate(tqdm(sorted(os.listdir(self.detections_path)), desc=f"Processing {self.sequence_name}", unit=" frames", disable=not print_bar)):
            txt_path = os.path.join(self.detections_path, txt_file)

            # READ TXT FILE AND PROCESS IT
            detections = self.get_detections(txt_path)

            # UPDATE TRACKER
            _, tracks = self.tracker.update(detections, i+1)

            # UPDATE ACTION RECOGNIZER
            _ = self.action_recognizer.recognize_frame(tracks)

            # ACCUMULATE RESULTS
            for track in tracks:
                self.data_dict["frame_id"].append(i+1)
                self.data_dict["tracker_id"].append(track.track_id)
                self.data_dict["class_id"].append(track.class_ids)
                self.data_dict["xl"].append(track.tlwh[0])
                self.data_dict["yt"].append(track.tlwh[1])
                self.data_dict["w"].append(track.tlwh[2])
                self.data_dict["h"].append(track.tlwh[3])
                self.data_dict["conf"].append(track.score)
                self.data_dict["SS"].append(track.SS)
                self.data_dict["SR"].append(track.SR)
                self.data_dict["FA"].append(track.FA)
                self.data_dict["G"].append(track.G)

        self.save_results_to_txt()

    def get_detections(self, txt_path):
        """
        Read detections from a txt file and return a Detections object.
        """
        txt_data = np.loadtxt(txt_path)
        detections = sv.Detections(
            xyxy=txt_data[:, 1:5],
            class_id=txt_data[:, 0],
            confidence=txt_data[:, 5],
        )
        return detections

    def save_results_to_txt(self):
        mot_results = np.column_stack((
            np.array(self.data_dict["frame_id"]),
            np.array(self.data_dict["tracker_id"]),
            np.array(self.data_dict["xl"]),
            np.array(self.data_dict["yt"]),
            np.array(self.data_dict["w"]),
            np.array(self.data_dict["h"]),
            np.array(self.data_dict["conf"]),
            np.array(self.data_dict["class_id"]),
            np.ones_like(self.data_dict["frame_id"]),   # Visibility all to 1
            np.array(self.data_dict["SS"]).astype(int),
            np.array(self.data_dict["SR"]).astype(int),
            np.array(self.data_dict["FA"]).astype(int),
            np.array(self.data_dict["G"]).astype(int),

        ))

        with open(self.txt_path, 'w') as file:
            np.savetxt(file, mot_results, fmt='%.6f', delimiter=',')

    def get_sequence_info(self):
        seqinfo_file = os.path.join(self.dataset_sequence, 'seqinfo.ini')
        config = configparser.ConfigParser()
        config.read(seqinfo_file)
        width = int(config['Sequence']['imWidth'])
        height = int(config['Sequence']['imHeight'])
        fps = int(config['Sequence']['frameRate'])
        return sv.VideoInfo(width, height, fps)


def generate_tracks(config, experiment_id=None, print_bar=False):
    """
    Generate tracks for all sequences in the dataset using the specified tracker and reading detections from the source
    detections directory.
    Args:
        config: Configuration dictionary
        experiment_id: Experiment ID to append to the output directory name
        print_bar: Whether to print a progress bar
    Returns:
        processor: The last SequenceProcessor object used
    """
    sequences_dir = config["source_detections_dir"]
    sequence_paths = [os.path.join(sequences_dir, name) for name in os.listdir(sequences_dir) if
                      os.path.isdir(os.path.join(sequences_dir, name))]

    for sequence_path in sequence_paths:
        processor = SequenceProcessor(config, sequence_path, experiment_id)
        processor.process_sequence(print_bar)
    return processor


if __name__ == "__main__":
    with open("./cfg/recognize.json", "r") as f:
        config = json.load(f)
    generate_tracks(config, print_bar=True)
