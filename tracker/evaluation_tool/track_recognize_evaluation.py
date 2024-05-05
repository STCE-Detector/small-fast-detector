import sys
import argparse
import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from PySide6.QtCore import QObject
from PySide6.QtGui import QImage
from PySide6.QtCore import Signal
import configparser
import json
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

from tracker.utils.cfg.parse_config import ConfigParser
from ultralytics import YOLO

import tracker.trackers as trackers

COLORS = sv.ColorPalette.default()

ci_build_and_not_headless = False
try:
    from cv2.version import ci_build, headless
    ci_and_not_headless = ci_build and not headless
except:
    pass
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_FONTDIR")


class SequenceProcessor(QObject):
    frame_ready = Signal(QImage, float)

    def __init__(self, config, sequence_path):
        super().__init__()
        self.config = config
        self.sequence_path = sequence_path
        self.output_dir = str(config.save_dir/"data")
        os.makedirs(self.output_dir, exist_ok=True)
        sequence_name = os.path.basename(sequence_path)
        self.txt_path = self.output_dir + f"/{sequence_name}.txt"

        # Parameters of the YOLO detector
        self.conf_threshold = config["conf_threshold"]
        self.iou_threshold = config["iou_threshold"]
        self.img_size = config["img_size"]
        self.max_det = config["max_det"]
        self.agnostic_nms = config["agnostic_nms"]

        self.device = config["device"]

        # Loaded YOLO model
        self.model = YOLO(config["source_weights_path"])

        # Obtaining directory information from the sequence
        self.sequence_info = self.get_sequence_info()

        self.tracker = getattr(trackers, config["tracker_name"])(config["tracker_args"], self.sequence_info)
        self.frame_skip_interval = 100 / (100 - config["fps_reduction"])
        self.data_dict = {
            "frame_id": [],
            "tracker_id": [],
            "class_id": [],
            "xl": [],
            "yt": [],
            "w": [],
            "h": [],
            "conf": []
        }

    def process_sequence(self):
        path_images = os.path.join(self.sequence_path, "img1")
        for img_file in tqdm(sorted(os.listdir(path_images)), desc="Processing Frames", unit="frame"):
            img_path = os.path.join(path_images, img_file)
            frame = cv2.imread(img_path)
            frame_count = int(os.path.splitext(img_file)[0])

            if frame is None:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_count >= self.frame_skip_interval:
                self.process_frame(frame_rgb, frame_count)
                frame_count -= self.frame_skip_interval

        self.save_results_to_txt()

    def process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        results = self.model.predict(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            max_det=self.max_det,
            agnostic_nms=self.agnostic_nms,
            classes=[0]
        )[0]

        detections = sv.Detections.from_ultralytics(results)
        detections, tracks = self.tracker.update(detections, frame)

        for track in tracks:
            self.data_dict["frame_id"].append(frame_number)
            self.data_dict["tracker_id"].append(track.track_id)
            self.data_dict["class_id"].append(track.class_ids)
            self.data_dict["xl"].append(track.tlwh[0])
            self.data_dict["yt"].append(track.tlwh[1])
            self.data_dict["w"].append(track.tlwh[2])
            self.data_dict["h"].append(track.tlwh[3])
            self.data_dict["conf"].append(track.score)

    def save_results_to_txt(self):
        with open(self.txt_path, 'a') as file:
            for i in range(len(self.data_dict["frame_id"])):
                line = f"{self.data_dict['frame_id'][i]}, {self.data_dict['tracker_id'][i]}, {self.data_dict['xl'][i]}, {self.data_dict['yt'][i]}, {self.data_dict['w'][i]}, {self.data_dict['h'][i]}, {self.data_dict["conf"][i]}, -1, -1, -1\n"
                file.write(line)


    def get_sequence_info(self):
        seqinfo_file = os.path.join(self.sequence_path, 'seqinfo.ini')
        config = configparser.ConfigParser()
        config.read(seqinfo_file)
        width = int(config['Sequence']['imWidth'])
        height = int(config['Sequence']['imHeight'])
        fps = int(config['Sequence']['frameRate'])
        # Create an object similar as video_info
        sequence_info = sv.VideoInfo(width, height, fps)

        return sequence_info

def main(config):

        sequences_dir = config["source_sequence_dir"]
        sequence_paths = [os.path.join(sequences_dir, name) for name in os.listdir(sequences_dir) if
                          os.path.isdir(os.path.join(sequences_dir, name))]

        start_time = time.time()  # Start timing
        for sequence_path in sequence_paths:
            print(f"==> 🔄 Processing sequence: {sequence_path}")
            processor = SequenceProcessor(config, sequence_path)
            processor.process_sequence()

        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        print(f"🕒 Total processing time: {elapsed_time:.2f} seconds")
        print(f"🖥 Processing was done using: {config['device'].upper()}")



if __name__ == "__main__":
    print(os.listdir("../cfg"))
    parser = argparse.ArgumentParser(description="Process multiple video sequences.")
    parser.add_argument("-c", "--config", default="./cfg/ByteTrack_evalutaion.json", type=str,
                        help="Path to the main configuration file. Default is '../cfg/ByteTrack_evalutaion.json'")
    config = ConfigParser.from_args(parser)
    main(config)
