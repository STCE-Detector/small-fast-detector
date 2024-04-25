import sys
import os
import argparse
import csv
from asyncio import Queue
from threading import Thread
from PIL.ImageQt import QImage
import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

import supervision as sv
from supervision import Detections
from tracker import ByteTrack
from tracker.action_recognition import ActionRecognizer
from tracker.gui.GUI import VideoDisplay
from tracker.gui.frameCapture import FrameCapture
from tracker.gui.frameProcessing import VideoWriter
from tracker.track_recognize import VideoProcessor
from tracker.utils.cfg.parse_config import ConfigParser
from tracker.utils.timer.utils import FrameRateCounter, Timer
from ultralytics import YOLO
from tqdm import tqdm

# Set high DPI scaling for PyQt
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

COLORS = sv.ColorPalette.default()

class VideoProcessor(QObject):
    frame_ready = pyqtSignal(QImage, float)

    def __init__(self, config) -> None:
        super(VideoProcessor, self).__init__()
        self.config = config
        self.conf_threshold = config["conf_threshold"]
        self.iou_threshold = config["iou_threshold"]
        self.img_size = config["img_size"]
        self.max_det = config["max_det"]
        self.source_video_path = config["source_stream_path"]
        self.output_dir = config.save_dir
        self.target_video_path = str(self.output_dir / "annotated_video.mp4")
        self.device = config["device"]
        self.video_stride = config["video_stride"]
        self.model = YOLO(config["source_weights_path"])
        self.model.fuse()
        self.model.to(config["device"])
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.detector_thread = Thread(target=self.run_detector)
        self.paused = False
        self.video_info = sv.VideoInfo.from_video_path(self.source_video_path)
        self.tracker = ByteTrack(config, frame_rate=self.video_info.fps)
        self.box_annotator = sv.BoundingBoxAnnotator(color=COLORS)
        self.trace_annotator = sv.TraceAnnotator(color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2)
        self.frame_capture = FrameCapture(self.source_video_path, stabilize=config["stabilize"],
                                          stream_mode=config["stream_mode"], logging=config["logging"])
        self.display = config["display"]
        self.save_video = config["save_video"]
        self.save_results = config["save_results"]
        self.csv_path = str(self.output_dir) + "/track_data.csv"
        if self.save_video and self.display:
            raise ValueError("Cannot display and save video at the same time")
        self.action_recognizer = ActionRecognizer(config["action_recognition"], self.video_info)
        self.class_names = {
            0: "person",
            1: "car",
            2: "truck",
            3: "uav",
            4: "airplane",
            5: "boat",
        }
        if self.save_video:
            self.video_writer = VideoWriter(self.target_video_path, frame_size=self.frame_capture.get_frame_size(),
                                            compression_mode=self.config["compression_mode"],
                                            logging=self.config["logging"],
                                            fps=self.frame_capture.get_fps())
            self.video_writer.start()
    def run_detector(self):
        while True:
            frame = self.input_queue.get()
            if frame is None:  # Use None as a signal to stop the thread
                break
            results = self.model.predict(frame, conf=self.conf_threshold, iou=self.iou_threshold,
                                         imgsz=self.img_size, device=self.device, max_det=self.max_det)[0]
            detections = sv.Detections.from_ultralytics(results)
            self.output_queue.put((frame, detections))

    def process_video(self):
        print(f"Processing video: {self.source_video_path} ...")
        fps_counter = FrameRateCounter()
        timer = Timer()
        self.frame_capture.start()
        pbar = tqdm(total=self.video_info.total_frames, desc="Processing Frames", unit="frame")

        data_dict = {
            "frame_id": [], "tracker_id": [], "class_id": [], "x1": [], "y1": [], "x2": [], "y2": []
        } if self.save_results else None

        frame = self.frame_capture.read()
        while frame is not None:
            if not self.paused:
                frame = self.frame_capture.read()
                if frame is None:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.input_queue.put(frame_rgb)
                if not self.output_queue.empty():
                    frame, detections = self.output_queue.get()
                    annotated_frame = self.process_frame(frame, detections, self.frame_capture.get_frame_count(),
                                                         fps_counter.value())
                    fps_counter.step()
                    self.handle_output(annotated_frame, fps_counter.value(), data_dict)
                pbar.update(1)
            else:
                pbar.update()

        self.cleanup(data_dict)
        pbar.close()
        print(f"\nTracking complete over {self.video_info.total_frames} frames.")
        print(f"Total time: {timer.elapsed():.2f} seconds")
        average_fps = self.video_info.total_frames / timer.elapsed()
        print(f"Average FPS: {average_fps:.2f}")

    def handle_output(self, annotated_frame, fps, data_dict):
        if self.save_video and not self.display:
            self.video_writer.write_frame(annotated_frame)
        if self.display:
            height, width, channel = annotated_frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            self.frame_ready.emit(q_image, fps)
        if self.save_results:
            for track in self.tracker.tracked_stracks:
                data_dict["frame_id"].append(self.frame_capture.get_frame_count())
                data_dict["tracker_id"].append(track.track_id)
                data_dict["class_id"].append(track.class_id)
                data_dict["x1"].append(track.tlbr[0])
                data_dict["y1"].append(track.tlbr[1])
                data_dict["x2"].append(track.tlbr[2])
                data_dict["y2"].append(track.tlbr[3])

    def process_frame(self, frame: np.ndarray, detections: Detections, frame_number: int, fps: float) -> np.ndarray:
        detections, tracks = self.tracker.update(detections, frame)
        ar_results = self.action_recognizer.recognize_frame(tracks)
        return self.annotate_frame(frame, detections, ar_results, frame_number, fps)

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections, ar_results, frame_number: int,
                       fps: float) -> np.ndarray:
        labels = [f"#{track.track_id} {self.class_names[track.class_id]} {track.confidence:.2f}"
                  for track in detections.tracked_stracks]
        annotated_frame = self.trace_annotator.annotate(frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections, labels)
        annotated_frame = self.action_recognizer.annotate(annotated_frame, ar_results)
        cv2.putText(annotated_frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return annotated_frame

    def cleanup(self, data_dict=None):
        self.frame_capture.stop()
        if self.save_video:
            self.video_writer.stop()
        if self.save_results:
            with open(self.csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data_dict.keys())
                writer.writerows(zip(*data_dict.values()))

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="YOLO and ByteTrack")
        parser.add_argument("-c", "--config", default="./ByteTrack.json", type=str,
                            help="config file path (default: None)")
        config = ConfigParser.from_args(parser)

        if config["display"]:
            app = QApplication(sys.argv)
            video_display = VideoDisplay(processor=VideoProcessor(config))
            video_display.show()
            sys.exit(app.exec())
        else:
            video_processor = VideoProcessor(config)
            video_processor.process_video()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO and ByteTrack")
    parser.add_argument("-c", "--config", default="./ByteTrack.json", type=str, help="config file path (default: None)")
    config = ConfigParser.from_args(parser)

    if config["display"]:
        app = QApplication(sys.argv)
        video_display = VideoDisplay(processor=VideoProcessor(config))
        video_display.show()
        sys.exit(app.exec())
    else:
        video_processor = VideoProcessor(config)
        video_processor.process_video()
