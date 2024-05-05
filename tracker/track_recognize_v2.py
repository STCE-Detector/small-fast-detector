import sys
import argparse
import csv
import os

from PySide6.QtCore import QObject
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Signal

from ultralytics.utils import IS_JETSON

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from tracker.action_recognition import ActionRecognizer
from tracker.gui.GUI import VideoDisplay
from tracker.gui.frameCapture import FrameCapture
from tracker.gui.frameProcessing import VideoWriter
from tracker.utils.cfg.parse_config import ConfigParser
from tracker.utils.timer.utils import FrameRateCounter, Timer
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

if IS_JETSON:
    from jetson_utils import videoSource, cudaToNumpy, cudaAllocMapped, cudaConvertColor, cudaDeviceSynchronize


class VideoProcessor(QObject):
    frame_ready = Signal(QImage, float)

    def __init__(self, config) -> None:
        super(VideoProcessor, self).__init__()

        # Read the YOLO detector parameters
        self.conf_threshold = config["conf_threshold"]
        self.iou_threshold = config["iou_threshold"]
        self.img_size = config["img_size"]
        self.max_det = config["max_det"]
        self.agnostic_nms = config["agnostic_nms"]

        self.source_video_path = config["source_stream_path"]
        self.output_dir = config.save_dir
        self.target_video_path = str(self.output_dir / "annotated_video.mp4")

        self.device = config["device"]

        # Load the YOLO model
        self.model = YOLO(config["source_weights_path"])

        # TODO: CHECK IF MAINTAIN THIS
        self.video_info = sv.VideoInfo.from_video_path(self.source_video_path)

        # TODO : CHECK TO PUT IN A THREAD
        self.tracker = getattr(trackers, config["tracker_name"])(config["tracker_args"], self.video_info)

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.trace_annotator = sv.TraceAnnotator(color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2)

        if not IS_JETSON:
            self.frame_capture = FrameCapture(self.source_video_path, stabilize=config["stabilize"],
                                              stream_mode=config["stream_mode"], logging=config["logging"])
            self.frame_capture.start()
        else:
            try:
                # from tracker.jetson.video import VideoSource
                self.frame_capture = videoSource(self.source_video_path)
                self.frame_capture.Open()
            except Exception as e:
                print(f"Failed to open video source: {e}")
                sys.exit(1)
        self.paused = False

        self.frame_skip_interval = 100/(100-config["fps_reduction"])

        self.display = config["display"]
        self.save_video = config["save_video"]
        self.save_results = config["save_results"]
        if self.save_video and self.display:
            raise ValueError("Cannot display and save video at the same time")
        if self.save_results:
            self.csv_path = str(self.output_dir) + "/track_data.csv"
            self.data_dict = {
                "frame_id": [],
                "tracker_id": [],
                "class_id": [],
                "x1": [],
                "y1": [],
                "x2": [],
                "y2": []
            }
        if self.save_video:
            self.video_writer = VideoWriter(self.target_video_path, frame_size=self.frame_capture.get_frame_size(),
                                            compression_mode=config["compression_mode"],
                                            logging=config["logging"],
                                            fps=self.frame_capture.get_fps())
            self.video_writer.start()

        self.class_names = {
            0: "person",
            1: "car",
            2: "truck",
            3: "uav",
            4: "airplane",
            5: "boat",
        }

        self.action_recognizer = ActionRecognizer(config["action_recognition"], self.video_info)

    def process_video(self):
        print(f"Processing video: {self.source_video_path} ...")
        print(f"Original video size: {self.video_info.resolution_wh}")
        print(f"Original video FPS: {self.video_info.fps}")
        print(f"Original video number of frames: {self.video_info.total_frames}\n")

        pbar = tqdm(total=self.video_info.total_frames, desc="Processing Frames", unit="frame")
        fps_counter = FrameRateCounter()
        timer = Timer()
        frame_count = 0
        while True:
            if not self.paused:
                try:
                    rgb_img = self.frame_capture.Capture()
                except:
                    continue
                frame_count += 1
                if rgb_img is None:
                    continue
                if IS_JETSON:
                    bgr_img = cudaAllocMapped(width=rgb_img.width,
                                              height=rgb_img.height,
                                              format='bgr8')

                    cudaConvertColor(rgb_img, bgr_img)
                    # make sure the GPU is done work before we convert to cv2
                    cudaDeviceSynchronize()
                    # convert to cv2 image (cv2 images are numpy arrays)
                    frame = cudaToNumpy(bgr_img)
                else:
                    frame = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                if frame_count >= self.frame_skip_interval:
                    annotated_frame = self.process_frame(frame, self.frame_capture.GetFrameCount(), fps_counter.value())
                    fps_counter.step()
                    frame_count -= self.frame_skip_interval

                    if self.save_video and not self.display:
                        self.video_writer.write_frame(annotated_frame)

                    if self.display:
                        height, width, channel = annotated_frame.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                        self.frame_ready.emit(q_image, fps_counter.value())

                    if self.save_results:
                        for track in self.tracker.active_tracks:
                            self.data_dict["frame_id"].append(self.frame_capture.get_frame_count())
                            self.data_dict["tracker_id"].append(track.track_id)
                            self.data_dict["class_id"].append(track.class_id)
                            self.data_dict["x1"].append(track.tlbr[0])
                            self.data_dict["y1"].append(track.tlbr[1])
                            self.data_dict["x2"].append(track.tlbr[2])
                            self.data_dict["y2"].append(track.tlbr[3])
                else:
                    # TODO: when static skipping is > 0, video not generated, solve this (skipping should start by true)
                    if self.save_video and not self.display:
                        #self.video_writer.write_frame(annotated_frame)
                        px=0
                    fps_counter.step()

                pbar.update(1)
                if not self.frame_capture.IsStreaming():
                    break

            if self.save_results:
                with open(self.csv_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(self.data_dict.keys())
                    writer.writerows(zip(*self.data_dict.values()))

        pbar.close()
        print(f"\nTracking complete over {self.video_info.total_frames} frames.")
        print(f"Total time: {timer.elapsed():.2f} seconds")
        avg_fps = self.video_info.total_frames / timer.elapsed()
        print(f"Average FPS: {avg_fps:.2f}")
        self.cleanup()

    def process_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        results = self.model.predict(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            max_det=self.max_det,
            agnostic_nms=self.agnostic_nms,
        )[0]
        # TODO: compare the results with the results from the ByteTrack tracker, losing detections
        detections = sv.Detections.from_ultralytics(results)
        detections, tracks = self.tracker.update(detections, frame)

        ar_results = self.action_recognizer.recognize_frame(tracks)
        return self.annotate_frame(frame, detections, ar_results, frame_number, fps)

    def annotate_frame(self, annotated_frame: np.ndarray, detections: sv.Detections, ar_results: None,
                       frame_number: int, fps: float) -> np.ndarray:

        labels = [f"#{tracker_id} {self.class_names[class_id]} {confidence:.2f}"
                  for tracker_id, class_id, confidence in
                  zip(detections.tracker_id, detections.class_id, detections.confidence)]
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections, labels)
        annotated_frame = self.action_recognizer.annotate(annotated_frame, ar_results)
        if self.save_video:
            cv2.putText(annotated_frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return annotated_frame

    def toggle_pause(self):
        self.paused = not self.paused

    def cleanup(self):
        print("Cleaning up...")
        if not IS_JETSON:
            self.frame_capture.stop()
        else:
            try:
                from tracker.jetson.video import VideoSource
                self.frame_capture = VideoSource(self.source_video_path)
            except Exception as e:
                print(f"Failed to open video source: {e}")
                sys.exit(1)

        if self.save_video:
            self.video_writer.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO and ByteTrack")
    parser.add_argument(
        "-c",
        "--config",
        default="./cfg/ByteTrack.json",
        type=str,
        help="config file path (default: None)",
    )
    config = ConfigParser.from_args(parser)
    if config["display"]:
        app = QApplication(sys.argv)
        video_display = VideoDisplay(processor=VideoProcessor(config), sync_fps=config["sync_fps"])
        video_display.show()
        sys.exit(app.exec())
    else:
        video_processor = VideoProcessor(config)
        video_processor.process_video()
