import argparse
import csv
import os

import onnxruntime
import torch

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext

from tracker.byte_track import ByteTrack
from tracker.action_recognition import ActionRecognizer
from tracker.utils.cfg.parse_config import ConfigParser
from tracker.utils.timer.utils import FrameRateCounter, Timer
import supervision as sv

COLORS = sv.ColorPalette.default()

try:
    from ultralytics import YOLO
except:
    print("Ultralytics not installed. Please install it using 'pip install ultralytics'")


class VideoProcessor:
    def __init__(self, config) -> None:

        # Initialize the YOLO parameters
        self.conf_threshold = config["conf_threshold"]
        self.iou_threshold = config["iou_threshold"]
        self.img_size = config["img_size"]
        self.max_det = config["max_det"]

        self.source_video_path = config["source_video_path"]

        self.output_dir = config.save_dir
        self.target_video_path = str(self.output_dir / "annotated_video.mp4")

        self.device = config["device"]
        self.video_stride = config["video_stride"]
        self.wait_time = 1
        self.slow_factor = 1
        try:
            self.model = YOLO(config["source_weights_path"])
            self.model.fuse()
        except:
            cuda = torch.cuda.is_available()
            # check if .onnx file
            if config["source_weights_path"].endswith(".onnx"):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
                self.model = onnxruntime.InferenceSession(config["source_weights_path"], providers=providers)
                output_names = [x.name for x in self.model.get_outputs()]
                metadata = self.model.get_modelmeta().custom_metadata_map

        self.video_info = sv.VideoInfo.from_video_path(self.source_video_path)

        self.tracker = ByteTrack(config, frame_rate=self.video_info.fps)

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.trace_annotator = sv.TraceAnnotator(color=COLORS, position=sv.Position.CENTER, trace_length=100,
                                                 thickness=2)

        self.display = config["display"]
        self.save_results = config["save_results"]
        self.csv_path = str(self.output_dir) + "/track_data.csv"

        self.class_names = {
            0: "person",
            1: "car",
            2: "truck",
            3: "uav",
            4: "airplane",
            5: "boat",
        }

        self.action_recognizer = ActionRecognizer(config["action_recognition"], self.video_info)

    def preprocess(self, img):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """

        # Get the height and width of the input image
        self.img_height, self.img_width = img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (640, 640))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, input_image, output):
        # Assuming outputs are already in the shape you've processed before
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]

        # Preparing lists for Detections data
        xyxy = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.conf_threshold:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Scale the bounding box back from the model's input size to the original image size
                x_factor = input_image.shape[1] / 640
                y_factor = input_image.shape[0] / 640
                left = x - w / 2
                top = y - h / 2
                right = x + w / 2
                bottom = y + h / 2

                # Adjust coordinates to match the original image size
                left *= x_factor
                top *= y_factor
                right *= x_factor
                bottom *= y_factor

                # Appending data for sv.Detections
                xyxy.append([left, top, right, bottom])
                scores.append(max_score)
                class_ids.append(class_id)

        # Convert lists to numpy arrays
        xyxy = np.array(xyxy)
        scores = np.array(scores)
        class_ids = np.array(class_ids)

        # Create sv.Detections instance
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=scores,
            class_id=class_ids
        )

        return detections

    def process_video(self):
        print(f"Processing video: {os.path.basename(self.source_video_path)} ...")
        print(f"Original video size: {self.video_info.resolution_wh}")
        print(f"Original video FPS: {self.video_info.fps}")
        print(f"Original video number of frames: {self.video_info.total_frames}\n")

        frame_generator = sv.get_video_frames_generator(source_path=self.source_video_path)

        data_dict = {
            "frame_id": [],
            "tracker_id": [],
            "class_id": [],
            "x1": [],
            "y1": [],
            "x2": [],
            "y2": [],
        }

        video_sink = sv.VideoSink(self.target_video_path, self.video_info) if not self.display else nullcontext()
        with video_sink as sink:
            fps_counter = FrameRateCounter()
            timer = Timer()

            for i, frame in enumerate(pbar := tqdm(frame_generator, total=self.video_info.total_frames)):
                pbar.set_description(f"[FPS: {fps_counter.value():.2f}] ")
                if i % self.video_stride == 0:
                    annotated_frame = self.process_frame(frame, i, fps_counter.value())
                    fps_counter.step()  # here

                    if not self.display:
                        sink.write_frame(annotated_frame)
                    else:
                        cv2.imshow("Processed Video", annotated_frame)

                        k = cv2.waitKey(int(self.wait_time * self.slow_factor))  # dd& 0xFF

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                        elif k == ord('p'):  # pause the video
                            cv2.waitKey(-1)  # wait until any key is pressed
                        elif k == ord('r'):  # resume the video
                            continue
                        elif k == ord('d'):
                            slow_factor = self.slow_factor - 1
                            print(slow_factor)
                        elif k == ord('i'):
                            slow_factor = self.slow_factor + 1
                            print(slow_factor)
                            break

                    # Store results
                    if self.save_results:
                        for track in self.tracker.tracked_stracks:
                            data_dict["frame_id"].append(track.frame_id)
                            data_dict["tracker_id"].append(track.track_id)
                            data_dict["class_id"].append(track.class_ids)
                            data_dict["x1"].append(track.tlbr[0])
                            data_dict["y1"].append(track.tlbr[1])
                            data_dict["x2"].append(track.tlbr[2])
                            data_dict["y2"].append(track.tlbr[3])

            if self.display:
                cv2.destroyAllWindows()

        # Print time and fps
        time_taken = f"{int(timer.elapsed() / 60)} min {int(timer.elapsed() % 60)} sec"
        avg_fps = self.video_info.total_frames / timer.elapsed()
        print(f"\nTracking complete over {self.video_info.total_frames} frames.")
        print(f"Total time: {time_taken}")
        print(f"Average FPS: {avg_fps:.2f}")

        # Save datadict in csv
        if self.save_results:
            with open(self.csv_path, "w") as f:
                w = csv.writer(f)
                w.writerow(data_dict.keys())
                w.writerows(zip(*data_dict.values()))

    def process_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        try:
            results = self.model(
                frame,
                verbose=False,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device,
                max_det=self.max_det
            )[0]
            detections = sv.Detections.from_ultralytics(results)
        except:
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            # ADAPT TO ONNX
            processed_frame = self.preprocess(frame)
            """resized_frame = cv2.resize(frame, expected_dim, interpolation=cv2.INTER_LINEAR)
            processed_frame = resized_frame.transpose(2, 0, 1)
            processed_frame = processed_frame[np.newaxis, :, :, :]"""
            results = self.model.run([output_name], {input_name: processed_frame.astype(np.float32)})

            detections = self.postprocess(frame, results)
        detections, tracks = self.tracker.update(detections, frame)

        ar_results = self.action_recognizer.recognize_frame(tracks)

        return self.annotate_frame(frame, detections, ar_results, frame_number, fps)

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections, ar_results: None, frame_number: int,
                       fps: float) -> np.ndarray:
        annotated_frame = frame.copy()

        labels = [f"#{tracker_id} {self.class_names[class_id]} {confidence:.2f}"
                  for tracker_id, class_id, confidence in
                  zip(detections.tracker_id, detections.class_id, detections.confidence)]
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections, labels)
        annotated_frame = self.action_recognizer.annotate(annotated_frame, ar_results)
        cv2.putText(annotated_frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return annotated_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO and ByteTrack")
    parser.add_argument(
        "-c",
        "--config",
        default="./ByteTrack.json",
        type=str,
        help="config file path (default: None)",
    )
    config = ConfigParser.from_args(parser)
    processor = VideoProcessor(config)
    processor.process_video()
