
import argparse
import csv
import json
import os
import time

import pandas as pd
import torch
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import cv2
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext

import tracker.trackers as trackers
from tracker.action_recognition import ActionRecognizer
from tracker.utils.cfg.parse_config import ConfigParser
from tracker.utils.timer.utils import FrameRateCounter, Timer
from ultralytics import YOLO
import supervision as sv
from ultralytics.utils.torch_utils import model_info
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="`BoxAnnotator` is deprecated")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
COLORS = sv.ColorPalette.default()


def is_numeric(value):
    """ Check if the value can be converted to a float. """
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def compute_averages(all_arch_results):
    """ Compute averages for numeric keys and concatenate unique values for non-numeric keys """
    if not all_arch_results:
        return pd.DataFrame()

    data = {}
    numeric_keys = [key for key in all_arch_results[0] if
                    all(is_numeric(res.get(key, 'NaN')) for res in all_arch_results)]

    for key in numeric_keys:
        valid_values = [float(res[key]) for res in all_arch_results if
                        res.get(key) is not None and not np.isnan(float(res[key]))]
        if valid_values:
            data[key] = np.mean(valid_values)
        else:
            data[key] = np.nan  # Handle cases where no valid numeric data is present

    non_numeric_keys = set(all_arch_results[0]) - set(numeric_keys)
    for key in non_numeric_keys:
        unique_values = set(res[key] for res in all_arch_results if key in res)
        data[key] = ', '.join(unique_values)  # Join all unique non-numeric values into a single string

    return pd.DataFrame([data])

class VideoBenchmark:
    def __init__(self, config) -> None:

        # Initialize the YOLO parameters
        self.config = config
        self.conf_threshold = self.config["conf_threshold"]
        self.iou_threshold = self.config["iou_threshold"]
        self.img_size = self.config["img_size"]
        self.max_det = self.config["max_det"]
        self.agnostic_nms = config["agnostic_nms"]

        self.output_dir = self.config.save_dir
        self.target_video_path = str(self.output_dir / "annotated_video.mp4")

        self.device = self.config["device"]
        self.wait_time = 1
        self.slow_factor = 1
        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.display = self.config["display"]
        self.save_video = self.config["save_video"]
        if self.save_video and self.display:
            raise ValueError("Cannot display and save video at the same time")

        self.trace_annotator = sv.TraceAnnotator(color=COLORS, position=sv.Position.CENTER, trace_length=100,
                                                 thickness=2)
        self.save_results = self.config["save_results"]
        self.csv_path = str(self.output_dir) + "/track_data.csv"

        # change to function
        self.model = None
        self.source_video_path = None
        self.video_info = None
        self.tracker = None
        self.action_recognizer = None
        self.model_times = []
        self.preprocessing = []
        self.post_processing_times = []
        self.post_processing_times_total = []
        self.tracker_times = []
        self.action_recognition_times = []
        self.annotated_frame_times = []
        self.loaded_video_times = []
        self.write_video_time_list = []
        self.timer_load_frame_list = []
        self.video_fps = None
        self.time_taken = None

        self.class_names = {
            0: "person",
            1: "car",
            2: "truck",
            3: "uav",
            4: "airplane",
            5: "boat",
        }

    def load_model(self, model_path, model_format):
        def check_file_exists(path, arch):
            """
            Purpose: To verify that a specified configuration file for the model exists at a given path. This is a utility function to ensure necessary files are present before attempting to load or benchmark a model.
            How It Works: It constructs the full path to the expected file and checks if the file exists there, raising an error if not. This helps prevent runtime errors due to missing files.
            """
            full_path = os.path.join(path, arch) + '.pt'
            print(full_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"El archivo {full_path} no existe.")
            else:
                print(f"El archivo {full_path} existe.")
            return full_path

        full_path = check_file_exists(model_path, model_format)
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        self.model = YOLO(full_path, task='detect')
        self.model.to(self.device)

    def export_model(self, arch, config_export):
        export_filename = arch
        try:
            args_str = json.dumps(config_export['args'], sort_keys=True)
            export_filename = f"{arch}.{config_export['format']}"
            export_path = f'./detectors/{export_filename}'
            args_dict = json.loads(args_str)
            unique_id = '_'.join(f"{key}_{value}" for key, value in args_dict.items())
            export_filename = f"{arch}_{config_export['format']}_{unique_id}"
            if config_export['format'] == 'pytorch':
                pass
            else:
                self.model.export(format=config_export['format'], device=self.device, **config_export['args'], project='./detectors/')
                print(f"Modelo {arch} exportado como {export_filename} a {export_path}")
                model_path = export_path if os.path.exists(export_path) else f"{export_path}.{config_export['format']}"
                self.model = YOLO(model_path, task='detect')
        except Exception as e:
            print(f"Error exporting model {arch} to format {config_export['format']}: {e}")
            pass
        return export_filename

    # get info video
    def initialize_video(self, video, video_path):
        def check_file_exists(path, arch):
            """
            Purpose: To verify that a specified configuration file for the model exists at a given path. This is a utility function to ensure necessary files are present before attempting to load or benchmark a model.
            How It Works: It constructs the full path to the expected file and checks if the file exists there, raising an error if not. This helps prevent runtime errors due to missing files.
            """
            full_path = os.path.join(path, arch) + '.mp4'
            print(full_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"El video {full_path} no existe.")
            else:
                print(f"El video {full_path} existe.")
            return full_path

        self.source_video_path = check_file_exists(video_path, video)
        self.video_info = sv.VideoInfo.from_video_path(self.source_video_path)
        self.tracker = getattr(trackers, config["tracker_name"])(config["tracker_args"], self.video_info)
        self.action_recognizer = ActionRecognizer(self.config["action_recognition"], self.video_info)

    def reset_times(self):
        self.model_times = []
        self.preprocessing = []
        self.post_processing_times = []
        self.post_processing_times_total = []
        self.tracker_times = []
        self.action_recognition_times = []
        self.annotated_frame_times = []
        self.loaded_video_times = []
        self.write_video_time_list = []
        self.timer_load_frame_list = []
        self.video_fps = None
        self.time_taken = None

    def run_benchmark(self, archs, videos, export_configs, path_model='./detectors/', path_videos='./videos/'):
        results = []  # List to store final averaged results

        for arch in archs:
            for config_export in export_configs:
                self.load_model(path_model, arch)
                _, n_p, _, flops = model_info(self.model.model)  # Get model info
                export_filename = self.export_model(arch, config_export)

                all_video_results = []  # Store results for each video for the current configuration
                for video in videos:
                    self.initialize_video(video, path_videos)
                    self.process_video(config_export)  # Process each video and gather data

                    video_results = {
                        'latency_tracker_ms': self.tracker_times,
                        'latency_total_ms': self.post_processing_times_total,
                        'latency_preprocess_ms': self.preprocessing,
                        'post_processing_ms': self.post_processing_times,
                        'inference_ms': self.model_times,
                        'action_recognition_ms': self.action_recognition_times,
                        'annotated_frame_times': self.annotated_frame_times,
                        'loaded_video_times': self.loaded_video_times,
                        'write_video_time_list': self.write_video_time_list,
                        'timer_load_frame_list': self.timer_load_frame_list,
                        'FPS_model': [np.mean([1 / (x / 1000) if x != 0 else 1 for x in self.model_times])],
                        'FPS_video': [float(self.video_fps)],
                        'time_taken_seconds': [float(self.time_taken)],
                    }
                    all_video_results.append(video_results)
                    print(video_results)
                    self.reset_times()

                # Compute averages of all metrics across videos for the current model configuration
                averaged_results = {
                    'model_name': export_filename,
                    'parameters_count': "{:,.0f}".format(n_p),
                    'GFLOPs': "{:,.2f}".format(flops)
                }

                # Add the averages of all metrics across videos for the current model configuration
                averaged_results.update({
                    key: np.nanmean(
                        [np.nanmean([x for x in v[key] if isinstance(x, (int, float))]) for v in all_video_results if
                         v[key]])
                    for key in all_video_results[0] if
                    any(isinstance(item, (int, float)) for sublist in all_video_results for item in sublist[key])
                })

                results.append(averaged_results)  # Append the averaged results for the current model configuration

        df = pd.DataFrame(results)
        df.to_csv("./benchmark_tracker.csv", index=False)

    def process_video(self, config_export):
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
            "write_video_time": []
        }
        timer_load_frame_end = 0
        timer_load_frame_start = 0
        video_sink = sv.VideoSink(self.target_video_path, self.video_info) if self.save_video else nullcontext()
        with video_sink as sink:
            fps_counter = FrameRateCounter()
            timer = Timer()
            for i, frame in enumerate(pbar := tqdm(frame_generator, total=self.video_info.total_frames)):
                if i != 0:
                    timer_load_frame_end = time.perf_counter()
                timer_load_frame = timer_load_frame_end - timer_load_frame_start
                self.timer_load_frame_list.append(timer_load_frame)
                pbar.set_description(f"[FPS: {fps_counter.value():.2f}] ")

                annotated_frame = self.process_frame(frame, i, fps_counter.value(), config_export)
                fps_counter.step() # here

                if self.save_video:
                    start_time_write_video = time.perf_counter()
                    sink.write_frame(annotated_frame)
                    write_video_time = time.perf_counter() - start_time_write_video
                    self.write_video_time_list.append(write_video_time)
                elif self.display:
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
                else:
                    pass

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
                        # data_dict["write_video_time"].append(write_video_time)
                timer_load_frame_start = time.perf_counter()
            if self.display:
                cv2.destroyAllWindows()

        # Print time and fps
        time_taken = f"{int(timer.elapsed() / 60)} min {int(timer.elapsed() % 60)} sec"
        avg_fps = self.video_info.total_frames / timer.elapsed()
        print(f"\nTracking complete over {self.video_info.total_frames} frames.")
        print(f"Total time: {time_taken}")
        print(f"Average FPS: {avg_fps:.2f}")
        self.video_fps = avg_fps
        self.time_taken = timer.elapsed()

        # Save datadict in csv
        """if self.save_results:
            with open(self.csv_path, "w") as f:
                w = csv.writer(f)
                w.writerow(data_dict.keys())
                w.writerows(zip(*data_dict.values()))"""

    def process_frame(self, frame: np.ndarray, frame_number: int, fps: float, config_export) -> np.ndarray:
        results = self.model.predict(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=config_export.get('args', {}).get('imgsz', False),
            device=self.device,
            max_det=self.max_det,
            half=config_export.get('args', {}).get('half', False),
            int8=config_export.get('args', {}).get('int8', False)
        )[0]
        # MODEL INFERENCE TIME
        model_speed_preprocess = results.speed['preprocess']
        model_speed_inference = results.speed['inference']
        model_speed_postprocess = results.speed['postprocess']
        start_time_post_processing = time.perf_counter()
        detections = sv.Detections.from_ultralytics(results)
        postprocessing_time = time.perf_counter() - start_time_post_processing
        model_speed_postprocess_total = model_speed_preprocess + model_speed_inference + model_speed_postprocess + postprocessing_time

        # TRACKER TIME HERE
        start_time_tracker = time.perf_counter()
        detections, tracks = self.tracker.update(detections, frame)
        tracker_update_time = time.perf_counter() - start_time_tracker

        # ACTION RECOGNITION TIME HERE
        start_time_action_recognition = time.perf_counter()
        ar_results = self.action_recognizer.recognize_frame(tracks)
        action_recognition_time = time.perf_counter() - start_time_action_recognition
        start_time_annotated_frame = time.perf_counter()
        annotated_frame = self.annotate_frame(frame, detections, ar_results,  frame_number, fps)
        annotated_frame_time = time.perf_counter() - start_time_annotated_frame

        self.preprocessing.append(model_speed_preprocess)
        self.model_times.append(model_speed_inference)
        self.post_processing_times.append(model_speed_postprocess)
        self.post_processing_times_total.append(model_speed_postprocess_total)
        self.tracker_times.append(tracker_update_time)
        self.action_recognition_times.append(action_recognition_time)
        self.annotated_frame_times.append(annotated_frame_time)

        return annotated_frame

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
        default="./cfg/ByteTrack.json",
        type=str,
        help="config file path (default: None)",
    )
    # path_models = '../ultralytics/cfg/models/v8/'
    path_models = './detectors/'
    # get all files in the path
    model_names = [f.split('.')[0] for f in os.listdir(path_models) if os.path.isfile(os.path.join(path_models, f))]
    model_names = sorted(model_names)

    path_videos = './videos/'
    videos = [f.split('.')[0] for f in os.listdir(path_videos) if os.path.isfile(os.path.join(path_videos, f))]

    config = ConfigParser.from_args(parser)
    export_configs = [
        {'format': 'pytorch', 'args': {'imgsz': 640, 'half': False}},
        {'format': 'pytorch', 'args': {'imgsz': 1280, 'half': False}},
        {'format': 'pytorch', 'args': {'imgsz': 640, 'half': True}},
        {'format': 'pytorch', 'args': {'imgsz': 1280, 'half': True}},
        {'format': 'torchscript', 'args': {'imgsz': 640, 'optimize': False}},
        {'format': 'torchscript', 'args': {'imgsz': 1280, 'optimize': False}},
        {'format': 'onnx', 'args': {'imgsz': 640, 'half': False, 'dynamic': False, 'int8': False, 'simplify': False}},
        {'format': 'onnx', 'args': {'imgsz': 1280, 'half': False, 'dynamic': False, 'int8': False, 'simplify': False}},
        {'format': 'onnx', 'args': {'imgsz': 640, 'half': False, 'dynamic': False, 'int8': False, 'simplify': True}},
        {'format': 'onnx', 'args': {'imgsz': 1280, 'half': False, 'dynamic': False, 'int8': False, 'simplify': True}},
        {'format': 'onnx', 'args': {'imgsz': 640, 'half': True, 'dynamic': False, 'int8': False, 'simplify': True}},
        {'format': 'onnx', 'args': {'imgsz': 1280, 'half': True, 'dynamic': False, 'int8': False, 'simplify': True}},
        {'format': 'engine',
         'args': {'imgsz': 640, 'half': False, 'dynamic': False, 'int8': False, 'simplify': False}},
        {'format': 'engine',
         'args': {'imgsz': 1280, 'half': False, 'dynamic': False, 'int8': False, 'simplify': False}},
        {'format': 'engine',
         'args': {'imgsz': 640, 'half': False, 'dynamic': False, 'int8': False, 'simplify': True}},
        {'format': 'engine',
         'args': {'imgsz': 1280, 'half': False, 'dynamic': False, 'int8': False, 'simplify': True}},
        {'format': 'engine',
         'args': {'imgsz': 640, 'half': True, 'dynamic': False, 'int8': False, 'simplify': True}},
        {'format': 'engine',
         'args': {'imgsz': 1280, 'half': True, 'dynamic': False, 'int8': False, 'simplify': True}},
        {'format': 'engine',
         'args': {'imgsz': 640, 'half': False, 'dynamic': True, 'int8': True, 'simplify': True}},
        {'format': 'engine',
         'args': {'imgsz': 1280, 'half': False, 'dynamic': True, 'int8': True, 'simplify': True}},
    ]
    benchmark = VideoBenchmark(config)
    benchmark.run_benchmark(model_names, videos, export_configs)
