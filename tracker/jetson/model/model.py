import time
from typing import List, Union

import jetson_utils
import numpy as np
import torch

from tracker.jetson.model.ops import letterbox_pytorch, process_nms_trt_results, process_nms_onnx_results, letterbox
from ultralytics.engine.results import Results
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import ops
from ultralytics.utils.checks import check_imgsz


class Yolov8:
    def __init__(self, cfg, labels, device=torch.device('cuda', 0)):
        self.cfg = cfg
        self.half = True
        self.device = device
        self.labels = labels
        self.setup_model()

        # Initialize timing lists
        self.preprocess_times = []
        self.inference_times = []
        self.postprocess_times = []
        self.last_prediction_time = [0, 0, 0]  # Variable to save the last prediction time

    def setup_model(self, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        self.model = AutoBackend(
            weights=self.cfg["source_weights_path"],
            fp16=self.half,
            fuse=True,
            verbose=verbose,
        )

        # Backward compatibility
        self.output_names = sorted(self.model.output_names) if hasattr(self.model, "output_names") else None
        self.nms = self.model.nms if hasattr(self.model, "nms") else False
        self.engine = self.model.engine if hasattr(self.model, "engine") else False
        self.onnx = self.model.onnx if hasattr(self.model, "onnx") else False
        self.device = self.model.device  # update device
        self.half = self.model.fp16  # update half
        self.dtype = torch.float16 if self.half else torch.float32
        self.model.eval()
        imgsz = (640, 640)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_imgsz(imgsz, stride=stride)
        self.bs = 1

        self.model.warmup(imgsz=(1 if pt or self.model.triton else self.bs, 3, *imgsz))

    def _preprocess(self, im_orig: Union[np.ndarray, List[np.ndarray]]):
        start_time = time.time()
        img_tensor = torch.as_tensor(im_orig, device='cuda', dtype=torch.float32)
        same_shapes = len({im_orig.shape}) == 1
        img_tensor, scale_ratio, pad_size = letterbox_pytorch(img_tensor, new_shape=(640, 640), auto=same_shapes and self.model.pt, dtype=torch.float32)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor /= 255
        img_tensor = img_tensor.half() if self.half else img_tensor.float()
        self.preprocess_times.append(time.time() - start_time)
        return img_tensor, (scale_ratio, pad_size)

    def _preprocess_cpu(self, im_orig: Union[np.ndarray, List[np.ndarray]]):
        start_time = time.time()
        if isinstance(im_orig, jetson_utils.cudaImage):
            im_orig = jetson_utils.cudaToNumpy(im_orig)
        not_tensor = not isinstance(im_orig, torch.Tensor)
        if not_tensor:
            same_shapes = len({im_orig.shape}) == 1
            im, scale_ratio, pad_size = letterbox(im_orig, auto=same_shapes and self.model.pt, stride=self.model.stride)
            im = np.stack([im])
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.half else im.float()
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        self.preprocess_times.append(time.time() - start_time)
        return im, (scale_ratio, pad_size)

    def predict(self, im_orig: Union[np.ndarray, torch.Tensor]):
        start_time = time.time()
        img, padding = self._preprocess(im_orig)
        self.inference_times.append(time.time() - start_time)

        start_time = time.time()
        data = self.model(img)
        self.inference_times[-1] += time.time() - start_time

        start_time = time.time()
        results = self._postprocess(data, img, [im_orig], padding)
        self.postprocess_times.append(time.time() - start_time)
        self.last_prediction_time = [self.preprocess_times[-1], self.inference_times[-1], self.postprocess_times[-1]]  # Save last prediction time
        self.print_avg_times()
        return results

    def _postprocess(self, preds, img, orig_imgs, padding):
        if self.engine:
            preds = process_nms_trt_results(preds, self.output_names)
        elif self.onnx:
            preds = process_nms_onnx_results(preds)
        else:
            raise NotImplementedError("NMS end2end model is supported only in `engine` and `onnx` mode")

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape, ratio_pad=padding)
            results.append(Results(orig_img, path=None, names=self.labels, boxes=pred))
        return results

    def print_avg_times(self):
        self.preprocess_avg = sum(self.preprocess_times) / len(self.preprocess_times) * 1000
        self.inference_avg = sum(self.inference_times) / len(self.inference_times) * 1000
        self.postprocess_avg = sum(self.postprocess_times) / len(self.postprocess_times) * 1000
        self.total_avg_time = (self.preprocess_avg + self.inference_avg + self.postprocess_avg) / 1000  # Convert to seconds
        self.fps = 1 / self.total_avg_time if self.total_avg_time > 0 else float('inf')
        print(
            f"Preprocess: {self.preprocess_avg:.4f} ms, Inference: {self.inference_avg:.4f} ms, Postprocess: {self.postprocess_avg:.4f} ms, FPS: {self.fps:.2f}")
        print(f"Last Prediction Time (Preprocess, Inference, Postprocess): {self.last_prediction_time}")