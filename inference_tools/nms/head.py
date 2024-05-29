from typing import Tuple

import torch
from torch import Tensor, Value, nn
from ultralytics.nn.modules import Detect
from ultralytics.utils.checks import check_version
from ultralytics.utils.tal import make_anchors

__all__ = ["UltralyticsDetect"]


class Efficient_TRT_NMS(torch.autograd.Function):
    """NMS block for YOLO-fused model for TensorRT."""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        iou_threshold: float = 0.65,
        score_threshold: float = 0.25,
        max_output_boxes: float = 100,
        box_coding: int = 1,
        background_class: int = -1,
        score_activation: int = 0,
        plugin_version: str = '1',
        class_agnostic=0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_boxes, num_classes = scores.shape
        num_detections = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        detection_boxes = torch.randn(batch_size, max_output_boxes, 4, dtype=torch.float32)
        detection_scores = torch.randn(batch_size, max_output_boxes, dtype=torch.float32)
        detection_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)

        return num_detections, detection_boxes, detection_scores, detection_classes

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        iou_threshold: float = 0.7,
        score_threshold: float = 0.1,
        max_output_boxes: float = 300,
        box_coding: int = 1,
        background_class: int = -1,
        score_activation: int = 0,
        plugin_version: str = '1',
        class_agnostic=0,
    ) -> Tuple[Value, Value, Value, Value]:
        return g.op(
            'TRT::EfficientNMS_TRT',
            boxes,
            scores,
            outputs=4,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            score_threshold_f=score_threshold,
            max_output_boxes_i=max_output_boxes,
            background_class_i=background_class,
            score_activation_i=score_activation,
            class_agnostic_i=class_agnostic,
            plugin_version_s=plugin_version,
        )

class TRT_YOLO_NMS(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
        class_agnostic=0,
    ):

        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        det_indices = torch.randint(0,num_boxes,(batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes, det_indices

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=1,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version="1",
                 score_activation=0,
                 score_threshold=0.25,
                 class_agnostic=0):
        out = g.op("TRT::YOLO_NMS_TRT",
                   boxes,
                   scores,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   iou_threshold_f=iou_threshold,
                   max_output_boxes_i=max_output_boxes,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   class_agnostic_i=class_agnostic,
                   score_threshold_f=score_threshold,
                   outputs=5)
        nums, boxes, scores, classes, det_indices = out
        return nums, boxes, scores, classes, det_indices


class ONNX_YOLO_TRT(nn.Module):
    '''onnx module with TensorRT NMS operation.'''

    def __init__(self, class_agnostic=False, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None, device=None,
                 n_classes=80):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device('cpu')
        self.class_agnostic = 1 if class_agnostic else 0
        self.background_class = -1,
        self.box_coding = 1,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres
        self.n_classes = n_classes

    def forward(self, x):
        if isinstance(x, list):
            x = x[1]
        x = x.permute(0, 2, 1)
        bboxes_x = x[..., 0:1]
        bboxes_y = x[..., 1:2]
        bboxes_w = x[..., 2:3]
        bboxes_h = x[..., 3:4]
        bboxes = torch.cat([bboxes_x, bboxes_y, bboxes_w, bboxes_h], dim=-1)
        bboxes = bboxes.unsqueeze(2)  # [n_batch, n_bboxes, 4] -> [n_batch, n_bboxes, 1, 4]
        obj_conf = x[..., 4:]
        scores = obj_conf
        num_det, det_boxes, det_scores, det_classes, det_indices = TRT_YOLO_NMS.apply(bboxes, scores,
                                                                                      self.background_class,
                                                                                      self.box_coding,
                                                                                      self.iou_threshold, self.max_obj,
                                                                                      self.plugin_version,
                                                                                      self.score_activation,
                                                                                      self.score_threshold,
                                                                                      self.class_agnostic)
        return num_det, det_boxes, det_scores, det_classes, det_indices


class End2End_TRT(nn.Module):
    '''export onnx or tensorrt model with NMS operation.'''

    def __init__(self, model, class_agnostic=False, max_obj=100, iou_thres=0.45, score_thres=0.25, mask_resolution=56,
                 pooler_scale=0.25, sampling_ratio=0, max_wh=None, device=None, n_classes=80, is_det_model=True):
        super().__init__()
        device = device if device else torch.device('cpu')
        assert isinstance(max_wh, (int)) or max_wh is None
        self.model = model.to(device)
        self.model.model[-1].end2end = True
        if is_det_model:
            self.patch_model = ONNX_YOLO_TRT
            self.end2end = self.patch_model(class_agnostic, max_obj, iou_thres, score_thres, max_wh, device, n_classes)
        else:
            assert False
        self.end2end.eval()

    def forward(self, x):
        x = self.model(x)
        x = self.end2end(x)
        return x
"""
===============================================================================
            Ultralytics Detect head for detection models
===============================================================================
"""


class UltralyticsDetect(Detect):
    """Ultralytics Detect head for detection models."""

    max_det = 300
    iou_thres = 0.7
    conf_thres = 0.1

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        # Using transpose for compatibility with Efficient_TRT_NMS
        return Efficient_TRT_NMS.apply(
            dbox.transpose(1, 2),
            cls.sigmoid().transpose(1, 2),
            self.iou_thres,
            self.conf_thres,
            self.max_det,
        )