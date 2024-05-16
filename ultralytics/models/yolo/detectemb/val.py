# Ultralytics YOLO 🚀, AGPL-3.0 license

from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class DetectionEmbValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a segmentation model.

    Example:
        ```python
        from ultralytics.models.yolo.segment import SegmentationValidator

        args = dict(model='yolov8n-seg.pt', data='coco8-seg.yaml')
        validator = SegmentationValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None
        self.process = None
        self.args.task = "detectemb"
        #self.metrics = SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)

    def preprocess(self, batch):
        """Preprocesses batch by converting masks to float and sending to device."""
        batch = super().preprocess(batch)
        # TODO: save_hybrid tags?
        batch["tags"] = batch["tags"].to(self.device).float()
        return batch

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
            nc=self.nc,
        )
    #def init_metrics(self, model):
    #def get_desc(self):

    def _prepare_batch(self, si, batch):
        """Prepares a batch for training or inference by processing images and targets."""
        prepared_batch = super()._prepare_batch(si, batch)
        midx = [si] if self.args.overlap_mask else batch["batch_idx"] == si
        prepared_batch["tags"] = batch["tags"][midx]
        return prepared_batch

    # TODO: not sure if should be used
    #def _prepare_pred(self, pred, pbatch, proto):
    #def update_metrics(self, preds, batch):
    #def _process_batch(self, detections, gt_bboxes, gt_cls, pred_masks=None, gt_masks=None, overlap=False, masks=False):
    #def plot_val_samples(self, batch, ni):
    #def plot_predictions(self, batch, preds, ni):
    #def pred_to_json(self, predn, filename, pred_masks):
