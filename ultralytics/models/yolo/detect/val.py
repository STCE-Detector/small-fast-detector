# Ultralytics YOLO 🚀, AGPL-3.0 license
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class DetectionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionValidator

        args = dict(model='yolov8n.pt', data='coco8.yaml')
        validator = DetectionValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = (
                [
                    torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                    for i in range(nb)
                ]
                if self.args.save_hybrid
                else []
            )  # for autolabelling

        return batch

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, '')  # validation path
        self.is_coco = (isinstance(val, str) and 'coco' in val and val.endswith(f'{os.sep}val2017.txt')) or \
                       os.path.isfile(os.path.join(self.data['path'], 'annotations', 'instances_val2017.json')) # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = list(range(1000))
        self.args.save_json |= (self.is_coco or self.is_lvis) and not self.training  # run on final val if training COCO
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        # Confusion matrix for two scenarios
        self.confusion_matrix_p = ConfusionMatrix(nc=self.nc, conf=0.3, iou_thres=0.3)
        self.confusion_matrix_r = ConfusionMatrix(nc=self.nc, conf=0.001, iou_thres=0.3)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

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
        )

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    # TODO: obb has not supported confusion_matrix yet.
                    if self.args.plots and self.args.task != "obb":
                        self.confusion_matrix_p.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                        self.confusion_matrix_r.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                # TODO: obb has not supported confusion_matrix yet.
                if self.args.plots and self.args.task != "obb":
                    self.confusion_matrix_p.process_batch(predn, bbox, cls)
                    self.confusion_matrix_r.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                file = self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, pbatch["ori_shape"], file)

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.metrics.speed = self.speed
        # TODO: which confusion matrix should be used?
        self.metrics.confusion_matrix = self.confusion_matrix_p

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        self.nt_per_class = np.bincount(
            stats["target_cls"].astype(int), minlength=self.nc
        )  # number of targets per class
        return self.metrics.results_dict

    def print_results(self, stats):
        """Prints training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i)))

        if self.args.plots:
            for normalize in [False, 'gt', 'pred']:
                self.confusion_matrix_p.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )
                self.confusion_matrix_r.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

        # Save micro metrics only if not training
        if not self.training:
            stats[f'metrics/muAR@{self.confusion_matrix_p.iou_thres}&{self.confusion_matrix_p.conf}'] = self.confusion_matrix_p.micro_recall
            stats[f'metrics/MAR@{self.confusion_matrix_p.iou_thres}&{self.confusion_matrix_p.conf}'] = self.confusion_matrix_p.macro_recall
            stats[f'metrics/muAP@{self.confusion_matrix_p.iou_thres}&{self.confusion_matrix_p.conf}'] = self.confusion_matrix_p.micro_precision
            stats[f'metrics/MAP@{self.confusion_matrix_p.iou_thres}&{self.confusion_matrix_p.conf}'] = self.confusion_matrix_p.macro_precision

            stats[f'metrics/muAR@{self.confusion_matrix_r.iou_thres}&{self.confusion_matrix_r.conf}'] = self.confusion_matrix_r.micro_recall
            stats[f'metrics/MAR@{self.confusion_matrix_r.iou_thres}&{self.confusion_matrix_r.conf}'] = self.confusion_matrix_r.macro_recall
            stats[f'metrics/muAP@{self.confusion_matrix_r.iou_thres}&{self.confusion_matrix_r.conf}'] = self.confusion_matrix_r.micro_precision
            stats[f'metrics/MAP@{self.confusion_matrix_r.iou_thres}&{self.confusion_matrix_r.conf}'] = self.confusion_matrix_r.macro_precision

            # Save macro metrics per class
            for i, c in enumerate(self.metrics.ap_class_index):
                stats[f"metrics/P_{self.names[c]}"] = self.metrics.class_result(i)[0]
                stats[f"metrics/R_{self.names[c]}"] = self.metrics.class_result(i)[1]

        return stats

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in predn.tolist():
            xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            with open(file, "a") as f:
                f.write(("%g " * len(line)).rstrip() % line + "\n")

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/instances_val2017.json"  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO  # noqa
                #from pycocotools.cocoeval import COCOeval  # noqa
                from ultralytics.utils.cocoeval import COCOeval

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                eval = COCOeval(anno, pred, "bbox")

                # Set Custom Area Ranges
                # TODO: should we set 10*15 as the minimum area? before it was 0**2
                eval.params.areaRng = [[0**2, 1e5 ** 2],
                                       [0**2, 16 ** 2],
                                       [16 ** 2, 32 ** 2],
                                       [32 ** 2, 96 ** 2],
                                       [96 ** 2, 1e5 ** 2]]
                eval.params.areaRngLbl = ['all', 'tiny', 'small', 'medium', 'large']
                eval.params.maxDets = [3, 30, 300]

                if self.is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                eval.evaluate()
                eval.accumulate()
                eval.summarize()

                # Update metrics
                stats[self.metrics.keys[-1]] = eval.stats[0]  # update mAP50-95
                stats[self.metrics.keys[-2]] = eval.stats[5]  # update mAP50

                # Set some metrics in the stats dictionary
                # Precision
                stats['metrics/AP(T)'] = eval.stats[1]
                stats['metrics/AP(S)'] = eval.stats[2]
                stats['metrics/AP(M)'] = eval.stats[3]
                stats['metrics/AP(L)'] = eval.stats[4]

                # Recall by IoU
                stats['metrics/AR(50:95)'] = eval.stats[25]
                stats['metrics/AR(75)'] = eval.stats[30]
                stats['metrics/AR(50)'] = eval.stats[31]

                # Recall by Area
                stats['metrics/AR(T)'] = eval.stats[26]
                stats['metrics/AR(S)'] = eval.stats[27]
                stats['metrics/AR(M)'] = eval.stats[28]
                stats['metrics/AR(L)'] = eval.stats[29]

                # Get all metrics in a dictionary
                self.extract_cocoeval_metrics(eval)

                # Save results to file
                results_file = self.save_dir / "evaluation_results.json"
                with results_file.open("w") as file:
                    json.dump(stats, file)
            except Exception as e:
                LOGGER.warning(f"pycocotools unable to run: {e}")
        return stats

    def extract_cocoeval_metrics(self, eval):
        """Extracts metrics from COCOeval object and saves them to a DataFrame."""

        # Function to append metrics
        def append_metrics(metrics, metric_type, iou, area, max_dets, value):
            metrics.append({
                'Metric Type': metric_type,
                'IoU': iou,
                'Area': area,
                'Max Detections': max_dets,
                'Value': value
            })

        # Initialize a list to store the metrics
        metrics_ = []

        # Extract metrics for bbox/segm evaluation
        iou_types = ['0.50:0.95', '0.50', '0.75']
        areas = eval.params.areaRngLbl
        max_dets = eval.params.maxDets

        # Extract AP metrics (indices 0-14: 3 IoUs * 5 areas)
        for i, iou in enumerate(iou_types):
            for j, area in enumerate(areas):
                idx = i * len(areas) + j
                append_metrics(metrics_, 'AP', iou, area, max_dets[-1], eval.stats[idx])

        # Extract AR metrics (indices 15-17: 3 maxDets for 'all' area)
        num_ap_metrics = len(iou_types) * len(areas)  # Total number of AP metrics

        # Iterate over max_dets to append AR metrics
        for i, md in enumerate(max_dets):
            for j, area in enumerate(areas):
                idx = num_ap_metrics + j + i * len(areas)  # Adjust index calculation for AR
                append_metrics(metrics_, 'AR', '0.50:0.95', area, md, eval.stats[idx])

        # Append AR metrics for 0.75 and 0.50 IoU
        for i, iou in enumerate(['0.75', '0.50']):
            append_metrics(metrics_, 'AR', iou, 'all', '300', eval.stats[idx + i + 1])

        # Convert to DataFrame
        df_metrics = pd.DataFrame(metrics_)

        # Save to file
        df_metrics.to_csv(self.save_dir / "cocoeval_results.csv", index=False)

        # Write to log
        self.metrics.cocoeval_df = df_metrics
