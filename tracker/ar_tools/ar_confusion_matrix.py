# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model validation metrics."""

import warnings

import numpy as np
import torch

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def plt_settings(rcparams=None, backend="Agg"):
    """
    Decorator to temporarily set rc parameters and the backend for a plotting function.

    Example:
        decorator: @plt_settings({"font.size": 12})
        context manager: with plt_settings({"font.size": 12}):

    Args:
        rcparams (dict): Dictionary of rc parameters to set.
        backend (str, optional): Name of the backend to use. Defaults to 'Agg'.

    Returns:
        (Callable): Decorated function with temporarily set rc parameters and backend. This decorator can be
            applied to any function that needs to have specific matplotlib rc parameters and backend for its execution.
    """

    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func):
        """Decorator to apply temporary rc parameters and backend to a function."""

        def wrapper(*args, **kwargs):
            """Sets rc parameters and backend, calls the original function, and restores the settings."""
            original_backend = plt.get_backend()
            if backend != original_backend:
                plt.close("all")  # auto-close()ing of figures upon backend switching is deprecated since 3.8
                plt.switch_backend(backend)

            with plt.rc_context(rcparams):
                result = func(*args, **kwargs)

            if backend != original_backend:
                plt.close("all")
                plt.switch_backend(original_backend)
            return result

        return wrapper

    return decorator


class ARConfusionMatrix:
    """
    A class for calculating and updating a confusion matrix for object detection and classification tasks.

    Attributes:
        task (str): The type of task, either 'detect' or 'classify'.
        matrix (np.array): The confusion matrix, with dimensions depending on the task.
        nc (int): The number of classes.
        conf (float): The confidence threshold for detections.
        iou_thres (float): The Intersection over Union threshold.
    """

    def __init__(self, nc=4, conf=0.25, iou_thres=0.45, class_names=()):
        """Initialize attributes for the YOLO model."""
        self.confusion_matrices = [torch.zeros((2, 2)) for _ in range(nc)]
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres
        self.class_names = {cls_name: i for i, cls_name in enumerate(class_names)}

        self.micro_recall = 0.0
        self.macro_recall = 0.0
        self.micro_precision = 0.0
        self.macro_precision = 0.0
        self.micro_f = 0.0
        self.macro_f = 0.0
        self.behaviour_precisions = []
        self.behaviour_recalls = []
        self.behaviour_f = []

    def process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Update confusion matrix for a multi-label object detection task.

        Args:
            detections (Array[N, 9]): Detected bounding boxes and their associated information.
                                      Each row should contain (x1, y1, x2, y2, conf, class1_conf, class2_conf, class3_conf, class4_conf).
            gt_bboxes (Array[M, 4]): Ground truth bounding boxes with xyxy format.
            gt_cls (Array[M, B]): The class labels for each category.
        """
        # Check dimensions
        assert detections.shape[1] == 5 + self.nc, f"Detected {detections.shape[1]} classes, expected {5 + self.nc}"
        assert gt_cls.shape[1] == self.nc, f"Detected {gt_cls.shape[1]} classes, expected {self.nc}"
        assert gt_bboxes.shape[1] == 4, f"Detected {gt_bboxes.shape[1]} coordinates, expected 4"

        # Filter detections by confidence score
        if detections is not None:
            detections = detections[detections[:, 4] > self.conf]

        if gt_cls.size(0) == 0:  # No ground truth labels
            if detections is not None:
                for i in range(self.nc):
                    detection_classes = detections[:, 5 + i].int()
                    for dc in detection_classes:
                        if dc != 0:
                            self.confusion_matrices[i][1, 0] += 1  # False Positive
            return

        if detections is None:
            for i in range(self.nc):
                gt_classes = gt_cls[:, i].int()
                for gc in gt_classes:
                    if gc != 0:
                        self.confusion_matrices[i][0, 1] += 1  # False Negative
            return

        gt_classes = gt_cls.int()
        detection_classes = detections[:, 5:].int()  # Binarize each class based on threshold
        iou = box_iou(gt_bboxes, detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        m0, m1, _ = matches.transpose().astype(int)

        for i in range(self.nc):  # Process each category independently
            gt_classes_i = gt_classes[:, i]
            detection_classes_i = detection_classes[:, i]

            matched_gt = set(m0)
            matched_det = set(m1)

            for j, gc in enumerate(gt_classes_i):
                if j in matched_gt:
                    k = m0 == j
                    dc = detection_classes_i[m1[k]][0]
                    if gc == dc and gc != 0:
                        self.confusion_matrices[i][1, 1] += 1  # True Positive
                    elif gc != 0 and dc == 0:
                        self.confusion_matrices[i][0, 1] += 1  # False Negative
                    elif gc == 0 and dc != 0:
                        self.confusion_matrices[i][1, 0] += 1  # False Positive
                    elif gc != dc and gc != 0 and dc != 0:
                        self.confusion_matrices[i][1, 0] += 1   # False Positive
                    elif gc == 0 and dc == 0:
                        self.confusion_matrices[i][0, 0] += 1  # True Negative
                else:
                    if gc != 0:
                        self.confusion_matrices[i][1, 0] += 1  # False Negative
                    else:
                        self.confusion_matrices[i][0, 0] += 1  # True Negative

            for l, dc in enumerate(detection_classes_i):
                if l not in matched_det:
                    if dc != 0:
                        self.confusion_matrices[i][1, 0] += 1  # False Positive
                    else:
                        self.confusion_matrices[i][0, 0] += 1  # True Negative

    def get_metrics(self):
        self.macro_metrics()
        self.micro_metrics()

        return self.micro_recall, self.micro_precision, self.macro_recall, self.macro_precision, self.micro_f, self.macro_f

    def save_results(self, save_dir, save=True, print_results=True):
        self.get_metrics()

        # Store metrics into a DataFrame and save it to a CSV file
        import pandas as pd
        metrics = [
            {'Class': 'Micro', 'Precision': self.micro_precision, 'Recall': self.micro_recall, 'F2': self.micro_f},
            {'Class': 'Macro', 'Precision': self.macro_precision, 'Recall': self.macro_recall, 'F2': self.macro_f}]
        for class_name, i in self.class_names.items():
            metrics.append({'Class': class_name, 'Precision': self.behaviour_precisions[i], 'Recall': self.behaviour_recalls[i], 'F2': self.behaviour_f[i]})
        metrics_df = pd.DataFrame(metrics)

        if print_results:
            # Display the DataFrame
            print("\nAction Recognition Metrics:")
            print(metrics_df.to_string(float_format="{:.4f}".format, index=False, col_space=10))

        if save:
            metrics_df.to_csv(save_dir + '/metrics.csv', index=False)

            # Save class confusion matrices
            for class_name, i in self.class_names.items():
                self.plot(class_name, save_dir=save_dir)

        return metrics_df

    def macro_metrics(self):
        combined_matrix = torch.stack(self.confusion_matrices, 0).sum(0)
        self.macro_recall = (combined_matrix[1, 1] / (combined_matrix[1, 1] + combined_matrix[0, 1])).item()
        self.macro_precision = (combined_matrix[1, 1] / (combined_matrix[1, 1] + combined_matrix[1, 0])).item()
        self.macro_f = self.f_metric(self.macro_precision, self.macro_recall, beta=2.0)

    def micro_metrics(self):
        for matrix in self.confusion_matrices:
            tp = matrix[1, 1]
            fp = matrix[1, 0]
            fn = matrix[0, 1]
            self.behaviour_precisions.append((tp / (tp + fp + 1e-9)).item())
            self.behaviour_recalls.append((tp / (tp + fn + 1e-9)).item())
        self.micro_recall = sum(self.behaviour_recalls) / len(self.behaviour_recalls)
        self.micro_precision = sum(self.behaviour_precisions) / len(self.behaviour_precisions)

        for precisions, recalls in zip(self.behaviour_precisions, self.behaviour_recalls):
            self.behaviour_f.append(self.f_metric(precisions, recalls, beta=2.0))
        self.micro_f = sum(self.behaviour_f) / len(self.behaviour_f)

    @staticmethod
    def f_metric(precision, recall, beta=1.0):
        return (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 1e-9)

    @plt_settings()
    def plot(self, name, save_dir="", on_plot=None):
        """
        Plot the confusion matrix using seaborn and save it to a file.

        Args:
            name (str): Name of the class.
            save_dir (str): Directory where the plot will be saved.
            on_plot (func): An optional callback to pass plots path and data when they are rendered.
        """
        import seaborn as sn

        array = self.confusion_matrices[self.class_names[name]]
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
        nc, nn = 2, 2  # number of classes, names
        sn.set(font_scale=1.2 if nc < 50 else 0.8)  # for label size
        ticklabels = ["Null", "Active"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 14},
                cmap="Blues",
                fmt=".0f",
                square=True,
                vmin=0.0,
                vmax=None,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))

        title = f"{name} Confusion Matrix @{self.iou_thres}&{self.conf}"
        suptitle = f"Recall: {round(self.behaviour_recalls[self.class_names[name]] * 100, 2)}% " + \
                     f"Precision: {round(self.behaviour_precisions[self.class_names[name]] * 100, 2)}%" + \
                        f" F-score: {round(self.behaviour_f[self.class_names[name]] * 100, 2)}%"
        plot_fname = str(save_dir) + f"/confusion_matrix@{self.iou_thres}&{self.conf}_{name}.png"

        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        plt.suptitle(title)
        plt.title(suptitle)
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname)


