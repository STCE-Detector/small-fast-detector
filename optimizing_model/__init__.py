import sys
from pathlib import Path

import rich_click as click
from loguru import logger

from .export import paddle_export, torch_export

__all__ = [
    'trtyolo',
    'torch_export',
    'paddle_export',
]

logger.configure(handlers=[{'sink': sys.stdout, 'colorize': True, 'format': "<level>[{level.name[0]}]</level> <level>{message}</level>"}])


@click.group()
def trtyolo():
    """Command line tool for exporting models and performing inference with TensorRT-YOLO."""
    pass


@trtyolo.command(
    help="Export models for TensorRT-YOLO. Supports YOLOv5, YOLOv8, PP-YOLOE, and PP-YOLOE+. For YOLOv6, YOLOv7, and YOLOv9 please use the official repository for exporting."
)
@click.option('--model_dir', help='Path to the directory containing the PaddleDetection PP-YOLOE model.', type=str)
@click.option('--model_filename', help='The filename of the PP-YOLOE model.', type=str)
@click.option('--params_filename', help='The filename of the PP-YOLOE parameters.', type=str)
@click.option('-w', '--weights', help='Path to YOLO weights for PyTorch.', type=str)
@click.option('-v', '--version', help='Torch YOLO version, e.g., yolov5, yolov8.', type=str)
@click.option('--imgsz', default=640, help='Inference image size. Defaults to 640.', type=int)
@click.option('--repo_dir', default=None, help='Directory containing the local repository (if using torch.hub.load).', type=str)
@click.option('-o', '--output', help='Directory path to save the exported model.', type=str, required=True)
@click.option('-b', '--batch', default=1, help='Total batch size for the model. Use -1 for dynamic batch size. Defaults to 1.', type=int)
@click.option('--max_boxes', default=100, help='Maximum number of detections to output per image. Defaults to 100.', type=int)
@click.option('--iou_thres', default=0.45, help='NMS IoU threshold for post-processing. Defaults to 0.45.', type=float)
@click.option('--conf_thres', default=0.25, help='Confidence threshold for object detection. Defaults to 0.25.', type=float)
@click.option('--opset_version', default=11, help='ONNX opset version. Defaults to 11.', type=int)
@click.option('-s', '--simplify', is_flag=True, help='Whether to simplify the exported ONNX. Defaults is True.')
def export(
    model_dir,
    model_filename,
    params_filename,
    weights,
    version,
    imgsz,
    repo_dir,
    output,
    batch,
    max_boxes,
    iou_thres,
    conf_thres,
    opset_version,
    simplify,
):
    """Export models for TensorRT-YOLO.

    This command allows exporting models for both PaddlePaddle and PyTorch frameworks to be used with TensorRT-YOLO.
    """
    if model_dir and model_filename and params_filename:
        paddle_export(
            model_dir=model_dir,
            model_filename=model_filename,
            params_filename=params_filename,
            batch=batch,
            output=output,
            max_boxes=max_boxes,
            iou_thres=iou_thres,
            conf_thres=conf_thres,
            opset_version=opset_version,
            simplify=simplify,
        )
    elif weights and version:
        if version in ["yolov6", "yolov7", "yolov9"]:
            logger.warning(f"please use {version} official repository for exporting.")
        else:
            torch_export(
                weights=weights,
                output=output,
                version=version,
                imgsz=imgsz,
                batch=batch,
                max_boxes=max_boxes,
                iou_thres=iou_thres,
                conf_thres=conf_thres,
                opset_version=opset_version,
                simplify=simplify,
                repo_dir=repo_dir,
            )
    else:
        logger.error("Please provide correct export parameters.")