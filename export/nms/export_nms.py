import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Optional

import onnx
import torch
from loguru import logger

from .head import UltralyticsDetect
from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz


__all__ = ['torch_export', ]

# Filter warnings
warnings.filterwarnings("ignore")

# For scripts
logger.configure(handlers=[{'sink': sys.stdout, 'colorize': True, 'format': "<level>[{level.name[0]}]</level> <level>{message}</level>"}])

HEADS = {
    "Detect": {"yolov8": UltralyticsDetect, "ultralytics": UltralyticsDetect},
}

OUTPUT_NAMES = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']

DYNAMIC_AXES = {
    "images": {0: "batch", 2: "height", 3: "width"},
    "num_dets": {0: "batch"},
    "det_boxes": {0: "batch"},
    "det_scores": {0: "batch"},
    "det_classes": {0: "batch"},
}


def load_model(version: str, weights: str, repo_dir: Optional[str] = None) -> Optional[torch.nn.Module]:
    """
    Load YOLO model based on version and weights.

    Args:
        version (str): YOLO version, e.g., yolov3, yolov5, yolov6, yolov7, yolov8, yolov9, ultralytics.
        weights (str): Path to YOLO weights for PyTorch.
        repo_dir (Optional[str], optional): Directory containing the local repository (if using torch.hub.load). Defaults to None.

    Returns:
        torch.nn.Module: Loaded YOLO model or None if the version is not supported.
    """
    yolo_versions_with_repo = {
        'yolov3': 'ultralytics/yolov3',
        'yolov5': 'ultralytics/yolov5',
    }

    source = 'github' if repo_dir is None else 'local'

    if version in yolo_versions_with_repo:
        repo_dir = yolo_versions_with_repo[version] if repo_dir is None else repo_dir
        return torch.hub.load(repo_dir, 'custom', path=weights, source=source, verbose=False)
    elif version in ['yolov8', 'ultralytics']:
        return YOLO(model=weights, verbose=False).model
    else:
        logger.error(f"YOLO version '{version}' not supported!")
        return None


def update_model(
    model: torch.nn.Module, version: str, dynamic: bool, max_boxes: int, iou_thres: float, conf_thres: float
) -> Optional[torch.nn.Module]:
    """
    Update YOLO model with dynamic settings.

    Args:
        model (torch.nn.Module): YOLO model to be updated.
        version (str): YOLO version, e.g., yolov3, yolov5, yolov6, yolov7, yolov8, yolov9, ultralytics.
        dynamic (bool): Whether to use dynamic settings.
        max_boxes (int): Maximum number of detections to output per image.
        iou_thres (float): NMS IoU threshold for post-processing.
        conf_thres (float): Confidence threshold for object detection.

    Returns:
        torch.nn.Module: Updated YOLO model or None if the version is not supported.
    """
    model = deepcopy(model).to(torch.device("cpu"))
    supported = False

    for m in model.modules():
        class_name = m.__class__.__name__
        if class_name in HEADS:
            detect_head = HEADS[class_name].get(version)
            if detect_head:
                supported = True
                detect_head.dynamic = dynamic
                detect_head.max_det = max_boxes
                detect_head.iou_thres = iou_thres
                detect_head.conf_thres = conf_thres
                m.__class__ = detect_head
            break

    if not supported:
        logger.error(f"YOLO version '{version}' detect head not supported!")
        return None

    return model


def torch_export(
    weights: str,
    output: str,
    version: str,
    imgsz: Optional[int] = 640,
    batch: Optional[int] = 1,
    max_boxes: Optional[int] = 300,
    iou_thres: Optional[float] = 0.7,
    conf_thres: Optional[float] = 0.1,
    opset_version: Optional[int] = 17,
    slim: Optional[bool] = False,
    repo_dir: Optional[str] = None,
) -> None:
    """
    Export YOLO model to ONNX format using Torch.

    Args:
        weights (str): Path to YOLO weights for PyTorch.
        output (str): Directory path to save the exported model.
        version (str): YOLO version, e.g., yolov3, yolov5, yolov6, yolov7, yolov8, yolov9, ultralytics.
        imgsz (Optional[int], optional): Inference image size. Defaults to 640.
        batch (Optional[int], optional): Total batch size for the model. Use -1 for dynamic batch size. Defaults to 1.
        max_boxes (Optional[int], optional): Maximum number of detections to output per image. Defaults to 100.
        iou_thres (Optional[float], optional): NMS IoU threshold for post-processing. Defaults to 0.45.
        conf_thres (Optional[float], optional): Confidence threshold for object detection. Defaults to 0.25.
        opset_version (Optional[int], optional): ONNX opset version. Defaults to 11.
        repo_dir (Optional[str], optional): Directory containing the local repository (if using torch.hub.load). Defaults to None.
    """
    logger.info("Starting export with Pytorch.")
    model = load_model(version, weights, repo_dir)
    if model is None:
        return

    dynamic = batch <= 0
    batch = 1 if batch <= 0 else batch
    model = update_model(model, version, dynamic, max_boxes, iou_thres, conf_thres)
    if model is None:
        return

    imgsz = check_imgsz(imgsz, stride=model.stride, min_dim=2)

    im = torch.zeros(batch, 3, *imgsz).to(torch.device("cpu"))

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    for _ in range(2):  # Warm-up run
        model(im)

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    onnx_filepath = output_path / (Path(weights).stem + "_nms" + ".onnx")

    torch.onnx.export(
        model=model,
        args=im,
        f=str(onnx_filepath),
        opset_version=opset_version,
        input_names=['images'],
        output_names=OUTPUT_NAMES,
        dynamic_axes=DYNAMIC_AXES if dynamic else None,
    )

    model_onnx = onnx.load(onnx_filepath)
    onnx.checker.check_model(model_onnx)

    # Update dynamic axes names
    shapes = {
        'num_dets': ["batch" if dynamic else batch, 1],
        'det_boxes': ["batch" if dynamic else batch, max_boxes, 4],
        'det_scores': ["batch" if dynamic else batch, max_boxes],
        'det_classes': ["batch" if dynamic else batch, max_boxes],
    }
    for node in model_onnx.graph.output:
        for idx, dim in enumerate(node.type.tensor_type.shape.dim):
            dim.dim_param = str(shapes[node.name][idx])
    if slim:
        try:
            import onnxslim
            logger.info(f"Slimming with {onnxslim.__version__}...")
            model_onnx, check = onnxslim.slim(model_onnx)
            print("Finish! Here is the difference:")
            assert check, "Slimmer ONNX model could not be validated"
        except Exception as e:
            logger.warning(f"slimmer failure: {e}")

    onnx.save(model_onnx, onnx_filepath)
    onnx.save(model_onnx, onnx_filepath)
    model_onnx = onnx.load(onnx_filepath)
    onnx.checker.check_model(model_onnx)

    logger.success(f'Export complete, results saved to {output}, visualize at https://netron.app')



def nms_export(config):
    torch_export(
        weights=config["model_path"],
        output=config["output"],
        version="yolov8",
        imgsz=config["imgsz"],
        batch=config["batch"],
        max_boxes=config["max_boxes"],
        iou_thres=config["iou_thres"],
        conf_thres=config["conf_thres"],
        opset_version=config["opset"],
        slim=config["simplify"],
    )

