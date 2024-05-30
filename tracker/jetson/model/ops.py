from typing import List, Union, Tuple
import cv2
import numpy as np
import torch
from torch import Tensor
import torch
import torch.nn.functional as F
import torchvision.transforms as torchtransforms
from ultralytics.utils import IS_JETSON
if IS_JETSON:
    import jetson_utils

def letterbox_cuda(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    resized_img = jetson_utils.cudaAllocMapped(width=new_unpad[0], height=new_unpad[1], format=img.format)
    jetson_utils.cudaResize(img, resized_img)

    bordered_img = jetson_utils.cudaAllocMapped(width=new_shape[1], height=new_shape[0], format=img.format)
    jetson_utils.cudaOverlay(resized_img, bordered_img, int(dw), int(dh))

    return bordered_img, ratio, (dw, dh)


def letterbox_pytorch(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32, dtype=torch.float32):
    # Tensor images are expected to be in (C, H, W)
    # Input here is in (H, W, C), convert it to (C, H, W)
    im = im.permute(2, 0, 1)  # Swap axes - PyTorch expects CHW
    shape = im.shape[1:]  # Current shape [height, width]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # Only scale down, do not scale up (for better val mAP)
        ratio = min(ratio, 1.0)

    # Compute padding
    new_unpad = (int(round(shape[0] * ratio)), int(round(shape[1] * ratio)))
    dw, dh = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]  # wh padding

    if auto:  # Minimum rectangle
        dw, dh = dw % stride, dh % stride  # wh padding
    dw /= 2  # Divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # Resize
        resized_im = F.interpolate(im.unsqueeze(0), size=new_unpad, mode='bilinear', align_corners=False).squeeze(0)
    else:
        resized_im = im

    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))

    # Create an empty tensor with the desired color and shape
    empty_tensor = torch.full((im.size(0), new_shape[0], new_shape[1]), color[0], device=im.device, dtype=dtype)

    # Add border - image might be smaller than output size, fill out border and centralize
    empty_tensor[:, top:top + resized_im.size(1), left:left + resized_im.size(2)] = resized_im

    return empty_tensor, (ratio, ratio), (dw, dh)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32, dtype=torch.float32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.
    """
    if isinstance(boxes, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, shape[0])  # y1, y2
    return boxes

def process_nms_trt_results(preds: List[Tensor], names: List[str]) -> List[Tensor]:
    """
    Filter TensorRT-like bounding box structure via `max_det`

    Args:
        preds (List[torch.Tensor]): list of
            `num_dets` (shape: Bx1)
            `bboxes` (shape: BxTOP_Kx4)
            `labels` (shape: BxTOP_Kx1)
            `scores` (shape: BxTOP_Kx1).
            Order is not guaranteed but sync with names.
        names (List[str]): list of outputs names like [`num_dets`, ...].

    Returns:
       (List[torch.Tensor]): YOLO-like list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class).
    """

    named_dict = dict(zip(names, preds))  # dict and zip dont copy data
    outputs = []

    for boxes, scores, labels, num_dets in zip(
            named_dict["det_boxes"],
            named_dict["det_scores"],
            named_dict["det_classes"],
            named_dict["num_dets"],
    ):
        boxes, scores, labels = boxes[:num_dets], scores[:num_dets, None], labels[:num_dets, None]
        outputs.append(torch.hstack([boxes, scores, labels]))

    return outputs


def process_nms_onnx_results(preds: Tensor) -> List[Tensor]:
    """
    Filter ONNX-like bounding box structure via `max_det`

    Args:
        preds (torch.Tensor): Tensor of shape Nx7. Contains
            `batch_index`     - 1
            `bboxes`          - 4
            `max_confidence`  - 1
            `class`           - 1

    Returns:
       (List[torch.Tensor]): YOLO-like list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class).
    """

    batch_index, yolo_dets = preds[..., 0], preds[..., 1:]
    bs = int(batch_index[-1])

    outputs = []

    for i in range(bs + 1):
        outputs.append(yolo_dets[batch_index == i])

    return outputs
