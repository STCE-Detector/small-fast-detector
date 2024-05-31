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
    im = im[[2, 1, 0], :, :] # Swap axes RGB TO BGR
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
    return im, (r, r), (dw, dh)

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

@torch.jit.script
def _dynsize_helper(crop_height_i, crop_width_i):
    """The input shape could be dynamic
    This will be exported as .ones().nonzero() with proper params
    """
    y = torch.arange(crop_height_i, dtype=torch.float32)
    x = torch.arange(crop_width_i, dtype=torch.float32)
    return y, x

def resize_bilinear(im,
                    resized_shape=None, output_crop_shape=None,
                    darknet=False, edge=True, axis=2):
    """Bilinear interpolate
    :param im: Image tensor shape (1xCxHxW)
    :type im: torch.Tensor
    :param resized_shape: shape of the resized image (H_r, W_r)
    :param output_crop_shape: shape of the output center crop (H_c, W_c)
    :param darknet: if should resize darknet-style
    :param edge: if should use edge (like in OpenCV)
    :param axis: height axis (0 or 2)
    :return: resized image
    :rtype: torch.Tensor
    """

    if resized_shape is None:
        assert output_crop_shape is not None, "No dimension given to resize"
        resized_shape = output_crop_shape

    input_height, input_width = im.shape[axis:axis + 2]
    if not isinstance(input_height, torch.Tensor):
        input_height, input_width = torch.tensor(input_height), torch.tensor(input_width)
    input_height, input_width = input_height.float(), input_width.float()

    assert resized_shape is not None, "No dimension given to resize"
    target_height, target_width = resized_shape
    if not isinstance(target_height, torch.Tensor):
        target_height, target_width = torch.tensor(target_height), torch.tensor(target_width)
    resized_shape_i = target_height, target_width
    target_height, target_width = target_height.float(), target_width.float()
    resized_shape = target_height, target_width

    top = left = None
    if output_crop_shape is None:
        crop_height_i, crop_width_i = resized_shape_i
        crop_height, crop_width = resized_shape
        top = 0
        left = 0
    else:
        crop_height_i, crop_width_i = output_crop_shape
        if not isinstance(crop_height_i, torch.Tensor):
            crop_height_i, crop_width_i = torch.tensor(crop_height_i), torch.tensor(crop_width_i)
        crop_height, crop_width = crop_height_i, crop_width_i

    if not crop_height.dtype.is_floating_point:
        crop_height, crop_width = crop_height.float(), crop_width.float()

    # TODO: ONNX does not like float in arange, can avoid .long() once issue #27718 is fixed in release
    if crop_height_i.dtype.is_floating_point:
        crop_height_i, crop_width_i = crop_height_i.long(), crop_width_i.long()

    # TODO: Use normal arange once issue #20075 is fixed in release
    y, x = _dynsize_helper(crop_height_i, crop_width_i)
    y, x = y.to(im.device), x.to(im.device)

    if top is None:
        assert left is None
        assert crop_height <= target_height and crop_width <= target_width, "invalid output_crop_shape"
        if not crop_height.dtype.is_floating_point:
            crop_height, crop_width = crop_height.float(), crop_width.float()
        # TODO: use .round() when PyTorch Issue # 25806 is fixed (round for ONNX is released)
        top = ((target_height - crop_height) / 2 + 0.5).floor()
        left = ((target_width - crop_width) / 2 + 0.5).floor()

    rh = target_height / input_height
    rw = target_width / input_width
    if edge:
        ty = (y + top + 1) / rh + 0.5 * (1 - 1.0 / rh) - 1
        tx = (x + left + 1) / rw + 0.5 * (1 - 1.0 / rw) - 1
        zero = torch.tensor(0.0, dtype=torch.float32)
        ty = torch.max(ty, zero)  # ty[ty < 0] = 0
        tx = torch.max(tx, zero)  # tx[tx < 0] = 0
    else:
        ty = (y + top) / rh
        tx = (x + left) / rw
    del y, x

    ity0 = ty.floor()
    if darknet:
        ity1 = ity0 + 1
    else:
        ity1 = ty.ceil()

    itx0 = tx.floor()
    if darknet:
        itx1 = itx0 + 1
    else:
        itx1 = tx.ceil()

    dy = ty - ity0
    dx = tx - itx0
    del ty, tx
    if axis == 0:
        dy = dy.view(-1, 1, 1)
        dx = dx.view(-1, 1)
    else:
        assert axis == 2, "Only 1xCxHxW and HxWxC inputs supported"
        dy = dy.view(-1, 1)
        dx = dx.view(-1)
    dydx = dy * dx

    # noinspection PyProtectedMember
    if torch._C._get_tracing_state():
        # always do clamp when tracing
        ity1 = torch.min(ity1, input_height - 1)
        itx1 = torch.min(itx1, input_width - 1)
    else:
        # TODO: use searchsorted once avaialble
        # items at the end could be out of bound (if upsampling)
        if ity1[-1] >= input_height:
            ity1[ity1 >= input_height] = input_height - 1
        if itx1[-1] >= input_width:
            itx1[itx1 >= input_width] = input_width - 1

    iy0 = ity0.long()
    ix0 = itx0.long()
    iy1 = ity1.long()
    ix1 = itx1.long()
    del ity0, itx0, ity1, itx1

    if not im.dtype.is_floating_point:
        im = im.float()
    im_iy0 = im.index_select(axis, iy0)
    im_iy1 = im.index_select(axis, iy1)
    d = im_iy0.index_select(axis + 1, ix0) * (1 - dx - dy + dydx) + \
        im_iy1.index_select(axis + 1, ix0) * (dy - dydx) + \
        im_iy0.index_select(axis + 1, ix1) * (dx - dydx) + \
        im_iy1.index_select(axis + 1, ix1) * dydx

    return d
