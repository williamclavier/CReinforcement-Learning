"""
Custom non-max suppression for Clash Royale detection.
Ported from KataCR/katacr/yolov8/custom_utils.py

Handles the 7-column output format: (x1, y1, x2, y2, confidence, class, belong)
"""

import torch
import torchvision
import time
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils import LOGGER


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for belong attribute.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4, num_boxes)
            containing the predicted boxes, classes, and belong. The last class index is 'belong'.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): Apriori labels for autolabelling.
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model (including belong).
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.
        rotated (bool): If True, handle rotated boxes.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 7) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, belong).
    """
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes (including belong)
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    time_limit = 2.0 + max_time_img * bs
    multi_label &= nc > 1

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)

    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)

    t = time.time()
    output = [torch.zeros((0, 7 + nm), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # confidence filter

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        # Split into box, cls (nc-1 classes), bel (1), mask
        # The model outputs nc classes where the last one is 'belong'
        box, cls, bel, mask = x.split((4, nc - 1, 1, nm), 1)

        # Best class only
        conf, j = cls.max(1, keepdim=True)
        bel = (bel > 0.5).float()
        x = torch.cat((box, conf, j.float(), bel), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]

        if rotated:
            from ultralytics.utils.ops import nms_rotated
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c
            i = torchvision.ops.nms(boxes, scores, iou_thres)

        i = i[:max_det]
        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break

    return output
