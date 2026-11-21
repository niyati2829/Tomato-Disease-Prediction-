"""Bounding-Box Coordinate Geometry & Post-Processing (REQ-08).

Formulations:
1. Box IoU:
       IoU(A, B) = Area(A cap B) / (Area(A) + Area(B) - Area(A cap B))
2. Coordinate Transformations:
       cx = (xmin + xmax) / 2,   cy = (ymin + ymax) / 2
       w  = xmax - xmin,         h  = ymax - ymin
3. Non-Maximum Suppression (NMS):
       Greedy suppression of lower-confidence candidate boxes with IoU > threshold.

Author: Crop Disease Detection Project Team
"""

from typing import List, Tuple

import numpy as np
import torch


def xyxy_to_cxcywh(boxes: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert bounding boxes from [xmin, ymin, xmax, ymax] to [cx, cy, w, h]."""
    is_torch = isinstance(boxes, torch.Tensor)
    if is_torch:
        xmin, ymin, xmax, ymax = boxes.unbind(-1)
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin
        return torch.stack([cx, cy, w, h], dim=-1)
    else:
        boxes = np.asarray(boxes, dtype=float)
        cx = (boxes[..., 0] + boxes[..., 2]) / 2.0
        cy = (boxes[..., 1] + boxes[..., 3]) / 2.0
        w = boxes[..., 2] - boxes[..., 0]
        h = boxes[..., 3] - boxes[..., 1]
        return np.stack([cx, cy, w, h], axis=-1)


def cxcywh_to_xyxy(boxes: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert bounding boxes from [cx, cy, w, h] to [xmin, ymin, xmax, ymax]."""
    is_torch = isinstance(boxes, torch.Tensor)
    if is_torch:
        cx, cy, w, h = boxes.unbind(-1)
        xmin = cx - 0.5 * w
        ymin = cy - 0.5 * h
        xmax = cx + 0.5 * w
        ymax = cy + 0.5 * h
        return torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    else:
        boxes = np.asarray(boxes, dtype=float)
        xmin = boxes[..., 0] - 0.5 * boxes[..., 2]
        ymin = boxes[..., 1] - 0.5 * boxes[..., 3]
        xmax = boxes[..., 0] + 0.5 * boxes[..., 2]
        ymax = boxes[..., 1] + 0.5 * boxes[..., 3]
        return np.stack([xmin, ymin, xmax, ymax], axis=-1)


def clip_boxes_to_image(
    boxes: torch.Tensor,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """Clip [xmin, ymin, xmax, ymax] coordinates within image boundaries [0, W] and [0, H].

    Args:
        boxes: Tensor of shape (N, 4).
        image_size: Tuple of (H, W).
    """
    H, W = image_size
    clipped = boxes.clone()
    clipped[:, 0] = torch.clamp(clipped[:, 0], min=0.0, max=float(W))
    clipped[:, 1] = torch.clamp(clipped[:, 1], min=0.0, max=float(H))
    clipped[:, 2] = torch.clamp(clipped[:, 2], min=0.0, max=float(W))
    clipped[:, 3] = torch.clamp(clipped[:, 3], min=0.0, max=float(H))
    return clipped


def validate_boxes(boxes: torch.Tensor | np.ndarray) -> bool:
    """Verify that all bounding boxes have strictly positive width and height."""
    if isinstance(boxes, torch.Tensor):
        if boxes.numel() == 0:
            return True
        valid_w = (boxes[:, 2] > boxes[:, 0])
        valid_h = (boxes[:, 3] > boxes[:, 1])
        return bool(torch.all(valid_w & valid_h).item())
    else:
        boxes = np.asarray(boxes)
        if len(boxes) == 0:
            return True
        valid_w = (boxes[:, 2] > boxes[:, 0])
        valid_h = (boxes[:, 3] > boxes[:, 1])
        return bool(np.all(valid_w & valid_h))


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute pairwise Intersection over Union between two box sets in [xmin, ymin, xmax, ymax].

    Args:
        boxes1: Tensor of shape (N, 4).
        boxes2: Tensor of shape (M, 4).

    Returns:
        Tensor of shape (N, M) with IoU values in [0.0, 1.0].
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)

    # Intersections
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - intersection
    iou = intersection / (union + eps)
    return iou


def non_max_suppression(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
) -> torch.Tensor:
    """Perform Non-Maximum Suppression (NMS).

    Args:
        boxes: Tensor of shape (N, 4) in [xmin, ymin, xmax, ymax].
        scores: Confidence scores tensor of shape (N,).
        iou_threshold: Overlap threshold above which suppressed boxes are discarded.
        score_threshold: Minimum confidence score to be considered.

    Returns:
        Tensor of integer indices of kept boxes.
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    # Filter out low-confidence predictions
    valid_idx = torch.nonzero(scores >= score_threshold).squeeze(1)
    if valid_idx.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    b = boxes[valid_idx]
    s = scores[valid_idx]

    # Sort descending by confidence score
    order = torch.argsort(s, descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(valid_idx[i].item())

        if order.numel() == 1:
            break

        current_box = b[i].unsqueeze(0)
        other_boxes = b[order[1:]]

        ious = box_iou(current_box, other_boxes).squeeze(0)
        # Keep boxes with IoU <= iou_threshold
        mask = ious <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long)

