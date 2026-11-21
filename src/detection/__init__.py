"""Pest Object Detection Package (REQ-08).

Provides:
- Bounding-box transformations, coordinate validation, clipping, IoU, and NMS.
- Detection evaluation metrics: Precision, Recall, mAP@0.5, and mAP@0.5:0.95.
- YOLO-style single-stage and SSDLite detector architectures.
- Dataset parsers (YOLO txt and VOC XML) and synthetic benchmark generator.
- Visualization with ground truth and predicted bounding boxes.
"""

from src.detection.boxes import (
    box_iou,
    clip_boxes_to_image,
    cxcywh_to_xyxy,
    non_max_suppression,
    validate_boxes,
    xyxy_to_cxcywh,
)
from src.detection.metrics import calculate_map, evaluate_detections

__all__ = [
    "box_iou",
    "clip_boxes_to_image",
    "cxcywh_to_xyxy",
    "xyxy_to_cxcywh",
    "validate_boxes",
    "non_max_suppression",
    "calculate_map",
    "evaluate_detections",
]

