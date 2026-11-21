"""Quantitative Object Detection Evaluation Metrics (REQ-08).

Formulations:
1. Precision = TP / (TP + FP)
2. Recall = TP / (TP + FN)
3. Average Precision (AP) = Area under the Precision-Recall curve:
       AP = sum_{k=1}^N (R_k - R_{k-1}) * P_interp(R_k)
4. Mean Average Precision:
       mAP@0.5 = (1 / C) * sum_{c=1}^C AP_c^{IoU=0.5}
       mAP@0.5:0.95 = Mean over IoU in [0.50, 0.55, ..., 0.95]

Author: Crop Disease Detection Project Team
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from src.detection.boxes import box_iou


def compute_class_ap(
    pred_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Compute Average Precision, Precision, and Recall for a single class at an IoU threshold.

    Args:
        pred_boxes: List of predicted box tensors (N_i, 4) across test images.
        pred_scores: List of score tensors (N_i,) across test images.
        gt_boxes: List of ground truth box tensors (M_i, 4) across test images.
        iou_threshold: Minimum IoU to count as a True Positive (e.g. 0.5).

    Returns:
        Tuple of (AP, Precision, Recall).
    """
    total_gt = sum(len(g) for g in gt_boxes)
    if total_gt == 0:
        return 0.0, 0.0, 0.0

    # Flatten predictions with image IDs
    all_preds = []
    for img_id, (boxes, scores) in enumerate(zip(pred_boxes, pred_scores)):
        for b, s in zip(boxes, scores):
            all_preds.append((img_id, float(s), b))

    if len(all_preds) == 0:
        return 0.0, 0.0, 0.0

    # Sort descending by confidence score
    all_preds.sort(key=lambda x: x[1], reverse=True)

    # Track matched ground truths per image
    matched_gt = {img_id: [False] * len(gt_boxes[img_id]) for img_id in range(len(gt_boxes))}

    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    for p_idx, (img_id, score, p_box) in enumerate(all_preds):
        img_gt = gt_boxes[img_id]
        if len(img_gt) == 0:
            fp[p_idx] = 1.0
            continue

        ious = box_iou(p_box.unsqueeze(0), img_gt).squeeze(0)
        best_iou, best_gt_idx = torch.max(ious, dim=0)
        best_iou = float(best_iou.item())
        best_gt_idx = int(best_gt_idx.item())

        if best_iou >= iou_threshold:
            if not matched_gt[img_id][best_gt_idx]:
                tp[p_idx] = 1.0
                matched_gt[img_id][best_gt_idx] = True
            else:
                fp[p_idx] = 1.0  # Duplicate detection
        else:
            fp[p_idx] = 1.0

    # Cumulative TP and FP
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    precisions = cum_tp / (cum_tp + cum_fp)
    recalls = cum_tp / total_gt

    # Compute AP via 101-point or trapezoidal interpolation
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Monotonic precision envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    final_prec = float(precisions[-1]) if len(precisions) > 0 else 0.0
    final_rec = float(recalls[-1]) if len(recalls) > 0 else 0.0

    return ap, final_prec, final_rec


def calculate_map(
    predictions_per_class: Dict[int, Tuple[List[torch.Tensor], List[torch.Tensor]]],
    ground_truth_per_class: Dict[int, List[torch.Tensor]],
    iou_thresholds: List[float] | None = None,
) -> Dict[str, Any]:
    """Calculate mAP@0.5 and mAP@0.5:0.95 across multiple detection classes.

    Args:
        predictions_per_class: Dict mapping class_id -> (list_of_pred_boxes, list_of_scores).
        ground_truth_per_class: Dict mapping class_id -> list_of_gt_boxes.
        iou_thresholds: List of thresholds (default: [0.50, 0.55, ..., 0.95]).

    Returns:
        Dictionary containing mAP@0.5, mAP@0.5:0.95, and per-class AP breakdown.
    """
    if iou_thresholds is None:
        iou_thresholds = [float(round(t, 2)) for t in np.arange(0.50, 0.96, 0.05)]

    classes = sorted(list(set(predictions_per_class.keys()) | set(ground_truth_per_class.keys())))

    class_ap_05 = {}
    class_prec_05 = {}
    class_rec_05 = {}
    class_aps_range = {c: [] for c in classes}

    for c in classes:
        pred_boxes, pred_scores = predictions_per_class.get(c, ([], []))
        gt_boxes = ground_truth_per_class.get(c, [])

        # mAP@0.5
        ap_05, prec_05, rec_05 = compute_class_ap(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)
        class_ap_05[c] = ap_05
        class_prec_05[c] = prec_05
        class_rec_05[c] = rec_05

        # All thresholds for COCO style mAP
        for t in iou_thresholds:
            ap_t, _, _ = compute_class_ap(pred_boxes, pred_scores, gt_boxes, iou_threshold=t)
            class_aps_range[c].append(ap_t)

    map_05 = float(np.mean(list(class_ap_05.values()))) if class_ap_05 else 0.0
    map_range = float(np.mean([np.mean(aps) for aps in class_aps_range.values()])) if class_aps_range else 0.0

    return {
        "mAP_0.5": map_05,
        "mAP_0.5:0.95": map_range,
        "per_class_AP_0.5": class_ap_05,
        "per_class_Precision_0.5": class_prec_05,
        "per_class_Recall_0.5": class_rec_05,
    }


def evaluate_detections(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int = 3,
) -> Dict[str, Any]:
    """Helper formatting multi-image batch predictions and targets for calculate_map."""
    preds_by_class: Dict[int, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {c: ([], []) for c in range(num_classes)}
    gts_by_class: Dict[int, List[torch.Tensor]] = {c: [] for c in range(num_classes)}

    for pred, target in zip(predictions, targets):
        for c in range(num_classes):
            # Target
            gt_mask = (target["labels"] == c)
            gts_by_class[c].append(target["boxes"][gt_mask].cpu())

            # Prediction
            pred_mask = (pred["labels"] == c)
            preds_by_class[c][0].append(pred["boxes"][pred_mask].cpu())
            preds_by_class[c][1].append(pred["scores"][pred_mask].cpu())

    return calculate_map(preds_by_class, gts_by_class)

