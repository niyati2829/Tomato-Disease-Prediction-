"""Visual Overlay Annotator for Tomato Field Surveillance Video.

Draws professional HUD elements:
- Top telemetry bar (timestamp, frame index, pipeline status).
- Pathology diagnosis badge (calibrated disease classification & confidence bar).
- Pest surveillance counter (Tuta absoluta leafminer mine instances).
- Bounding boxes for detected pest lesions.
- Academic status badge ('Lesion Segmentation: Blocked by Dataset Availability').
"""

from typing import Tuple

import cv2
import numpy as np

from src.surveillance.frame_analyzer import FrameAnalysisResult


class VideoAnnotator:
    """Renders real-time HUD annotations onto video frames."""

    def __init__(self) -> None:
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.primary_color = (0, 204, 255)  # Gold/Yellow
        self.healthy_color = (46, 204, 113)  # Emerald Green
        self.disease_color = (68, 68, 238)  # Coral Red
        self.pest_color = (255, 102, 0)  # Bright Orange/Blue (BGR: Blue=0, Green=102, Red=255)

    def annotate(self, frame_bgr: np.ndarray, result: FrameAnalysisResult) -> np.ndarray:
        """Annotate frame with diagnosis, pest boxes, and telemetry."""
        canvas = frame_bgr.copy()
        H, W = canvas.shape[:2]

        # 1. Top Telemetry Banner
        banner_h = 56
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (W, banner_h), (20, 24, 28), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        # Title & Timestamp
        cv2.putText(
            canvas,
            "TOMATO SURVEILLANCE PIPELINE (PHASE G)",
            (14, 22),
            self.font,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        time_text = f"Frame: {result.frame_idx:04d} | Time: {result.timestamp_seconds:.2f}s"
        cv2.putText(
            canvas,
            time_text,
            (14, 44),
            self.font,
            0.45,
            (180, 190, 200),
            1,
            cv2.LINE_AA,
        )

        # Right-side Live Status
        status_text = "REC | REAL-TIME"
        cv2.circle(canvas, (W - 140, 28), 6, (0, 0, 240), -1)
        cv2.putText(
            canvas,
            status_text,
            (W - 125, 33),
            self.font,
            0.48,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        # 2. Bottom-Left Pathology Badge
        badge_w, badge_h = 320, 84
        bx, by = 14, H - badge_h - 14

        overlay = canvas.copy()
        cv2.rectangle(overlay, (bx, by), (bx + badge_w, by + badge_h), (18, 22, 26), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
        cv2.rectangle(canvas, (bx, by), (bx + badge_w, by + badge_h), (60, 70, 80), 1)

        is_healthy = "healthy" in result.disease_label.lower()
        badge_accent = self.healthy_color if is_healthy else self.disease_color

        cv2.rectangle(canvas, (bx, by), (bx + 6, by + badge_h), badge_accent, -1)

        # Disease Label & Calibrated Confidence
        cv2.putText(
            canvas,
            f"Diagnosis: {result.disease_label}",
            (bx + 16, by + 24),
            self.font,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        conf_pct = result.disease_confidence * 100.0
        cv2.putText(
            canvas,
            f"Calibrated Confidence (T*=1.30): {conf_pct:.1f}%",
            (bx + 16, by + 46),
            self.font,
            0.42,
            (200, 210, 220),
            1,
            cv2.LINE_AA,
        )

        # Confidence Bar
        bar_x, bar_y, bar_w, bar_h = bx + 16, by + 56, 280, 10
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 55, 60), -1)
        fill_w = int(bar_w * np.clip(result.disease_confidence, 0.0, 1.0))
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), badge_accent, -1)

        # 3. Bottom-Right Pest & Status Badge
        p_badge_w, p_badge_h = 300, 84
        px, py = W - p_badge_w - 14, H - p_badge_h - 14

        overlay = canvas.copy()
        cv2.rectangle(overlay, (px, py), (px + p_badge_w, py + p_badge_h), (18, 22, 26), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
        cv2.rectangle(canvas, (px, py), (px + p_badge_w, py + p_badge_h), (60, 70, 80), 1)

        pest_accent = (0, 165, 255) if result.pest_count > 0 else (120, 130, 140)
        cv2.rectangle(canvas, (px, py), (px + 6, py + p_badge_h), pest_accent, -1)

        det_type = getattr(result, "pest_detector_type", "tinyyolo")
        if det_type == "yolov8n-seg":
            pest_title = f"Tuta Damage: {result.pest_count} detected"
            model_ref = "Model: ResNet18 + YOLOv8n-Seg"
        else:
            pest_title = f"Pests (Tuta absoluta): {result.pest_count} detected"
            model_ref = "Model: ResNet18 + TinyYOLODetector"

        cv2.putText(
            canvas,
            pest_title,
            (px + 16, py + 24),
            self.font,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Status line (Academic transparency)
        status_line = "Lesion Mask: PlantSeg U-Net" if "unet" in getattr(result, "segmentation_status", "").lower() else "Lesion Mask: Blocked (No GT)"
        cv2.putText(
            canvas,
            status_line,
            (px + 16, py + 48),
            self.font,
            0.40,
            (160, 170, 180),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            model_ref,
            (px + 16, py + 68),
            self.font,
            0.38,
            (130, 140, 150),
            1,
            cv2.LINE_AA,
        )

        # 4. Render Pest Segmentation Masks (if available from YOLOv8n-seg)
        pest_masks = getattr(result, "pest_masks", None)
        if pest_masks:
            mask_overlay = canvas.copy()
            for mask, label in zip(pest_masks, result.pest_labels):
                if mask is not None and len(mask) >= 3:
                    pts = np.asarray(mask, dtype=np.int32).reshape((-1, 1, 2))
                    fill_color = (0, 0, 230) if "tuta" in label.lower() else (0, 180, 50)
                    cv2.fillPoly(mask_overlay, [pts], fill_color)
            cv2.addWeighted(mask_overlay, 0.35, canvas, 0.65, 0, canvas)

            # Crisp contour boundary for maximum visual clarity
            for mask, label in zip(pest_masks, result.pest_labels):
                if mask is not None and len(mask) >= 3:
                    pts = np.asarray(mask, dtype=np.int32).reshape((-1, 1, 2))
                    line_color = (0, 50, 255) if "tuta" in label.lower() else (0, 220, 80)
                    cv2.polylines(canvas, [pts], isClosed=True, color=line_color, thickness=2)

        # 5. Render Pest Bounding Boxes
        for box, score, label in zip(result.pest_boxes, result.pest_scores, result.pest_labels):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)

            is_tuta = "tuta" in label.lower()
            box_color = (0, 80, 255) if is_tuta else (0, 220, 100)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

            display_tag = f"Tuta Damage {score:.2f}" if is_tuta else f"Healthy Foliage {score:.2f}"
            (tw, th), _ = cv2.getTextSize(display_tag, self.font, 0.42, 1)
            cv2.rectangle(
                canvas,
                (x1, max(0, y1 - th - 6)),
                (x1 + tw + 8, y1),
                box_color,
                -1,
            )
            cv2.putText(
                canvas,
                display_tag,
                (x1 + 4, max(10, y1 - 4)),
                self.font,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return canvas
