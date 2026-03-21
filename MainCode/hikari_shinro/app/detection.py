"""
detection.py — YOLO-World Open-Vocabulary Object Detection
============================================================
Hikari Shinro AI · CodeRonin · Ahouba 3.0 · IIIT Manipur

Uses YOLO-World (Ultralytics) for zero-shot open-vocabulary detection.
New objects are detected by passing text labels — zero retraining required.
Directly satisfies PS1: "new objects can be added without training".
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Default labels — LLM agent dynamically overrides these every frame
DEFAULT_LABELS = [
    "door", "chair", "table", "person", "steps", "stairs",
    "wall", "obstacle", "cup", "bag", "laptop", "phone",
]


class YOLOWorldDetector:
    """
    Wraps Ultralytics YOLO-World for open-vocabulary detection.
    Model is loaded once at startup and reused for all frames.
    Text labels are swapped per-frame with zero overhead.
    """

    def __init__(self, model_size: str = "yolov8s-worldv2"):
        """
        Args:
            model_size: 'yolov8s-worldv2' (fast, ~50MB) or
                        'yolov8m-worldv2' (better accuracy, ~100MB)
        """
        self.model_size  = model_size
        self.model       = None
        self.current_labels: List[str] = DEFAULT_LABELS[:]
        self._loaded     = False

    def load(self):
        """Load model — call once at startup."""
        if self._loaded:
            return
        try:
            from ultralytics import YOLOWorld
            logger.info(f"Loading YOLO-World model: {self.model_size} ...")
            self.model = YOLOWorld(self.model_size)
            self.set_labels(DEFAULT_LABELS)
            self._loaded = True
            logger.info("YOLO-World loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO-World: {e}")
            raise

    def set_labels(self, labels: List[str]):
        """
        Hot-swap detection labels — no retraining needed.
        Called every time LLM agent updates detect_next list.
        """
        if not labels:
            return
        clean = [l.strip().lower() for l in labels if l.strip()]
        if clean == self.current_labels:
            return
        self.current_labels = clean
        if self.model is not None:
            self.model.set_classes(clean)
            logger.debug(f"Labels updated: {clean}")

    def detect(
        self,
        frame: np.ndarray,
        labels: Optional[List[str]] = None,
        conf_threshold: float = 0.15,
        imgsz: int = 640,
    ) -> List[Dict]:
        """
        Run detection on a single BGR frame.

        Args:
            frame:          BGR numpy array from OpenCV
            labels:         optional label override for this frame
            conf_threshold: minimum confidence to include
            imgsz:          inference size (lower = faster)

        Returns:
            List of detection dicts:
            {
                label, conf,
                x1, y1, x2, y2,   # pixel coords (int)
                cx, cy,            # centre pixel (int)
                w, h,              # box size (int)
            }
        """
        if not self._loaded or self.model is None:
            return []

        if labels is not None:
            self.set_labels(labels)

        try:
            results = self.model.predict(
                frame,
                imgsz=imgsz,
                conf=conf_threshold,
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"YOLO detection error: {e}")
            return []

        detections = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            try:
                xyxy  = boxes.xyxy[i].cpu().numpy().astype(int)
                conf  = float(boxes.conf[i].cpu())
                cls   = int(boxes.cls[i].cpu())

                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                label = self.current_labels[cls] if cls < len(self.current_labels) else "unknown"

                detections.append({
                    "label": label,
                    "conf":  round(conf, 3),
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "cx": (x1 + x2) // 2,
                    "cy": (y1 + y2) // 2,
                    "w":  x2 - x1,
                    "h":  y2 - y1,
                })
            except Exception:
                continue

        return detections

    def draw_boxes(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        trails: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes + labels + trails on frame.
        Returns annotated copy of frame.
        """
        out = frame.copy()

        # Color map per label (consistent across frames)
        COLOR_PALETTE = [
            (0, 212, 255),   # cyan
            (255, 160, 32),  # amber
            (46, 204, 113),  # green
            (155, 89, 182),  # purple
            (231, 76,  60),  # red
            (52, 152, 219),  # blue
        ]

        label_colors: Dict[str, tuple] = {}
        for det in detections:
            lbl = det["label"]
            if lbl not in label_colors:
                idx = len(label_colors) % len(COLOR_PALETTE)
                label_colors[lbl] = COLOR_PALETTE[idx]

        for det in detections:
            color = label_colors.get(det["label"], (0, 212, 255))
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

            # Bounding box — 2px thick
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Corner accents (anime HUD style)
            corner_len = 12
            for cx_, cy_, dx, dy in [
                (x1, y1,  1,  1),
                (x2, y1, -1,  1),
                (x1, y2,  1, -1),
                (x2, y2, -1, -1),
            ]:
                cv2.line(out, (cx_, cy_), (cx_ + dx * corner_len, cy_), color, 2)
                cv2.line(out, (cx_, cy_), (cx_, cy_ + dy * corner_len), color, 2)

            # Label background + text
            label_text = f"{det['label']}  {det.get('distance_text', '')}"
            conf_text  = f"{det['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(out, label_text, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

            # Kalman tracking trail
            if trails and "trail" in det and len(det["trail"]) > 1:
                trail = det["trail"]
                for j in range(1, len(trail)):
                    p1 = (int(trail[j-1][0]), int(trail[j-1][1]))
                    p2 = (int(trail[j][0]),   int(trail[j][1]))
                    alpha = j / len(trail)
                    cv2.line(out, p1, p2, color, max(1, int(alpha * 2)))

        return out

    def overlay_astar_path(
        self,
        frame: np.ndarray,
        path,
        grid_w: int = 32,
        grid_h: int = 24,
    ) -> np.ndarray:
        """Draw A* path dots on frame."""
        if path is None or len(path) < 2:
            return frame
        out   = frame.copy()
        fh, fw = frame.shape[:2]
        cell_w = fw / grid_w
        cell_h = fh / grid_h

        for i, (col, row) in enumerate(path):
            px = int((col + 0.5) * cell_w)
            py = int((row + 0.5) * cell_h)
            alpha = (i / len(path))
            radius = max(3, int(6 * alpha))
            color = (0, int(200 * alpha), 50 + int(100 * alpha))
            cv2.circle(out, (px, py), radius, color, -1)
            if i > 0:
                prev_col, prev_row = path[i - 1]
                p1 = (int((prev_col + 0.5) * cell_w), int((prev_row + 0.5) * cell_h))
                cv2.line(out, p1, (px, py), (0, 180, 80), 1)

        return out
