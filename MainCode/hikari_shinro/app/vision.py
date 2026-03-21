"""
vision.py — Camera Capture + Frame Pipeline
============================================
Hikari Shinro AI · CodeRonin · Ahouba 3.0 · IIIT Manipur

Manages webcam capture, frame preprocessing, and the
main vision processing loop that ties detection + depth together.
"""

import cv2
import numpy as np
import threading
import time
import base64
import logging
from typing import Optional, Callable, Tuple

logger = logging.getLogger(__name__)

# Target frame dimensions
FRAME_W = 640
FRAME_H = 480
TARGET_FPS = 12


class CameraCapture:
    """
    Thread-safe webcam capture. Reads frames in background thread
    so the main pipeline always gets the most recent frame.
    """

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._cap      = None
        self._frame    = None
        self._lock     = threading.Lock()
        self._running  = False
        self._thread   = None

    def start(self) -> bool:
        """Open camera and start capture thread. Returns True on success."""
        self._cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            logger.error(f"Cannot open camera device {self.device_id}")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self._cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Camera capture started.")
        return True

    def _capture_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                resized = cv2.resize(frame, (FRAME_W, FRAME_H))
                with self._lock:
                    self._frame = resized
            time.sleep(1.0 / (TARGET_FPS * 2))

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()

    def is_running(self) -> bool:
        return self._running and self._cap is not None and self._cap.isOpened()


class FramePipeline:
    """
    Runs detection + depth + quant on each camera frame.
    Emits annotated JPEG frames via callback for Socket.IO streaming.
    """

    JPEG_QUALITY   = 70     # lower = faster transfer
    DEPTH_ALPHA    = 0.3    # depth overlay transparency

    def __init__(
        self,
        detector,
        depth_estimator,
        quant_engine,
        on_frame: Callable[[str, dict], None],  # (jpeg_b64, metadata)
    ):
        self.detector    = detector
        self.depth       = depth_estimator
        self.quant       = quant_engine
        self.on_frame    = on_frame

        self._target_labels = []
        self._lock          = threading.Lock()
        self._running       = False

    def set_labels(self, labels: list):
        with self._lock:
            self._target_labels = labels[:]
            self.detector.set_labels(labels)

    def process_frame(
        self,
        frame: np.ndarray,
        target_label: str = "",
    ) -> Tuple[np.ndarray, dict]:
        """
        Full single-frame pipeline:
          1. YOLO-World detection
          2. MiDaS depth estimation (parallel in same call — sequential here for simplicity)
          3. Quant engine (Kalman + A* + distance)
          4. Annotate frame

        Returns:
            annotated_frame (BGR numpy array)
            metadata dict
        """
        h, w = frame.shape[:2]

        with self._lock:
            labels = self._target_labels[:]

        # ── 1. DETECTION ──────────────────────────────────
        raw_detections = self.detector.detect(frame, labels=labels or None)

        # ── 2. DEPTH ──────────────────────────────────────
        depth_map = self.depth.estimate(frame)

        # ── 3. QUANT ENGINE ───────────────────────────────
        enriched, scene_desc, astar_path = self.quant.process(
            raw_detections,
            depth_map,
            target_label=target_label,
            frame_w=w,
            frame_h=h,
        )

        # ── 4. ANNOTATE ───────────────────────────────────
        annotated = frame.copy()

        # Depth overlay (subtle)
        if depth_map is not None:
            annotated = self.depth.blend_overlay(annotated, depth_map, alpha=self.DEPTH_ALPHA)

        # Bounding boxes + Kalman trails
        annotated = self.detector.draw_boxes(annotated, enriched, trails=True)

        # A* path overlay
        if astar_path:
            annotated = self.detector.overlay_astar_path(annotated, astar_path)

        # HUD overlay text
        annotated = self._draw_hud(annotated, enriched, scene_desc, target_label)

        metadata = {
            "detections":    enriched,
            "scene":         scene_desc,
            "astar_path":    astar_path,
            "depth_available": depth_map is not None,
            "target_label":  target_label,
            "object_count":  len(enriched),
        }

        return annotated, metadata

    def _draw_hud(
        self,
        frame: np.ndarray,
        detections: list,
        scene: str,
        target: str,
    ) -> np.ndarray:
        """Draw anime-styled HUD overlay on frame."""
        h, w = frame.shape[:2]
        out  = frame.copy()

        # Top HUD bar
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 28), (13, 17, 23), -1)
        cv2.addWeighted(overlay, 0.8, out, 0.2, 0, out)
        cv2.putText(out, "視覚  HIKARI SHINRO AI", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 210, 220), 1, cv2.LINE_AA)
        cv2.putText(out, f"OBJ:{len(detections)}  TARGET:{target or 'none'}", (w - 200, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        # Corner brackets (HUD aesthetic)
        bracket_len, bracket_w = 18, 2
        bracket_color = (0, 210, 220)
        for bx, by, sx, sy in [(0, 0, 1, 1), (w, 0, -1, 1), (0, h, 1, -1), (w, h, -1, -1)]:
            cv2.line(out, (bx, by), (bx + sx * bracket_len, by), bracket_color, bracket_w)
            cv2.line(out, (bx, by), (bx, by + sy * bracket_len), bracket_color, bracket_w)

        # Bottom scene description bar
        scene_short = scene[:90] + "..." if len(scene) > 90 else scene
        overlay2 = out.copy()
        cv2.rectangle(overlay2, (0, h - 30), (w, h), (13, 17, 23), -1)
        cv2.addWeighted(overlay2, 0.82, out, 0.18, 0, out)
        cv2.putText(out, scene_short, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        return out

    def frame_to_jpeg_b64(self, frame: np.ndarray) -> str:
        """Encode BGR frame to base64 JPEG string for Socket.IO."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY]
        _, buffer = cv2.imencode(".jpg", frame, encode_params)
        return base64.b64encode(buffer).decode("utf-8")

    def emit_frame(self, frame: np.ndarray, metadata: dict):
        """Encode and emit frame via callback."""
        jpeg_b64 = self.frame_to_jpeg_b64(frame)
        self.on_frame(jpeg_b64, metadata)
