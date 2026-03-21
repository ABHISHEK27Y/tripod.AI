"""
depth.py — MiDaS Monocular Depth Estimation
=============================================
Hikari Shinro AI · CodeRonin · Ahouba 3.0 · IIIT Manipur

Single RGB frame → relative depth map in milliseconds.
Satisfies PS1: "Depth estimation (Monocular or Stereo)".
"""

import cv2
import numpy as np
import torch
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class DepthEstimator:
    """
    Wraps Intel ISL MiDaS (DPT or MiDaS Small) for monocular depth.
    Loads model once at startup, runs inference per frame.

    Output: depth map array (H×W float32) — higher value = closer to camera.
    """

    # Model options (trade accuracy for speed):
    #   "DPT_Large"    — most accurate, ~430MB, slow
    #   "DPT_Hybrid"   — balanced, ~320MB
    #   "MiDaS_small"  — fastest, ~83MB — best for hackathon demo
    MODEL_TYPE = "MiDaS_small"

    def __init__(self, model_type: str = None):
        self.model_type = model_type or self.MODEL_TYPE
        self.model      = None
        self.transform  = None
        self.device     = None
        self._loaded    = False

    def load(self):
        """Load MiDaS model. Call once at startup."""
        if self._loaded:
            return
        try:
            logger.info(f"Loading MiDaS model: {self.model_type} ...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            self.model = torch.hub.load(
                "intel-isl/MiDaS",
                self.model_type,
                trust_repo=True,
            )
            self.model.to(self.device)
            self.model.eval()

            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS",
                "transforms",
                trust_repo=True,
            )
            if self.model_type in ("DPT_Large", "DPT_Hybrid"):
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

            self._loaded = True
            logger.info("MiDaS loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load MiDaS: {e}")
            raise

    @torch.no_grad()
    def estimate(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Run depth estimation on a BGR OpenCV frame.

        Returns:
            depth_map: float32 array (same H×W as input frame),
                       values normalised 0–1 (1 = closest to camera).
            Returns None on failure.
        """
        if not self._loaded or self.model is None:
            return None

        try:
            # MiDaS expects RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Apply model-specific transform
            input_batch = self.transform(frame_rgb).to(self.device)

            # Inference
            prediction = self.model(input_batch)

            # Upsample back to original frame size
            h, w = frame_bgr.shape[:2]
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth = prediction.cpu().numpy().astype(np.float32)

            # Normalise to 0–1 (higher = closer to camera)
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-5:
                depth = (depth - d_min) / (d_max - d_min)
            else:
                depth = np.zeros_like(depth)

            return depth

        except Exception as e:
            logger.warning(f"Depth estimation error: {e}")
            return None

    def colorize(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Convert normalised depth map to a colourized BGR image
        for overlay on the HUD.

        Colour scheme:
          Red/hot = close (high depth value)
          Blue/cool = far  (low depth value)
        """
        if depth_map is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        depth_uint8 = (depth_map * 255).astype(np.uint8)
        # COLORMAP_PLASMA: purple=far, yellow=close — visually intuitive
        colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
        return colored

    def blend_overlay(
        self,
        frame: np.ndarray,
        depth_map: np.ndarray,
        alpha: float = 0.35,
    ) -> np.ndarray:
        """
        Blend depth colormap onto the camera frame at given alpha.
        Returns annotated frame.
        """
        if depth_map is None:
            return frame
        colored   = self.colorize(depth_map)
        h, w      = frame.shape[:2]
        colored_r = cv2.resize(colored, (w, h))
        overlay   = cv2.addWeighted(frame, 1 - alpha, colored_r, alpha, 0)
        return overlay

    def get_horizon_line(self, depth_map: np.ndarray) -> int:
        """
        Estimate the 'obstacle horizon' — the row below which
        obstacles are likely in the walking path.
        Returns row index as integer.
        """
        if depth_map is None:
            return depth_map.shape[0] // 2 if depth_map is not None else 240
        h = depth_map.shape[0]
        # Average depth in bottom third of frame (floor area)
        bottom_mean = depth_map[2 * h // 3:, :].mean()
        # Horizon: first row where mean depth > 60% of floor depth
        for row in range(h // 3, 2 * h // 3):
            if depth_map[row, :].mean() > bottom_mean * 0.6:
                return row
        return h // 2
