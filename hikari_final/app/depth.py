"""depth.py — MiDaS Depth Estimation · Hikari Shinro AI"""
import cv2
import numpy as np
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DepthEstimator:

    MODEL_TYPE = "MiDaS_small"

    def __init__(self, model_type=None):
        self.model_type = model_type or self.MODEL_TYPE
        self.model      = None
        self.transform  = None
        self.device     = None
        self._loaded    = False

    def load(self):
        if self._loaded:
            return
        try:
            logger.info(f"Loading MiDaS {self.model_type}...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model  = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
            self.model.to(self.device)
            self.model.eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = transforms.small_transform
            self._loaded = True
            logger.info("MiDaS loaded.")
        except Exception as e:
            logger.error(f"MiDaS load error: {e}")
            raise

    @torch.no_grad()
    def estimate(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        if not self._loaded:
            return None
        try:
            rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            batch = self.transform(rgb).to(self.device)
            pred  = self.model(batch)
            h, w  = frame_bgr.shape[:2]
            pred  = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=(h,w), mode="bicubic", align_corners=False
            ).squeeze()
            depth = pred.cpu().numpy().astype(np.float32)
            mn, mx = depth.min(), depth.max()
            if mx - mn > 1e-5:
                depth = (depth - mn) / (mx - mn)
            else:
                depth = np.zeros_like(depth)
            return depth
        except Exception as e:
            logger.warning(f"Depth error: {e}")
            return None

    def colorize(self, depth_map):
        if depth_map is None:
            return np.zeros((480,640,3), dtype=np.uint8)
        return cv2.applyColorMap((depth_map*255).astype(np.uint8), cv2.COLORMAP_PLASMA)

    def blend_overlay(self, frame, depth_map, alpha=0.3):
        if depth_map is None:
            return frame
        h, w  = frame.shape[:2]
        col   = cv2.resize(self.colorize(depth_map), (w,h))
        return cv2.addWeighted(frame, 1-alpha, col, alpha, 0)
