import cv2
import torch
import numpy as np


class DepthEstimator:
    def __init__(self, model_type="MiDaS_small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load MiDaS
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.to(self.device)
        self.midas.eval()

        # load transforms
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "MiDaS_small":
            self.transform = transforms.small_transform
        else:
            self.transform = transforms.dpt_transform

    def estimate(self, frame):
        """
        input: BGR frame (opencv)
        output: depth_map (float32 numpy, normalized)
        """

        # convert BGR -> RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # normalize depth for stability
        depth_map = (depth_map - depth_map.min()) / (
            depth_map.max() - depth_map.min() + 1e-6
        )

        return depth_map