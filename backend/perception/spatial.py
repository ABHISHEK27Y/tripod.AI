import numpy as np


class SpatialEstimator:
    def __init__(self):
        pass

    def depth_to_range(self, mean_depth):
        """
        Convert relative depth → qualitative range
        Tunable thresholds (scene dependent)
        """

        if mean_depth > 0.65:
            return "near"
        elif mean_depth > 0.35:
            return "mid"
        else:
            return "far"

    def estimate(self, detections, depth_map):
        spatial_objects = []

        h, w = depth_map.shape

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            # clamp bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            region = depth_map[y1:y2, x1:x2]

            if region.size == 0:
                continue

            mean_depth = float(np.mean(region))

            # convert to qualitative distance
            distance_range = self.depth_to_range(mean_depth)

            # direction estimation
            cx = (x1 + x2) / 2
            rel_pos = cx / w

            if rel_pos < 0.33:
                direction = "left"
            elif rel_pos < 0.66:
                direction = "center"
            else:
                direction = "right"

            spatial_objects.append({
                "label": det["label"],
                "confidence": det["confidence"],
                "range": distance_range,
                "direction": direction,
            })

        return spatial_objects