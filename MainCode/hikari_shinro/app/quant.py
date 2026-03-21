"""
quant.py — Quantitative Engine
================================
Hikari Shinro AI · CodeRonin · Ahouba 3.0 · IIIT Manipur

Four explicit quantitative algorithms:
  ① Kalman Filter   — multi-object tracking across frames
  ② A* Pathfinding  — shortest safe route on depth-map grid
  ③ Euclidean Dist  — real-world distance from depth values
  ④ Conf. Threshold — statistical filtering of detections
"""

import numpy as np
import heapq
from scipy.linalg import inv
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────
# ① KALMAN FILTER — Object Tracker
# ─────────────────────────────────────────────────────────
class KalmanTracker:
    """
    Constant-velocity Kalman filter for tracking a single object.

    State vector x = [cx, cy, vx, vy]  (centre-x, centre-y, velocity-x, velocity-y)
    Measurement  z = [cx, cy]           (observed centre from bounding box)

    Predict:  x̂ₖ = F·xₖ₋₁         Pₖ = F·Pₖ₋₁·Fᵀ + Q
    Update:   Kₖ = Pₖ·Hᵀ·(H·Pₖ·Hᵀ + R)⁻¹
              x̂ₖ = x̂ₖ + Kₖ·(zₖ − H·x̂ₖ)
    """

    _id_counter = 0

    def __init__(self, cx: float, cy: float):
        KalmanTracker._id_counter += 1
        self.id = KalmanTracker._id_counter
        self.missed_frames = 0
        self.history: List[Tuple[float, float]] = []

        dt = 1.0  # time step (1 frame)

        # State transition matrix F (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)

        # Measurement matrix H (we observe cx, cy only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Process noise covariance Q
        self.Q = np.eye(4, dtype=float) * 0.1

        # Measurement noise covariance R
        self.R = np.eye(2, dtype=float) * 5.0

        # Initial state
        self.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=float)

        # Initial error covariance
        self.P = np.eye(4, dtype=float) * 500.0

    def predict(self) -> Tuple[float, float]:
        """Predict next position. Returns (cx, cy)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.history.append((float(self.x[0]), float(self.x[1])))
        if len(self.history) > 30:
            self.history.pop(0)
        return float(self.x[0]), float(self.x[1])

    def update(self, cx: float, cy: float):
        """Correct prediction with new measurement."""
        z = np.array([[cx], [cy]], dtype=float)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ inv(S)            # Kalman gain
        y = z - self.H @ self.x                   # innovation
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        self.missed_frames = 0

    def get_position(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])


class MultiObjectTracker:
    """
    Manages a pool of KalmanTrackers, one per detected object class.
    Matches detections to existing trackers by IoU / centroid distance.
    """

    MAX_MISSED = 8  # drop tracker after N missed frames

    def __init__(self):
        self.trackers: Dict[str, KalmanTracker] = {}   # key = label

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Args:
            detections: list of {label, cx, cy, w, h, conf, distance_m}
        Returns:
            enriched detections with tracker_id and trail (history)
        """
        active_labels = set()

        for det in detections:
            label = det["label"]
            cx, cy = det["cx"], det["cy"]
            active_labels.add(label)

            if label not in self.trackers:
                self.trackers[label] = KalmanTracker(cx, cy)
            else:
                self.trackers[label].predict()
                self.trackers[label].update(cx, cy)

            det["tracker_id"] = self.trackers[label].id
            det["trail"] = list(self.trackers[label].history[-10:])  # last 10 positions

        # Increment missed frames for absent trackers
        for label in list(self.trackers.keys()):
            if label not in active_labels:
                self.trackers[label].missed_frames += 1
                if self.trackers[label].missed_frames > self.MAX_MISSED:
                    del self.trackers[label]

        return detections


# ─────────────────────────────────────────────────────────
# ② A* PATHFINDING — Safe Navigation Grid
# ─────────────────────────────────────────────────────────
class AStarNavigator:
    """
    Builds a 2D obstacle grid from the MiDaS depth map.
    Cells where depth > OBSTACLE_THRESHOLD (i.e. objects close to camera)
    are marked as blocked. A* finds the shortest safe path from bottom-
    centre (user position) to the target object's cell.

    f(n) = g(n) + h(n)
    g(n) = actual cost from start
    h(n) = Manhattan distance heuristic to goal
    """

    GRID_W = 32
    GRID_H = 24
    OBSTACLE_DEPTH_THRESH = 0.72   # normalised depth (0=far, 1=close)
    OBSTACLE_RADIUS = 1            # expand obstacles by N cells for safety

    def build_grid(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Downsample depth map to GRID_W×GRID_H grid.
        Returns boolean grid: True = obstacle (blocked), False = free.
        """
        from PIL import Image
        h, w = depth_map.shape[:2]
        depth_norm = depth_map.astype(float)
        if depth_norm.max() > 0:
            depth_norm /= depth_norm.max()

        # Resize to grid
        pil = Image.fromarray((depth_norm * 255).astype(np.uint8))
        pil_small = pil.resize((self.GRID_W, self.GRID_H), Image.NEAREST)
        grid_vals = np.array(pil_small) / 255.0

        obstacles = grid_vals > self.OBSTACLE_DEPTH_THRESH

        # Expand obstacles (safety margin)
        expanded = obstacles.copy()
        for dy in range(-self.OBSTACLE_RADIUS, self.OBSTACLE_RADIUS + 1):
            for dx in range(-self.OBSTACLE_RADIUS, self.OBSTACLE_RADIUS + 1):
                shifted = np.roll(np.roll(obstacles, dy, axis=0), dx, axis=1)
                expanded |= shifted

        return expanded

    def find_path(
        self,
        depth_map: np.ndarray,
        target_cx_norm: float,
        target_cy_norm: float,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find path from (GRID_W//2, GRID_H-1) → target cell.

        Args:
            depth_map:      raw MiDaS depth array
            target_cx_norm: normalised x of target (0–1)
            target_cy_norm: normalised y of target (0–1)

        Returns:
            List of (col, row) waypoints, or None if no path found.
        """
        grid = self.build_grid(depth_map)

        start = (self.GRID_W // 2, self.GRID_H - 1)
        goal  = (
            min(int(target_cx_norm * self.GRID_W), self.GRID_W - 1),
            min(int(target_cy_norm * self.GRID_H), self.GRID_H - 1),
        )

        if grid[goal[1], goal[0]]:
            # Target cell is blocked — search nearest free cell
            for r in range(1, 5):
                for dc in range(-r, r + 1):
                    for dr in range(-r, r + 1):
                        nc = goal[0] + dc
                        nr = goal[1] + dr
                        if 0 <= nc < self.GRID_W and 0 <= nr < self.GRID_H:
                            if not grid[nr, nc]:
                                goal = (nc, nr)
                                break
                    else:
                        continue
                    break
                else:
                    continue
                break

        # A* search
        open_heap = []
        heapq.heappush(open_heap, (0, start))
        came_from: Dict[Tuple, Optional[Tuple]] = {start: None}
        g_cost: Dict[Tuple, float] = {start: 0.0}

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan

        neighbours_dirs = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current == goal:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for dc, dr in neighbours_dirs:
                nb = (current[0] + dc, current[1] + dr)
                if not (0 <= nb[0] < self.GRID_W and 0 <= nb[1] < self.GRID_H):
                    continue
                if grid[nb[1], nb[0]]:
                    continue

                step_cost = 1.0 if (dc == 0 or dr == 0) else 1.414
                new_g = g_cost[current] + step_cost

                if nb not in g_cost or new_g < g_cost[nb]:
                    g_cost[nb] = new_g
                    f = new_g + heuristic(nb, goal)
                    heapq.heappush(open_heap, (f, nb))
                    came_from[nb] = current

        return None  # no path found

    def path_to_instruction(
        self,
        path: Optional[List[Tuple[int, int]]],
    ) -> str:
        """Convert A* path to natural language navigation instruction."""
        if not path or len(path) < 2:
            return "path clear"

        start = path[0]
        end   = path[-1]
        mid   = path[len(path) // 2]

        dx = end[0] - start[0]
        dy = start[1] - end[1]   # y axis inverted (row 0 = top)

        steps = len(path)
        if steps < 5:
            dist_desc = "very close"
        elif steps < 12:
            dist_desc = "nearby"
        else:
            dist_desc = "ahead"

        if abs(dx) < 2:
            direction = "straight ahead"
        elif dx < 0:
            direction = "to your left"
        else:
            direction = "to your right"

        # Check for obstacle detour (significant lateral deviation in mid-path)
        mid_dx = mid[0] - start[0]
        if abs(mid_dx) > 4:
            detour_side = "left" if mid_dx < 0 else "right"
            return f"go {direction} — curve {detour_side} to avoid obstacle"

        return f"move {direction}"


# ─────────────────────────────────────────────────────────
# ③ EUCLIDEAN DISTANCE — Real-world distance to objects
# ─────────────────────────────────────────────────────────
class DistanceEstimator:
    """
    Converts MiDaS relative depth values to approximate metric distances.

    MiDaS outputs relative inverse depth (not metric). We use a linear
    calibration:  distance_m = DEPTH_SCALE / mean_depth_in_bbox

    DEPTH_SCALE is empirically set for typical indoor environments.
    For the hackathon demo, ~3.5 works well for laptop webcam at desk.
    """

    DEPTH_SCALE = 3.5    # tune this per environment

    def estimate_object_distance(
        self,
        depth_map: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
    ) -> float:
        """
        Euclidean distance to object: sqrt(dx² + dy² + dz²)
        Simplified to dz (depth) since we don't have full 3D calibration.

        Returns distance in metres (approximate).
        """
        h, w = depth_map.shape[:2]
        # Clip bounding box to frame
        x1c = max(0, min(x1, w - 1))
        y1c = max(0, min(y1, h - 1))
        x2c = max(x1c + 1, min(x2, w))
        y2c = max(y1c + 1, min(y2, h))

        region = depth_map[y1c:y2c, x1c:x2c].astype(float)
        if region.size == 0 or region.max() == 0:
            return 99.0

        region_norm = region / depth_map.max() if depth_map.max() > 0 else region

        # Use median of inner 50% to reject outliers
        flat = region_norm.flatten()
        flat.sort()
        inner = flat[len(flat) // 4 : 3 * len(flat) // 4]
        mean_depth = float(inner.mean()) if len(inner) > 0 else float(flat.mean())

        if mean_depth < 1e-4:
            return 99.0

        distance_m = self.DEPTH_SCALE / mean_depth
        return round(min(distance_m, 15.0), 2)   # cap at 15 m

    def distance_to_text(self, distance_m: float) -> str:
        """Human-readable distance label."""
        if distance_m < 0.5:
            return "very close — stop!"
        if distance_m < 1.0:
            return f"{distance_m:.1f}m — caution"
        if distance_m < 3.0:
            return f"{distance_m:.1f}m ahead"
        return f"{distance_m:.0f}m away"


# ─────────────────────────────────────────────────────────
# ④ CONFIDENCE THRESHOLDING
# ─────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.40   # detections below this are discarded

def filter_by_confidence(detections: List[Dict], threshold: float = CONFIDENCE_THRESHOLD) -> List[Dict]:
    """
    Statistical filtering: only pass detections where
    P(object | features) ≥ threshold.

    Also deduplicates overlapping detections of the same label
    (keeps highest confidence one).
    """
    passed = [d for d in detections if d.get("conf", 0) >= threshold]

    # Deduplicate same-label detections (keep highest conf)
    best: Dict[str, Dict] = {}
    for d in passed:
        label = d["label"]
        if label not in best or d["conf"] > best[label]["conf"]:
            best[label] = d

    return list(best.values())


# ─────────────────────────────────────────────────────────
# SCENE DESCRIPTION BUILDER
# ─────────────────────────────────────────────────────────
def build_scene_description(detections: List[Dict], path_instruction: str) -> str:
    """
    Convert detection list + path into a single scene description string
    that the LLM agent can reason about.
    """
    if not detections:
        return f"No objects detected. {path_instruction}."

    parts = []
    for det in sorted(detections, key=lambda d: d.get("distance_m", 99)):
        label = det["label"]
        dist  = det.get("distance_m", "?")
        cx    = det.get("cx_norm", 0.5)

        if cx < 0.35:
            side = "left"
        elif cx > 0.65:
            side = "right"
        else:
            side = "centre"

        parts.append(f"{label} at {dist}m ({side})")

    scene = ", ".join(parts)
    return f"Objects detected: {scene}. Navigation: {path_instruction}."


# ─────────────────────────────────────────────────────────
# MAIN QUANT ENGINE — ties all four components together
# ─────────────────────────────────────────────────────────
class QuantEngine:
    def __init__(self):
        self.tracker   = MultiObjectTracker()
        self.navigator = AStarNavigator()
        self.estimator = DistanceEstimator()

    def process(
        self,
        raw_detections: List[Dict],
        depth_map: Optional[np.ndarray],
        target_label: str = "",
        frame_w: int = 640,
        frame_h: int = 480,
    ) -> Tuple[List[Dict], str, Optional[List]]:
        """
        Full quant pipeline for one frame.

        Returns:
            enriched_detections — with distance, trail, tracker_id
            scene_description   — text string for LLM agent
            astar_path          — list of (col,row) grid waypoints or None
        """
        # ④ Confidence filter
        filtered = filter_by_confidence(raw_detections)

        # ③ Distance estimation
        for det in filtered:
            if depth_map is not None:
                det["distance_m"] = self.estimator.estimate_object_distance(
                    depth_map,
                    det["x1"], det["y1"], det["x2"], det["y2"],
                )
                det["distance_text"] = self.estimator.distance_to_text(det["distance_m"])
            else:
                det["distance_m"]   = 99.0
                det["distance_text"] = "unknown distance"

            # Normalised centre (for A* and side classification)
            det["cx_norm"] = (det["cx"] / frame_w) if frame_w > 0 else 0.5
            det["cy_norm"] = (det["cy"] / frame_h) if frame_h > 0 else 0.5

        # ① Kalman tracking
        enriched = self.tracker.update(filtered)

        # ② A* pathfinding
        astar_path = None
        path_instruction = "path clear, proceed"

        if depth_map is not None:
            # Find target object
            target = next((d for d in enriched if d["label"] == target_label), None)
            if target is None and enriched:
                target = min(enriched, key=lambda d: d.get("distance_m", 99))

            if target:
                astar_path = self.navigator.find_path(
                    depth_map,
                    target["cx_norm"],
                    target["cy_norm"],
                )
                path_instruction = self.navigator.path_to_instruction(astar_path)

        scene = build_scene_description(enriched, path_instruction)
        return enriched, scene, astar_path
