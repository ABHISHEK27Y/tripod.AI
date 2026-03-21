import time
from state.obj_state import ObjectState


class WorldState:
    def __init__(self, stale_timeout=1.0):
        self.objects = {}   # id → ObjectState
        self.stale_timeout = stale_timeout

        # derived environment signals
        self.zone_clearance = {"left": "far", "center": "far", "right": "far"}
        self.collision_risk = False
        self.target_visible = False

    def update(self, tracked_spatial_objects, target_label="door"):
        """
        tracked_spatial_objects = [
            {id, label, bbox, range, direction}
        ]
        """

        now = time.time()

        # --- update or create object states ---
        for obj in tracked_spatial_objects:
            oid = obj["id"]

            if oid not in self.objects:
                self.objects[oid] = ObjectState(
                    oid,
                    obj["label"],
                    obj["bbox"],
                    obj["range"],
                    obj["direction"],
                )
            else:
                self.objects[oid].update(
                    obj["bbox"],
                    obj["range"],
                    obj["direction"],
                )

        # --- CRITICAL FIX: if no observations in current frame, clear ALL objects ---
        # This handles vision loss (e.g., camera covered) immediately instead of waiting for stale_timeout
        if len(tracked_spatial_objects) == 0:
            self.objects.clear()
        else:
            # --- remove stale objects (only if we have current observations) ---
            stale_ids = [
                oid for oid, o in self.objects.items()
                if o.is_stale(self.stale_timeout)
            ]
            for oid in stale_ids:
                del self.objects[oid]

        # --- recompute derived signals ---
        self._compute_zone_clearance()
        self._compute_collision()
        self._compute_target_visibility(target_label)

    def _compute_zone_clearance(self):
        """
        Clearance is qualitative:
        near < mid < far
        We choose worst (closest obstacle) per zone.
        """
        clearance_rank = {"near": 0, "mid": 1, "far": 2}

        zone_scores = {"left": 2, "center": 2, "right": 2}

        for o in self.objects.values():
            r = o.smoothed_range
            z = o.direction
            zone_scores[z] = min(zone_scores[z], clearance_rank[r])

        # convert back to label
        inv = {0: "near", 1: "mid", 2: "far"}
        self.zone_clearance = {z: inv[s] for z, s in zone_scores.items()}

    def _compute_collision(self):
        """
        Collision if any object in near range.
        """
        self.collision_risk = any(
            o.smoothed_range == "near" for o in self.objects.values()
        )

    def _compute_target_visibility(self, target_label):
        self.target_visible = any(
            o.label == target_label for o in self.objects.values()
        )

    def get_target_objects(self, target_label):
        return [
            o.to_dict()
            for o in self.objects.values()
            if o.label == target_label
        ]

    def snapshot(self):
        """
        Export stable world representation for navigation layer.
        """
        return {
            "objects": [o.to_dict() for o in self.objects.values()],
            "zone_clearance": self.zone_clearance,
            "collision_risk": self.collision_risk,
            "target_visible": self.target_visible,
        }