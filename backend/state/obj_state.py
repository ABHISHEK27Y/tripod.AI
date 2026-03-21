from collections import deque
import time


class ObjectState:
    def __init__(self, object_id, label, bbox, range_label, direction):
        self.id = object_id
        self.label = label

        # latest observation
        self.bbox = bbox
        self.range = range_label
        self.direction = direction

        # temporal properties
        self.last_seen = time.time()
        self.first_seen = self.last_seen

        # short history buffer (for smoothing later)
        self.range_history = deque(maxlen=5)
        self.range_history.append(range_label)

        self.direction_history = deque(maxlen=5)
        self.direction_history.append(direction)

        # derived future fields
        self.smoothed_range = range_label
        self.velocity = None   # placeholder for Task-3.5+

    def update(self, bbox, range_label, direction):
        """
        Update object state with new observation
        """
        self.bbox = bbox
        self.range = range_label
        self.direction = direction
        self.last_seen = time.time()

        self.range_history.append(range_label)
        self.direction_history.append(direction)

        self._smooth_range()

    def _smooth_range(self):
        """
        Simple majority smoothing for qualitative depth
        """
        counts = {}
        for r in self.range_history:
            counts[r] = counts.get(r, 0) + 1

        self.smoothed_range = max(counts, key=counts.get)

    def is_stale(self, timeout=1.0):
        """
        If object not seen recently → remove from world
        """
        return (time.time() - self.last_seen) > timeout

    def to_dict(self):
        """
        Export stable representation for navigation layer
        """
        return {
            "id": self.id,
            "label": self.label,
            "bbox": self.bbox,
            "range": self.smoothed_range,
            "direction": self.direction,
        }