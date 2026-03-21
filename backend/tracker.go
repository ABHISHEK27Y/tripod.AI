import numpy as np


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter_area + 1e-6

    return inter_area / union


class Track:
    def __init__(self, bbox, label, track_id):
        self.bbox = bbox
        self.label = label
        self.id = track_id
        self.age = 0  # frames since last match


class IOUTracker:
    def __init__(self, iou_threshold=0.3, max_age=10):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        updated_tracks = []
        assigned = set()

        for track in self.tracks:
            best_iou = 0
            best_det = None
            best_idx = -1

            for i, det in enumerate(detections):
                if i in assigned:
                    continue

                if det["label"] != track.label:
                    continue

                score = iou(track.bbox, det["bbox"])

                if score > best_iou:
                    best_iou = score
                    best_det = det
                    best_idx = i

            if best_iou > self.iou_threshold:
                track.bbox = best_det["bbox"]
                track.age = 0
                updated_tracks.append(track)
                assigned.add(best_idx)
            else:
                track.age += 1
                if track.age < self.max_age:
                    updated_tracks.append(track)

        # create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in assigned:
                updated_tracks.append(
                    Track(det["bbox"], det["label"], self.next_id)
                )
                self.next_id += 1

        self.tracks = updated_tracks

        return self.tracks