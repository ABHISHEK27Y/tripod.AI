import cv2
import time

from perception.camera import Camera
from perception.detector import Detector
from perception.depth import DepthEstimator
from perception.spatial import SpatialEstimator
from perception.tracker import IOUTracker


def draw(frame, tracks, spatial_objects):
    for track, obj in zip(tracks, spatial_objects):
        x1, y1, x2, y2 = track.bbox

        text = f"ID {track.id} {obj['label']} {obj['range']} {obj['direction']}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


def main():
    cam = Camera()
    det = Detector()
    depth = DepthEstimator()
    spatial = SpatialEstimator()
    tracker = IOUTracker()

    cam.open()

    prev = time.time()

    try:
        while True:
            frame = cam.read()
            if frame is None:
                break

            detections = det.detect(frame)
            tracks = tracker.update(detections)

            # build fake detection list from tracks for spatial
            tracked_dets = [
                {"bbox": t.bbox, "label": t.label, "confidence": 1.0}
                for t in tracks
            ]

            depth_map = depth.estimate(frame)
            spatial_objects = spatial.estimate(tracked_dets, depth_map)

            for t, obj in zip(tracks, spatial_objects):
                print(
                    f"ID {t.id} → {obj['label']} → {obj['range']} → {obj['direction']}"
                )

            draw(frame, tracks, spatial_objects)

            now = time.time()
            fps = 1 / (now - prev)
            prev = now

            cv2.putText(
                frame,
                f"FPS {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            cv2.imshow("Tracking Vision", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cam.release()


if __name__ == "__main__":
    main()