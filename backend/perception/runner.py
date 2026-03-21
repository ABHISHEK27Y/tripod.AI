import cv2
import time

from perception.camera import Camera
from perception.detector import Detector
from perception.depth import DepthEstimator
from perception.spatial import SpatialEstimator
from perception.tracker import IOUTracker

from state.world_state import WorldState
from state.navigation_state import NavigationState
from navigation.navigator import NavigatorFSM


def draw(frame, world_snapshot, nav_snapshot, decision):
    # draw objects
    for obj in world_snapshot["objects"]:
        x1, y1, x2, y2 = obj["bbox"]

        text = f"ID {obj['id']} {obj['label']} {obj['range']} {obj['direction']}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # draw FSM state
    cv2.putText(
        frame,
        f"STATE: {nav_snapshot['state']}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
    )

    # draw decision
    if decision:
        cv2.putText(
            frame,
            f"ACTION: {decision['action']}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3,
        )


def main():
    cam = Camera()
    det = Detector()
    depth = DepthEstimator()
    spatial = SpatialEstimator()
    tracker = IOUTracker()

    world = WorldState()
    nav_state = NavigationState()
    navigator = NavigatorFSM(target_label="door")

    cam.open()
    prev = time.time()

    try:
        while True:
            frame = cam.read()
            if frame is None:
                break

            # ---------- perception ----------
            detections = det.detect(frame)
            tracks = tracker.update(detections)

            tracked_dets = [
                {"id": t.id, "bbox": t.bbox, "label": t.label, "confidence": 1.0}
                for t in tracks
            ]

            depth_map = depth.estimate(frame)
            spatial_objects = spatial.estimate(tracked_dets, depth_map)

            # ---------- world state ----------
            world.update(spatial_objects, target_label="door")
            world_snapshot = world.snapshot()

            # ---------- FSM navigation ----------
            decision = navigator.decide(world_snapshot, nav_state)
            nav_snapshot = nav_state.snapshot()

            if decision:
                print(
                    f"[{nav_snapshot['state']}] → {decision['action']} | {decision['reason']}"
                )

            # ---------- visualization ----------
            draw(frame, world_snapshot, nav_snapshot, decision)

            # ---------- FPS ----------
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

            cv2.imshow("Assistive Navigation FSM", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cam.release()


if __name__ == "__main__":
    main()