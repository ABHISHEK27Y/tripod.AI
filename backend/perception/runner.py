import cv2
import time

from perception.camera import Camera
from perception.detector import Detector


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]

        text = f"{label} {conf:.2f}"

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
    camera = Camera()
    detector = Detector()

    camera.open()

    prev_time = time.time()

    try:
        while True:
            frame = camera.read()

            if frame is None:
                print("⚠️ Failed to read frame")
                break

            detections = detector.detect(frame)

            # print detections to console
            for d in detections:
                print(f"{d['label']} ({d['confidence']:.2f})")

            draw_detections(frame, detections)

            # FPS calculation
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            cv2.imshow("Tripod Vision", frame)

            # press q to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.release()


if __name__ == "__main__":
    main()