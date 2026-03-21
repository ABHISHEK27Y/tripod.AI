from ultralytics import YOLO


class Detector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]

        detections = []

        if results.boxes is None:
            return detections

        for box in results.boxes:
            conf = float(box.conf[0])

            if conf < self.conf_threshold:
                continue

            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

        return detections