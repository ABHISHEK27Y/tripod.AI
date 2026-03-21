from ultralytics import YOLOWorld

class Detector:
    def __init__(self, model_path="yolov8s-world.pt", conf_threshold=0.6):
        # YOLOWorld brings open-vocabulary detection with zero-shot capabilities.
        self.model = YOLOWorld(model_path)
        self.conf_threshold = conf_threshold
        
        # Initial classes (can be generic or specific depending on the scene)
        self.current_classes = ["door", "chair", "person", "table"]
        self.model.set_classes(self.current_classes)

    def set_target_classes(self, classes_list):
        """Dynamically updates the classes YOLO-World looks for without retraining."""
        self.current_classes = classes_list
        self.model.set_classes(self.current_classes)
        print(f"[Detector] YOLO-World classes updated to: {self.current_classes}")

    def detect(self, frame):
        # Use classes specific for this frame
        results = self.model(frame, verbose=False)[0]
        detections = []

        if results.boxes is None:
            return detections

        for box in results.boxes:
            conf = float(box.conf[0])

            # Layer 6 Quant Requirement: Confidence Profile
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