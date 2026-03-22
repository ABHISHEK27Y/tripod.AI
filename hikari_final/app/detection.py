"""detection.py — YOLO-World Detection · Hikari Shinro AI"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)
DEFAULT_LABELS = ["door","chair","table","person","steps","stairs","wall","obstacle","cup","bag"]


class YOLOWorldDetector:

    def __init__(self, model_size="yolov8s-worldv2"):
        self.model_size     = model_size
        self.model          = None
        self.current_labels = DEFAULT_LABELS[:]
        self._loaded        = False

    def load(self):
        if self._loaded:
            return
        try:
            # Patch torch.load to allow weights_only=False (PyTorch 2.6 fix)
            import torch
            _orig = torch.load
            def _safe(*a, **kw):
                kw.setdefault("weights_only", False)
                return _orig(*a, **kw)
            torch.load = _safe

            from ultralytics import YOLOWorld
            logger.info(f"Loading YOLO-World {self.model_size}...")
            self.model   = YOLOWorld(self.model_size)
            self._loaded = True
            self.set_labels(DEFAULT_LABELS)
            logger.info("YOLO-World loaded.")
        except Exception as e:
            logger.error(f"YOLO load failed: {e}")
            raise

    def set_labels(self, labels):
        clean = [l.strip().lower() for l in labels if l.strip()]
        if not clean:
            return
        self.current_labels = clean
        if self.model:
            try:
                self.model.set_classes(clean)
            except Exception:
                pass

    def detect(self, frame, labels=None, conf_threshold=0.15, imgsz=640):
        if not self._loaded:
            return []
        if labels:
            self.set_labels(labels)
        try:
            r = self.model.predict(frame, imgsz=imgsz, conf=conf_threshold, verbose=False)
        except Exception as e:
            logger.warning(f"Detect error: {e}")
            return []
        out = []
        if not r or r[0].boxes is None:
            return out
        for i in range(len(r[0].boxes)):
            try:
                b = r[0].boxes
                xy = b.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])
                cls = int(b.cls[i].cpu())
                lbl = self.current_labels[cls] if cls < len(self.current_labels) else "object"
                out.append({
                    "label": lbl, "conf": round(float(b.conf[i].cpu()), 3),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": (x1+x2)//2, "cy": (y1+y2)//2,
                    "w": x2-x1, "h": y2-y1,
                })
            except Exception:
                continue
        return out

    def draw_boxes(self, frame, detections):
        out = frame.copy()
        COLORS = [(0,212,255),(255,160,32),(46,204,113),(155,89,182),(231,76,60)]
        cm = {}
        for d in detections:
            if d["label"] not in cm:
                cm[d["label"]] = COLORS[len(cm) % len(COLORS)]
        for d in detections:
            c = cm.get(d["label"], (0,212,255))
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            cv2.rectangle(out, (x1,y1), (x2,y2), c, 2)
            for bx,by,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(out,(bx,by),(bx+dx*14,by),c,2)
                cv2.line(out,(bx,by),(bx,by+dy*14),c,2)
            txt = f"{d['label']} {d.get('distance_text','')}"
            (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(out,(x1,y1-th-10),(x1+tw+8,y1),c,-1)
            cv2.putText(out,txt,(x1+4,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,0,0),1,cv2.LINE_AA)
            if "trail" in d and len(d["trail"]) > 1:
                for j in range(1, len(d["trail"])):
                    cv2.line(out,
                        (int(d["trail"][j-1][0]),int(d["trail"][j-1][1])),
                        (int(d["trail"][j][0]),int(d["trail"][j][1])), c, 1)
        return out

    def overlay_astar(self, frame, path, gw=32, gh=24):
        if not path or len(path) < 2:
            return frame
        out = frame.copy()
        fh, fw = frame.shape[:2]
        cw, ch = fw/gw, fh/gh
        for i,(col,row) in enumerate(path):
            px, py = int((col+0.5)*cw), int((row+0.5)*ch)
            a = i/len(path)
            cv2.circle(out,(px,py),max(3,int(6*a)),(0,int(200*a),50+int(100*a)),-1)
            if i > 0:
                pc,pr = path[i-1]
                cv2.line(out,(int((pc+0.5)*cw),int((pr+0.5)*ch)),(px,py),(0,180,80),1)
        return out
