"""vision.py — Camera Capture + Frame Pipeline · Hikari Shinro AI"""
import cv2, numpy as np, threading, time, base64, logging
from typing import Optional, Callable, Tuple
logger = logging.getLogger(__name__)

FRAME_W, FRAME_H, TARGET_FPS = 640, 480, 12


class CameraCapture:

    def __init__(self, device_id=0):
        self.device_id = device_id
        self._cap      = None
        self._frame    = None
        self._lock     = threading.Lock()
        self._running  = False

    def start(self):
        # CAP_DSHOW fixes Windows MSMF error
        self._cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # fallback without flag
            self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            logger.error(f"Cannot open camera {self.device_id}")
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self._cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info("Camera started.")
        return True

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            time.sleep(1./(TARGET_FPS*2))

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._cap: self._cap.release()


class FramePipeline:

    def __init__(self, detector, depth_estimator, quant_engine, on_frame: Callable):
        self.detector  = detector
        self.depth     = depth_estimator
        self.quant     = quant_engine
        self.on_frame  = on_frame
        self._labels   = []
        self._lock     = threading.Lock()

    def set_labels(self, labels):
        with self._lock:
            self._labels = labels[:]
            self.detector.set_labels(labels)

    def process_frame(self, frame, target_label=""):
        h, w = frame.shape[:2]
        with self._lock:
            labels = self._labels[:]

        raw    = self.detector.detect(frame, labels=labels or None)
        depth  = self.depth.estimate(frame)
        enriched, scene, path = self.quant.process(raw, depth, target_label, w, h)

        out = frame.copy()
        if depth is not None:
            out = self.depth.blend_overlay(out, depth, alpha=0.28)
        out = self.detector.draw_boxes(out, enriched)
        if path:
            out = self.detector.overlay_astar(out, path)
        out = self._hud(out, enriched, scene, target_label)

        meta = {
            "detections":  enriched,
            "scene":       scene,
            "astar_path":  path,
            "target_label": target_label,
            "object_count": len(enriched),
        }
        return out, meta

    def _hud(self, frame, dets, scene, target):
        h, w = frame.shape[:2]
        out  = frame.copy()
        ov   = out.copy()
        cv2.rectangle(ov,(0,0),(w,28),(13,17,23),-1)
        cv2.addWeighted(ov,0.8,out,0.2,0,out)
        cv2.putText(out,"視覚  HIKARI SHINRO AI",(8,18),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,210,220),1,cv2.LINE_AA)
        cv2.putText(out,f"OBJ:{len(dets)} TARGET:{target or 'none'}",(w-220,18),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,180,180),1,cv2.LINE_AA)
        for bx,by,sx,sy in[(0,0,1,1),(w,0,-1,1),(0,h,1,-1),(w,h,-1,-1)]:
            cv2.line(out,(bx,by),(bx+sx*18,by),(0,210,220),2)
            cv2.line(out,(bx,by),(bx,by+sy*18),(0,210,220),2)
        short = scene[:85]+"..." if len(scene)>85 else scene
        ov2=out.copy(); cv2.rectangle(ov2,(0,h-30),(w,h),(13,17,23),-1)
        cv2.addWeighted(ov2,0.82,out,0.18,0,out)
        cv2.putText(out,short,(8,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,200),1,cv2.LINE_AA)
        return out

    def frame_to_b64(self, frame):
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf).decode("utf-8")

    def emit_frame(self, frame, meta):
        self.on_frame(self.frame_to_b64(frame), meta)
