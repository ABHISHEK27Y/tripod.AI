import cv2


class Camera:
    def __init__(self, device_index=0, width=640, height=480):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.device_index)

        if not self.cap.isOpened():
            raise RuntimeError("❌ Cannot open camera")

        # set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        if self.cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        cv2.destroyAllWindows()