"""Live webcam sensory stream.

Frames never leave the machine and are never written to disk: each frame is reduced to a
48x64 grey image + a motion-energy image (fed to the ANN's conv encoder) and a motion
centroid (which drives the orb in the ``webcam_chase`` task). If no camera is available,
or ``enabled=False``, a synthetic moving blob is used so everything still runs.
"""
from __future__ import annotations

import base64
import threading
import time

import numpy as np

H, W = 48, 64


class Webcam:
    def __init__(self, enabled: bool = False, index: int = 0):
        self.enabled, self.index = enabled, index
        self.frame = np.zeros((2, H, W), np.float32)
        self.centroid = (0.5, 0.5, 0.0)
        self.thumb_jpeg = b""
        self.live = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._yy, self._xx = np.mgrid[0:H, 0:W]
        self._k = 0
        self._prev = None

        if self.enabled:
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        else:
            self._t = None

    def _run(self):
        cap = None
        try:
            import cv2
            cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(self.index)
            if not cap.isOpened():
                cap = None
        except Exception:
            cap = None
        self.live = cap is not None
        prev = None
        while not self._stop.is_set():
            if cap is not None:
                import cv2
                ok, bgr = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                grey = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (W, H)).astype(np.float32) / 255.0
                ok, jpg = cv2.imencode(".jpg", cv2.resize(bgr, (160, 120)), [cv2.IMWRITE_JPEG_QUALITY, 60])
                thumb = jpg.tobytes() if ok else b""
                motion = np.abs(grey - prev) if prev is not None else np.zeros_like(grey)
                prev = grey
                m = motion.sum()
                if m > 1.0:
                    c = ((motion * self._xx).sum() / m / W, (motion * self._yy).sum() / m / H, float(min(1.0, m / 200)))
                else:
                    c = (self.centroid[0], self.centroid[1], 0.0)
                with self._lock:
                    self.frame = np.stack([grey, np.clip(motion * 4, 0, 1)]).astype(np.float32)
                    self.centroid = c
                    self.thumb_jpeg = thumb
            else:
                time.sleep(0.1)
        if cap is not None:
            cap.release()

    def read(self):
        if not self.enabled:
            self._k += 1
            cx = W / 2 + W / 3 * np.sin(self._k / 40)
            cy = H / 2 + H / 4 * np.cos(self._k / 57)
            grey = np.exp(-((self._xx - cx) ** 2 + (self._yy - cy) ** 2) / 40.0).astype(np.float32)
            motion = np.abs(grey - self._prev) if self._prev is not None else np.zeros_like(grey)
            self._prev = grey
            m = motion.sum()
            c = ((motion * self._xx).sum() / m / W, (motion * self._yy).sum() / m / H, float(min(1.0, m / 200))) if m > 1.0 else (0.5, 0.5, 0.0)
            self.frame = np.stack([grey, np.clip(motion * 4, 0, 1)]).astype(np.float32)
            self.centroid = c
            return self.frame.copy(), self.centroid
        with self._lock:
            return self.frame.copy(), self.centroid

    def thumb_b64(self):
        with self._lock:
            return base64.b64encode(self.thumb_jpeg).decode() if self.thumb_jpeg else ""

    def close(self):
        self._stop.set()
