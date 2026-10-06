"""Legacy template-mode preprocessing; hybrid mode uses the streaming VideoReader."""
import math
from pathlib import Path

import cv2
from src.read_ini import config
from src.match import to_binary_adaptive


class FrameProcess:
    def to_frame(self, filename):
        path = Path(filename)
        if not path.is_absolute():
            path = Path(config.get("File Path", "VIDEO_PATH")) / path
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {path}")
            self.fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(self.fps) or self.fps <= 0:
                raise ValueError("Video has no valid frame rate")
            images = []
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                crop = frame[h * 29 // 36:h * 8 // 9, w // 16:w * 15 // 16]
                binary = to_binary_adaptive(crop, 11, 0)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                images.append((len(images) / self.fps, binary))
            if not images:
                raise ValueError("Video contains no readable frames")
            return images
        finally:
            capture.release()
