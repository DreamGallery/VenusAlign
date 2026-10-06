"""Lazy RapidOCR adapter. No model is loaded in template mode or unit tests."""
from dataclasses import dataclass
from pathlib import Path
import copy

import numpy as np
import cv2


@dataclass
class Reading:
    text: str
    confidence: float
    box: tuple  # x0, y0, x1, y1, relative to the subtitle ROI
    model: str = ""


def reading_order(boxes):
    """Group boxes into horizontal rows before sorting left-to-right.

    A few pixels of baseline noise must not reverse fragments of the same line.
    """
    rows = []
    for i in sorted(range(len(boxes)), key=lambda i: float(boxes[i, :, 1].mean())):
        center = float(boxes[i, :, 1].mean())
        height = max(1.0, float(np.ptp(boxes[i, :, 1])))
        row = next((row for row in rows if abs(center - row[0]) <= min(height, row[1]) * .5), None)
        if row is None:
            rows.append([center, height, [i]])
        else:
            row[2].append(i)
    return [i for row in rows for i in sorted(row[2], key=lambda i: float(boxes[i, :, 0].min()))]


def outlined_text(image):
    """Recover light glyph interiors enclosed by dark game subtitle outlines.

    White-on-black outlined dots otherwise resemble digits to OCR. Components
    touching the ROI edge or spanning the background are deliberately excluded.
    This is only an OCR input; tracking always uses the unmodified video frame.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dark = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)[1]
    filled = np.zeros_like(gray)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (3 < h < height * .95 and 3 < w < width * .3
                and x > 0 and y > 0 and x + w < width and y + h < height):
            cv2.drawContours(filled, [contour], -1, 255, -1)
    return 255 - cv2.bitwise_and(filled, 255 - dark)


class SubtitleOCR:
    def __init__(self, config):
        try:
            from rapidocr import RapidOCR, EngineType, LangRec, ModelType, OCRVersion
        except ImportError as exc:
            raise RuntimeError("Hybrid mode requires: pip install -r requirements-ocr.txt") from exc
        preset = config.get("Hybrid", "ocr_model", fallback="japan_v4")
        self.preset = preset
        self.preserve_lines = False
        presets = {
            "japan_v4": ("PPOCRV4", "JAPAN", "MOBILE"),
            "v5_mobile": ("PPOCRV5", "CH", "MOBILE"),
            "v6_small": ("PPOCRV6", "CH", "SMALL"),
            "v6_medium": ("PPOCRV6", "CH", "MEDIUM"),
        }
        if preset not in presets:
            raise ValueError(f"Unknown ocr_model: {preset}; choose {list(presets)}")
        version, lang, size = presets[preset]
        try:
            params = {
                "Rec.ocr_version": getattr(OCRVersion, version),
                "Rec.lang_type": getattr(LangRec, lang),
                "Rec.model_type": getattr(ModelType, size),
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Det.ocr_version": OCRVersion.PPOCRV4,
                "Det.model_type": ModelType.MOBILE,
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Global.use_cls": False,
                "Global.text_score": 0.0,
                "Global.log_level": "error",
            }
        except AttributeError as exc:
            raise RuntimeError(f"Installed RapidOCR does not support {preset}; upgrade RapidOCR.") from exc
        model_path = config.get("Hybrid", "rec_model_path", fallback="").strip()
        if model_path:
            if not Path(model_path).is_file():
                raise ValueError(f"OCR model does not exist: {model_path}")
            params["Rec.model_path"] = model_path
        self.use_detection = config.getboolean("Hybrid", "use_detection", fallback=True)
        self.fallback_preprocess = config.getboolean("Hybrid", "fallback_preprocess", fallback=True)
        self.fallback_model = config.get("Hybrid", "fallback_model", fallback="v6_small").strip()
        if self.fallback_model == preset:
            self.fallback_model = ""
        self.secondary = None
        self.config = config
        self.engine = RapidOCR(params=params)

    def alternatives(self, image):
        """Retry stylized outlines only when the original reading cannot be aligned."""
        if self.fallback_preprocess:
            yield self.read(outlined_text(image))
        if self.fallback_model:
            if self.secondary is None:
                secondary_config = copy.deepcopy(self.config)
                secondary_config.set("Hybrid", "ocr_model", self.fallback_model)
                secondary_config.set("Hybrid", "fallback_model", "")
                secondary_config.set("Hybrid", "rec_model_path", "")
                self.secondary = SubtitleOCR(secondary_config)
            self.secondary.preserve_lines = self.preserve_lines
            yield self.secondary.read(image)
        if not self.fallback_preprocess:
            return
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        yield self.read(255 - gray)
        clean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, -12)
        yield self.read(255 - clean)

    def read(self, image):
        output = self.engine(image, use_det=self.use_detection, use_cls=False, use_rec=True)
        texts, scores = getattr(output, "txts", None), getattr(output, "scores", None)
        if texts is None or scores is None or len(texts) == 0:
            return None
        boxes = getattr(output, "boxes", None)
        if boxes is None:
            h, w = image.shape[:2]
            box = (0, 0, w, h)
            order = list(range(len(texts)))
        else:
            boxes = np.asarray(boxes)
            # Reading order for horizontal subtitles; keep multiline order.
            order = reading_order(boxes)
            h, w = image.shape[:2]
            box = (max(0, int(boxes[:, :, 0].min()) - 2),
                   max(0, int(boxes[:, :, 1].min()) - 2),
                   min(w, int(np.ceil(boxes[:, :, 0].max())) + 3),
                   min(h, int(np.ceil(boxes[:, :, 1].max())) + 3))
        # The detector may isolate an ellipsis as a separate low-confidence box.
        # Weight fragments by length; script similarity and repeated observations
        # independently gate acceptance in HybridAligner.
        weights = [max(1, len(text.strip())) for text in texts]
        confidence = sum(float(s) * w for s, w in zip(scores, weights)) / sum(weights)
        joined, row_center, row_height = [], None, None
        for i in order:
            if self.preserve_lines and boxes is not None:
                center = float(boxes[i, :, 1].mean())
                height = max(1.0, float(np.ptp(boxes[i, :, 1])))
                if row_center is not None and abs(center - row_center) > min(height, row_height) * .5:
                    joined.append('\n')
                    row_center = None
                if row_center is None:
                    row_center, row_height = center, height
            joined.append(texts[i])
        return Reading(''.join(joined), confidence, box, self.preset)
