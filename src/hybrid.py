"""Bounded-memory subtitle alignment using sparse OCR and real-frame templates."""
from collections import deque
from dataclasses import dataclass
import json
import math
import unicodedata

import cv2
import numpy as np


def normalize(text):
    text = unicodedata.normalize("NFKC", text.replace("\\N", "").replace("\\n", ""))
    # OCR often confuses small/large kana. Fold for matching only, never for ASS text.
    text = text.translate(str.maketrans("ぁぃぅぇぉゃゅょっァィゥェォャュョッ", "あいうえおやゆよつアイウエオヤユヨツ"))
    return "".join(c for c in text if not c.isspace()
                   and not unicodedata.category(c).startswith("P"))


def similarity(a, b):
    """Normalized Levenshtein similarity (punctuation normalization is separate)."""
    if not a or not b:
        return 0.0
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        new = [i]
        for j, cb in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = new
    return 1.0 - row[-1] / max(len(a), len(b))


@dataclass
class Options:
    roi: tuple = (0.0625, 0.80555556, 0.9375, 0.88888889)
    ocr_interval: float = 0.20
    recheck_interval: float = 1.0
    buffer_seconds: float = 2.0
    max_buffer_frames: int = 180
    confirmations: int = 2
    end_confirm_frames: int = 3
    stable_frames: int = 2
    lookahead: int = 5
    min_confidence: float = 0.65
    min_similarity: float = 0.80
    candidate_margin: float = 0.08
    track_threshold: float = 0.65
    onset_threshold: float = 0.65
    change_threshold: float = 0.12
    short_line_seconds: float = 0.9

    @classmethod
    def from_config(cls, config):
        defaults = cls()
        values = {}
        for name, default in vars(defaults).items():
            if name == "roi":
                values[name] = tuple(json.loads(config.get("Hybrid", name, fallback=json.dumps(default))))
            elif isinstance(default, int):
                values[name] = config.getint("Hybrid", name, fallback=default)
            else:
                values[name] = config.getfloat("Hybrid", name, fallback=default)
        return cls(**values).validate()

    def validate(self):
        if len(self.roi) != 4 or not all(math.isfinite(v) for v in self.roi):
            raise ValueError("roi must contain four finite coordinates")
        x0, y0, x1, y1 = self.roi
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("roi must be normalized [left, top, right, bottom] within 0..1")
        for name in ("ocr_interval", "recheck_interval", "buffer_seconds", "max_buffer_frames",
                     "confirmations", "end_confirm_frames", "stable_frames", "lookahead", "short_line_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.recheck_interval < self.ocr_interval:
            raise ValueError("recheck_interval must be >= ocr_interval")
        for name in ("min_confidence", "min_similarity", "candidate_margin", "track_threshold",
                     "onset_threshold", "change_threshold"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be within (0, 1]")
        return self


@dataclass
class VideoFrame:
    index: int
    time: float
    image: np.ndarray


class VideoReader:
    def __init__(self, path, roi, ranges=None):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            self.cap.release()
            raise ValueError(f"Cannot open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.cap.release()
            raise ValueError("Video has no valid frame rate")
        self.roi = roi
        self.ranges = ranges
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        index = 0
        range_index = 0
        try:
            while True:
                if self.ranges is not None:
                    while range_index < len(self.ranges) and index >= self.ranges[range_index][1]:
                        range_index += 1
                    if range_index >= len(self.ranges):
                        break
                    if index < self.ranges[range_index][0]:
                        if not self.cap.grab():
                            break
                        index += 1
                        continue
                ok, image = self.cap.read()
                if not ok:
                    break
                # CFR contract: decoder timestamps and OCR latency never set timing.
                time = index / self.fps
                h, w = image.shape[:2]
                x0, y0, x1, y1 = self.roi
                crop = image[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
                if min(crop.shape[:2]) < 3:
                    raise ValueError("Subtitle ROI is too small")
                yield VideoFrame(index, time, crop.copy())
                index += 1
        finally:
            self.close()

    def close(self):
        self.cap.release()


def edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.Canny(gray, 60, 160)


@dataclass
class Sample:
    index: int
    time: float
    edge: np.ndarray


class VisualTemplate:
    def __init__(self, edge, box):
        x0, y0, x1, y1 = box
        h, w = edge.shape
        self.box = (max(0, x0), max(0, y0), min(w, x1), min(h, y1))
        x0, y0, x1, y1 = self.box
        self.image = edge[y0:y1, x0:x1].copy()
        self.valid = self.image.size > 0 and cv2.countNonZero(self.image) >= 10
        # First quarter helps recover the beginning of a typewriter animation.
        self.prefix_width = max(1, self.image.shape[1] // 4) if self.image.size else 1

    @staticmethod
    def overlap(reference, current):
        a, b = reference > 0, current > 0
        na, nb = int(a.sum()), int(b.sum())
        if min(na, nb) < 5:
            return 0.0
        kernel = np.ones((3, 3), np.uint8)
        da = cv2.dilate(reference, kernel) > 0
        db = cv2.dilate(current, kernel) > 0
        recall = float((a & db).sum()) / na
        precision = float((b & da).sum()) / nb
        return 2 * recall * precision / max(recall + precision, 1e-9)

    def score(self, edge, prefix=False):
        if not self.valid:
            return 0.0
        x0, y0, x1, y1 = self.box
        current = edge[y0:y1, x0:x1]
        reference = self.image
        if current.shape != reference.shape:
            return 0.0
        if prefix:
            reference, current = reference[:, :self.prefix_width], current[:, :self.prefix_width]
        return self.overlap(reference, current)


def choose_candidate(text, texts, cursor, options, limit=None):
    query = normalize(text)
    if not query:
        return None
    stop = min(len(texts), cursor + options.lookahead, limit if limit is not None else len(texts))
    candidates = [(similarity(query, texts[i]), i) for i in range(cursor, stop) if texts[i]]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if not candidates:
        return None
    score, index = candidates[0]
    # Short words and names are too ambiguous for fuzzy matching.
    if score < options.min_similarity or (min(len(query), len(texts[index])) <= 4 and query != texts[index]):
        return None
    # Identical repeated lines are resolved by script order, never skipped on a tie.
    different = [s for s, i in candidates[1:] if texts[i] != texts[index]]
    if different and score - max(different) < options.candidate_margin:
        return None
    return index, score


class HybridAligner:
    def __init__(self, reader, ocr, options=None):
        self.reader, self.ocr = reader, ocr
        self.options = (options or Options()).validate()
        self.stats = {"frames": 0, "ocr_calls": 0, "peak_buffer_frames": 0, "peak_buffer_bytes": 0}

    def align(self, events, skip_before=None):
        """Return JSON-serializable evidence; unmatched lines have no fabricated times."""
        opts = self.options
        texts = [normalize(event.Text) for event in events]
        results = [{"index": i, "text": e.Text, "status": "unmatched",
                    "reason": "no_confident_ocr_match"} for i, e in enumerate(events)]
        history = deque(maxlen=opts.max_buffer_frames)
        skip_before = dict(skip_before or {})
        cursor, active, pending = 0, None, None
        last_ocr, last_index, last_edge = -math.inf, -1, None
        stable, dirty, search_from = 0, True, 0.0
        applied_gaps = set()

        def finish(end_frame, reason):
            nonlocal cursor, active, search_from
            if active is None:
                return
            result = results[active["index"]]
            end_frame = max(result["start_frame"] + 1, end_frame)
            result.update(end_frame=end_frame, end=end_frame / self.reader.fps, end_reason=reason)
            cursor = active["index"] + 1
            search_from = result["end"]
            active = None

        def onset(template, anchor_frame):
            start, misses = anchor_frame, 0
            available = [s for s in history if search_from <= s.time and s.index <= anchor_frame]
            for sample in reversed(available):
                full_score = template.score(sample.edge)
                present = max(full_score, template.score(sample.edge, prefix=True))
                if active is not None:
                    old_score = active["template"].score(sample.edge)
                    if old_score >= opts.track_threshold and old_score > full_score + .05:
                        present = 0.0
                if present >= opts.onset_threshold:
                    start, misses = sample.index, 0
                else:
                    misses += 1
                    if misses >= opts.end_confirm_frames:
                        break
            truncated = bool(available and start == available[0].index
                             and available[0].time > search_from + 1 / self.reader.fps)
            return start, truncated

        try:
            for frame in self.reader:
                self.stats["frames"] += 1
                last_index = frame.index
                if cursor >= len(events):
                    break
                if active is None and cursor in skip_before and cursor not in applied_gaps:
                    search_from += skip_before[cursor]
                    applied_gaps.add(cursor)
                    history.clear()
                    pending = None
                if frame.time < search_from:
                    continue
                edge = edges(frame.image)
                history.append(Sample(frame.index, frame.time, edge))
                while history and frame.time - history[0].time > opts.buffer_seconds:
                    history.popleft()
                self.stats["peak_buffer_frames"] = max(self.stats["peak_buffer_frames"], len(history))
                self.stats["peak_buffer_bytes"] = max(self.stats["peak_buffer_bytes"], len(history) * edge.nbytes)
                changed = 1.0
                if last_edge is not None and last_edge.shape == edge.shape:
                    union = np.count_nonzero((edge > 0) | (last_edge > 0))
                    changed = np.count_nonzero(edge != last_edge) / max(union, 1)
                last_edge = edge
                stable = stable + 1 if changed < opts.change_threshold else 0
                dirty = dirty or changed >= opts.change_threshold

                if active is not None:
                    score = active["template"].score(edge)
                    if score >= opts.track_threshold:
                        active["misses"] = 0
                    else:
                        if active["misses"] == 0:
                            active["first_miss"] = frame.index
                        active["misses"] += 1
                        if active["misses"] >= opts.end_confirm_frames:
                            finish(active["first_miss"], "visual_change")
                            pending, dirty = None, True
                            # Apply explicit MV gaps before accepting another subtitle.
                            if cursor in skip_before and cursor not in applied_gaps:
                                continue
                if cursor >= len(events):
                    break
                elapsed = frame.time - last_ocr
                if elapsed < opts.ocr_interval:
                    continue
                periodic = elapsed >= opts.recheck_interval
                if (not periodic and active is not None and pending is None
                        and (not dirty or stable < opts.stable_frames)):
                    continue
                last_ocr, dirty = frame.time, False
                self.stats["ocr_calls"] += 1
                reading = self.ocr.read(frame.image)
                barriers = [i for i in skip_before if i > cursor]
                limit = min(barriers) if barriers else None

                def accept_reading(value):
                    if value is None or not math.isfinite(value.confidence) or value.confidence < opts.min_confidence:
                        return None
                    return choose_candidate(value.text, texts, cursor, opts, limit)

                candidate = accept_reading(reading)
                if candidate is None and reading is not None and hasattr(self.ocr, "alternatives"):
                    for alternative in self.ocr.alternatives(frame.image):
                        self.stats["ocr_calls"] += 1
                        candidate = accept_reading(alternative)
                        if candidate is not None:
                            reading = alternative
                            break
                if candidate is None:
                    pending = None
                    continue
                index, text_score = candidate
                if active is not None and index == active["index"]:
                    pending = None
                    continue
                template = VisualTemplate(edge, reading.box)
                if not template.valid:
                    pending = None
                    continue
                if pending is not None and pending["index"] == index and frame.time - pending["time"] <= opts.recheck_interval * 2:
                    pending["count"] += 1
                    pending["time"] = frame.time
                else:
                    pending = {"index": index, "count": 1, "time": frame.time,
                               "anchor_frame": frame.index, "template": template}
                # Duration is only a prior for confirmation cost, never an output boundary.
                short = 0 < getattr(events[index], "Duration", 0) <= opts.short_line_seconds
                strong_short = (short and text_score >= .999 and reading.confidence >= .95
                                and len(texts[index]) >= 5 and stable >= opts.stable_frames)
                if pending["count"] < opts.confirmations and not strong_short:
                    # Request another OCR soon even when the image stays unchanged.
                    dirty = True
                    continue
                start_frame, truncated = onset(pending["template"], pending["anchor_frame"])
                if active is not None:
                    finish(start_frame, "ocr_transition")
                for skipped in range(cursor, index):
                    results[skipped]["reason"] = "skipped_by_sequence_alignment"
                cursor = index
                start = start_frame / self.reader.fps
                results[index].update(status="matched", start_frame=start_frame, start=start, ocr_text=reading.text,
                                      ocr_model=reading.model,
                                      ocr_confidence=reading.confidence, text_similarity=text_score,
                                      onset_buffer_limited=truncated)
                results[index].pop("reason", None)
                active = {"index": index, "template": template, "misses": 0}
                pending = None
                print(f"Hybrid: {index + 1}/{len(events)} {events[index].Text} @ {start:.3f}s")
            if active is not None:
                end = active["first_miss"] if active["misses"] else last_index + 1
                finish(end, "video_end")
        finally:
            self.reader.close()
        self.stats["timing"] = "CFR: zero-based frame index / fps; end_frame is exclusive"
        return results
