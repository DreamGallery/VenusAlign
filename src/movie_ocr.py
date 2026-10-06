"""为脚本中没有对应台词的非实时渲染视频补充识别字幕。"""
from collections import deque
from dataclasses import replace
import json
import math
import unicodedata

import cv2
import numpy as np

from src.hybrid import Sample, VisualTemplate, edges


def movie_text_image(image, threshold):
    """Keep glyph interiors even where the outline touches a dark background."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dark = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    contours, hierarchy = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    glyphs = np.zeros_like(gray)
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            _, _, width, height = cv2.boundingRect(contour)
            if hierarchy[0, i, 3] >= 0 and 2 < height < gray.shape[0] * .95 and 2 < width < gray.shape[1] * .3:
                cv2.drawContours(glyphs, [contour], -1, 255, -1)
    return 255 - cv2.bitwise_and(glyphs, 255 - dark)


def movie_options(config, base):
    return replace(
        base,
        roi=tuple(json.loads(config.get('Movie OCR', 'roi', fallback=json.dumps(base.roi)))),
        min_confidence=config.getfloat('Movie OCR', 'min_confidence', fallback=.85),
        ocr_interval=config.getfloat('Movie OCR', 'ocr_interval', fallback=.2),
        recheck_interval=config.getfloat('Movie OCR', 'recheck_interval', fallback=.6),
        confirmations=config.getint('Movie OCR', 'confirmations', fallback=2),
    ).validate()


def scan_regions(references, results, frame_count, manual=()):
    """根据已对齐台词的帧边界确定非实时渲染视频的扫描区间。

    没有台词分隔的连续非实时渲染视频共用区间，重叠区间合并。
    排除已匹配台词以避免重复，脚本时长不直接用于输出时间轴。
    """
    if frame_count <= 0:
        raise ValueError('非实时渲染视频 OCR 需要有效的视频总帧数')
    matched = {i: r for i, r in enumerate(results) if r['status'] == 'matched'}
    windows = []
    for ref in references:
        before = [matched[i] for i in ref['before'] if i in matched]
        after = [matched[i] for i in ref['after'] if i in matched]
        # Adjacent source files provide outer bounds for an otherwise empty part.
        before += [r for i, r in matched.items() if i < ref['segment_start']]
        after += [r for i, r in matched.items() if i >= ref['segment_end']]
        lo = max((r['end_frame'] for r in before), default=0)
        hi = min((r['start_frame'] for r in after), default=frame_count)
        if lo < hi:
            windows.append({'start_frame': lo, 'end_frame': hi, 'assets': [ref['asset']],
                            'bounds': 'neighboring_dialogues' if before and after else 'open_ended'})
    for lo, hi in manual:
        if not (0 <= lo < hi <= frame_count):
            raise ValueError(f'非实时渲染视频帧范围 {lo}:{hi} 超出 0:{frame_count}')
        windows.append({'start_frame': lo, 'end_frame': hi, 'assets': [], 'bounds': 'manual'})
    merged = []
    for window in sorted(windows, key=lambda w: w['start_frame']):
        if merged and window['start_frame'] <= merged[-1]['end_frame']:
            merged[-1]['end_frame'] = max(merged[-1]['end_frame'], window['end_frame'])
            merged[-1]['assets'] = sorted(set(merged[-1]['assets'] + window['assets']))
            merged[-1]['bounds'] = 'merged'
        else:
            merged.append(dict(window))
    known = sorted((r['start_frame'], r['end_frame']) for r in matched.values())
    regions = []
    for window in merged:
        lo, hi = window['start_frame'], window['end_frame']
        for start, end in known:
            if end <= lo or start >= hi:
                continue
            if start > lo:
                regions.append({**window, 'start_frame': lo, 'end_frame': start})
            lo = max(lo, end)
            if lo >= hi:
                break
        if lo < hi:
            regions.append({**window, 'start_frame': lo, 'end_frame': hi})
    return regions


def text_key(text):
    # Unlike script matching, do not fold small kana when there is no source text.
    return ''.join(c for c in unicodedata.normalize('NFKC', text)
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


def same_text(a, b):
    # Without source text, one changed character can mean a different sentence.
    return a == b


class MovieExtractor:
    def __init__(self, reader, ocr, options):
        self.reader, self.ocr, self.options = reader, ocr, options.validate()
        self.stats = {'frames': 0, 'ocr_calls': 0, 'rejected_readings': 0, 'peak_buffer_frames': 0}

    def extract(self, regions):
        opts, results = self.options, []
        history = deque(maxlen=opts.max_buffer_frames)
        region_index, active, pending = 0, None, None
        last_ocr, last_index, floor = -math.inf, -1, 0

        def finish(end, reason):
            nonlocal active, floor
            if active is None:
                return
            row = active['row']
            row.update(end_frame=max(row['start_frame'] + 1, end), end_reason=reason)
            row['end'] = row['end_frame'] / self.reader.fps
            results.append(row)
            floor, active = row['end_frame'], None

        def onset(template, anchor):
            start, misses = anchor, 0
            available = [s for s in history if floor <= s.index <= anchor]
            for sample in reversed(available):
                score = template.score(sample.edge)
                if active is not None:
                    old_score = active['template'].score(sample.edge)
                    if old_score >= opts.track_threshold and old_score >= score - .01:
                        score = 0.0
                if score >= opts.onset_threshold:
                    start, misses = sample.index, 0
                else:
                    misses += 1
                    if misses >= opts.end_confirm_frames:
                        break
            return start, bool(available and start == available[0].index and start > floor)

        try:
            for frame in self.reader:
                last_index = frame.index
                while region_index < len(regions) and frame.index >= regions[region_index]['end_frame']:
                    finish(regions[region_index]['end_frame'], 'region_end')
                    region_index += 1
                    history.clear()
                    pending, last_ocr = None, -math.inf
                if region_index >= len(regions):
                    break
                region = regions[region_index]
                if frame.index < region['start_frame']:
                    continue
                floor = max(floor, region['start_frame'])
                self.stats['frames'] += 1
                edge = edges(frame.image)
                history.append(Sample(frame.index, frame.time, edge))
                while history and frame.time - history[0].time > opts.buffer_seconds:
                    history.popleft()
                self.stats['peak_buffer_frames'] = max(self.stats['peak_buffer_frames'], len(history))
                if active is not None:
                    if active['template'].score(edge) >= opts.track_threshold:
                        active['misses'] = 0
                    else:
                        if not active['misses']:
                            active['first_miss'] = frame.index
                        active['misses'] += 1
                        if active['misses'] >= opts.end_confirm_frames:
                            finish(active['first_miss'], 'visual_change')
                            pending = None
                interval = opts.recheck_interval if active is not None and pending is None else opts.ocr_interval
                if frame.time - last_ocr < interval:
                    continue
                last_ocr = frame.time
                self.stats['ocr_calls'] += 1
                reading = self.ocr.read(frame.image)
                agreements = 1

                def usable(value):
                    return (value is not None and math.isfinite(value.confidence)
                            and value.confidence >= opts.min_confidence
                            and len(text_key(value.text)) >= 2
                            and any(c.isalpha() for c in value.text))

                if (reading is not None and (not usable(reading) or active is None
                        or not same_text(text_key(reading.text), active['key']))):
                    candidates = [reading] if usable(reading) else []
                    raw_width = max(1, reading.box[2] - reading.box[0])
                    for threshold in (40, 70):
                        self.stats['ocr_calls'] += 1
                        alternative = self.ocr.read(movie_text_image(frame.image, threshold))
                        # Do not prefer confident but truncated preprocessing output.
                        if (usable(alternative)
                                and alternative.box[2] - alternative.box[0] >= raw_width * .95):
                            candidates.append(alternative)
                    if candidates:
                        reading = max(candidates, key=lambda r: r.confidence)
                        agreements = sum(same_text(text_key(r.text), text_key(reading.text)) for r in candidates)
                if not usable(reading):
                    self.stats['rejected_readings'] += 1
                    pending = None
                    continue
                key = text_key(reading.text)
                if active is not None and not same_text(key, active['key']):
                    clean = movie_text_image(frame.image, 70)
                    x0, y0, x1, y1 = active['template'].box
                    previous = active['signature'][y0:y1, x0:x1] == 0
                    current = clean[y0:y1, x0:x1] == 0
                    union = np.count_nonzero(previous | current)
                    if union >= 10 and np.count_nonzero(previous != current) / union < .02:
                        # Identical visible glyphs with a revised OCR spelling are
                        # one subtitle. A real character change alters this mask.
                        if reading.confidence > active['row']['ocr_confidence']:
                            active['row'].update(text=reading.text, ocr_confidence=reading.confidence,
                                                 ocr_model=reading.model)
                            active['key'] = key
                        active['row']['observations'] += 1
                        active['last_confirmed'] = frame.index
                        pending = None
                        continue
                if active is not None and same_text(key, active['key']):
                    active['row']['observations'] += 1
                    active['last_confirmed'] = frame.index
                    if reading.confidence > active['row']['ocr_confidence']:
                        active['row'].update(text=reading.text, ocr_confidence=reading.confidence,
                                             ocr_model=reading.model)
                    pending = None
                    continue
                if active is not None and active['template'].score(edge) >= .98 and agreements < 2:
                    # OCR spelling fluctuations on the same image are not new cues.
                    pending = None
                    continue
                template = VisualTemplate(edge, reading.box)
                if not template.valid:
                    pending = None
                    continue
                if (pending is not None and same_text(key, pending['key'])
                        and frame.time - pending['time'] <= opts.recheck_interval * 2
                        and pending['template'].score(edge) >= opts.track_threshold):
                    pending['count'] += 1
                    pending['time'] = frame.time
                    if reading.confidence > pending['reading'].confidence:
                        pending['reading'] = reading
                else:
                    pending = {'key': key, 'count': 1, 'time': frame.time, 'anchor': frame.index,
                               'template': template, 'reading': reading}
                if pending['count'] < opts.confirmations:
                    continue
                start, limited = onset(pending['template'], pending['anchor'])
                if active is not None:
                    # No prefix backtracking: avoid absorbing a similar preceding line.
                    start = max(start, active['last_confirmed'] + 1)
                    finish(start, 'ocr_transition')
                best = pending['reading']
                row = {'status': 'ocr_only', 'source': 'movie_ocr', 'needs_review': True,
                       'text': best.text, 'start_frame': start, 'start': start / self.reader.fps,
                       'ocr_confidence': best.confidence, 'ocr_model': best.model,
                       'observations': pending['count'], 'onset_buffer_limited': limited,
                       'movie_assets': region['assets'], 'ocr_box': list(best.box)}
                active = {'row': row, 'key': key, 'template': template, 'misses': 0,
                          'last_confirmed': frame.index, 'signature': movie_text_image(frame.image, 70)}
                pending = None
                print(f'非实时渲染视频 OCR: {best.text} @ {start / self.reader.fps:.3f}s')
            if active is not None:
                end = active['first_miss'] if active['misses'] else last_index + 1
                finish(min(end, regions[region_index]['end_frame']), 'video_end')
        finally:
            self.reader.close()
        return results
