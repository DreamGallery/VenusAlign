"""Rendered-template alignment with CFR frame-index timing."""
import json
from src.read_ini import config
from src.adv_text import to_time
from src.match import draw_text, compare


def time_fix(event, image_list, start_file_index, stream):
    binary, mask = draw_text(
        event.Text,
        json.loads(config.get("File Path", "FONT_PATH")),
        json.loads(config.get("Font Config", "font_size")),
        config.getint("Font Config", "stroke_width"),
        config.getint("Font Config", "kerning"),
    )
    threshold = config.getfloat("Arg", "threshold")
    start = None
    for index in range(start_file_index, len(image_list)):
        if compare(image_list[index][1], binary, threshold, mask):
            start = index
            break
    if start is None:
        raise ValueError(f"Cannot find subtitle: {event.Text}")
    misses, end = 0, len(image_list)
    confirm = config.getint("Hybrid", "end_confirm_frames", fallback=3)
    for index in range(start + 1, len(image_list)):
        if compare(image_list[index][1], binary, threshold, mask):
            misses = 0
        else:
            misses += 1
            if misses >= confirm:
                end = index - misses + 1
                break
    event.start_frame, event.end_frame = start, end
    event.Start = to_time(start / stream.fps)
    event.End = to_time(end / stream.fps)
    return end
