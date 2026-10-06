"""VenusAlign for IDOLY PRIDE: generate ASS subtitles with OCR and frame alignment."""
import argparse
import copy
import json
from pathlib import Path
import time

from src.adv_text import extract, get_title, frame_to_time, video_references
from src.read_ini import config
from src.events import AssEvents
from src.ass_part import script_info, garbage, style, event


def load_events(script=None, script_csv=None):
    filename = script or config.get("Info", "game_file_name").strip()
    if not filename:
        raise ValueError("Set Info.game_file_name in config.ini")
    dialogues = extract(filename, script_csv)
    movies = video_references(filename, dialogues)
    skip_before = {}
    if config.getboolean("Sub", "MV_exists"):
        duration = config.getfloat("Sub", "mv_skip_seconds", fallback=0.0)
        if duration < 0:
            raise ValueError("mv_skip_seconds must be non-negative")
        skip_before[len(dialogues)] = duration
        subfile = config.get("Sub", "sub_file_name")
        second = extract(subfile)
        movies += video_references(subfile, second, len(dialogues))
        dialogues += second
    events = []
    for dialogue in dialogues:
        item = AssEvents()
        item.from_dialogue(dialogue)
        events.append(item)
    if not events and not movies:
        raise ValueError("No subtitle events found in the input script")
    return events, skip_before, get_title(filename), movies


def write_ass(path, events, results, fps, movie_results=()):
    need_comment = config.getboolean("Arg", "need_comment")
    lines = [script_info, garbage, style, event]
    rows = [(item, result) for item, result in zip(events, results) if result["status"] == "matched"]
    for result in movie_results:
        # OCR output is literal text, never ASS override code.
        text = result['text'].replace('\\', '＼').replace('{', '｛').replace('}', '｝')
        text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\n', r'\N')
        item = AssEvents(Text=text, Name='OCR', Style='IdolyPride Normal')
        rows.append((item, result))
    for item, result in sorted(rows, key=lambda row: row[1]['start_frame']):
        # CFR frames are authoritative; preserve legacy centisecond truncation.
        item.Start = frame_to_time(result["start_frame"], fps)
        item.End = frame_to_time(result["end_frame"], fps)
        if need_comment:
            lines.extend([item.echo_dialogue(), item.echo_comment()])
        else:
            lines.append(item.echo_dialogue() + item.Text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def run_hybrid(events, skip_before, video, output, movies=(), movie_mode='auto', movie_ranges=()):
    from src.hybrid import HybridAligner, Options, VideoReader
    from src.ocr import SubtitleOCR

    options = Options.from_config(config)
    ocr = SubtitleOCR(config) if events else None
    reader = VideoReader(video, options.roi)
    aligner = HybridAligner(reader, ocr, options)
    started = time.perf_counter()
    if events:
        results = aligner.align(events, skip_before)
    else:
        results = []
        reader.close()
    movie_results, regions, movie_stats = [], [], {}
    if movie_mode != 'off' and (movies or movie_ranges or movie_mode == 'all'):
        from src.movie_ocr import MovieExtractor, movie_options, scan_regions
        manual = movie_ranges or ([(0, reader.total_frames)] if movie_mode == 'all' else [])
        regions = scan_regions([] if manual else movies, results, reader.total_frames, manual)
        if regions:
            opts = movie_options(config, options)
            movie_config = copy.deepcopy(config)
            movie_config.set('Hybrid', 'ocr_model', config.get('Movie OCR', 'ocr_model', fallback='v6_small'))
            movie_config.set('Hybrid', 'rec_model_path', config.get('Movie OCR', 'rec_model_path', fallback=''))
            movie_config.set('Hybrid', 'fallback_model', '')
            movie_ocr = SubtitleOCR(movie_config)
            movie_ocr.preserve_lines = True
            movie_reader = VideoReader(video, opts.roi, [(r['start_frame'], r['end_frame']) for r in regions])
            movie_extractor = MovieExtractor(movie_reader, movie_ocr, opts)
            movie_results = movie_extractor.extract(regions)
            movie_stats = movie_extractor.stats
    matched = sum(r["status"] == "matched" for r in results)
    report = {
        "video": str(video), "fps": reader.fps, "video_frames": reader.total_frames,
        "options": vars(options),
        "ocr_model": config.get("Hybrid", "ocr_model", fallback="japan_v4"),
        "ass_time_quantization": "truncate",
        "stats": {**aligner.stats, "elapsed_seconds": time.perf_counter() - started,
                  "matched": matched, "unmatched": len(events) - matched},
        "events": results,
        "movie_references": movies, "movie_regions": regions,
        "movie_events": movie_results, "movie_stats": movie_stats,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_suffix(".review.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if matched == 0 and not movie_results:
        raise RuntimeError(f"No confident matches; ASS not written. Inspect {report_path}")
    write_ass(output, events, results, reader.fps, movie_results)
    if movie_mode != 'off' and (movies or movie_ranges or movie_mode == 'all'):
        write_ass(output.with_suffix('.movies.ass'), [], [], reader.fps, movie_results)
    print(f"已保存 {output}：匹配 {matched}/{len(events)} 条脚本字幕，补充 {len(movie_results)} 条非实时渲染视频 OCR 字幕；复核报告：{report_path}")
    return report


def run_template(events, skip_before, video_name, output):
    from src.frame import FrameProcess
    from src.time_fix import time_fix

    stream = FrameProcess()
    images = stream.to_frame(video_name)
    images.sort(key=lambda x: float(x[0]))
    cursor, results = 0, []
    for i, item in enumerate(events):
        cursor += int(skip_before.get(i, 0) * stream.fps)
        cursor = time_fix(item, images, cursor, stream)
        results.append({"status": "matched", "start_frame": item.start_frame,
                        "end_frame": item.end_frame})
    write_ass(output, events, results, stream.fps)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("hybrid", "template"),
                        default=config.get("Arg", "match_mode", fallback="hybrid"))
    parser.add_argument("--video", help="Override the configured video path")
    parser.add_argument("--script", help="Original Hoshimi-Adv Resource/adv_*.txt path")
    parser.add_argument("--script-csv", help="Optional Hoshimi/Teleprompter CSV to verify against TXT")
    parser.add_argument("--player-name", help="Player name as displayed in the recording")
    parser.add_argument("--output", help="Override the output ASS path")
    parser.add_argument("--movie-ocr", choices=('auto', 'off', 'all'), default='auto',
                        help="非实时渲染视频 OCR：auto 扫描引用区间，off 关闭，all 扫描无脚本字幕区间")
    parser.add_argument("--movie-range", action='append', default=[], metavar='START_FRAME:END_FRAME',
                        help="用指定帧范围替代非实时渲染视频的自动扫描区间；不含结束帧，可重复指定")
    args = parser.parse_args()
    if args.player_name:
        config.set("Info", "player_name", args.player_name)
    movie_ranges = []
    for value in args.movie_range:
        try:
            lo, hi = map(int, value.split(':'))
            if not 0 <= lo < hi:
                raise ValueError()
        except ValueError:
            parser.error('--movie-range requires START_FRAME:END_FRAME with 0 <= start < end')
        movie_ranges.append((lo, hi))
    if args.mode == 'template' and (movie_ranges or args.movie_ocr == 'all'):
        parser.error('非实时渲染视频 OCR 需要 hybrid 模式')
    if args.movie_ocr == 'off' and movie_ranges:
        parser.error('--movie-range cannot be used with --movie-ocr off')
    events, skip_before, title, movies = load_events(args.script, args.script_csv)
    video_name = args.video or config.get("Info", "video_file_name").strip()
    if not video_name:
        raise ValueError("Set Info.video_file_name or pass --video")
    video = Path(video_name) if args.video else Path(config.get("File Path", "VIDEO_PATH")) / video_name
    output = Path(args.output) if args.output else Path(config.get("File Path", "ASS_PATH")) / f"{title}.ass"
    if args.mode == "hybrid":
        run_hybrid(events, skip_before, video, output, movies, args.movie_ocr, movie_ranges)
    else:
        if not events:
            parser.error('此脚本仅包含非实时渲染视频，请使用 hybrid 模式')
        run_template(events, skip_before, str(video.resolve()), output)


if __name__ == "__main__":
    main()
