"""Project adapters around the vendored HoshimiToolkit ADV parser."""
import csv
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import re

from src.read_ini import config
from src.toolkit_parser import ASSIGNMENT, TAG, _value_end, fields, matches_csv_identifier


def script_path(filename):
    path = Path(filename)
    return path if path.is_absolute() or path.is_file() else Path(config.get('File Path', 'TXT_PATH')) / path


def command_attributes(content):
    """Read clip metadata with Toolkit's assignment/value-boundary scanner."""
    line = content.rstrip('\r\n')
    tag = TAG.match(line)
    if not tag:
        raise ValueError('Malformed ADV command')
    result, consumed = {}, 0
    for assignment in ASSIGNMENT.finditer(line):
        if assignment.start() < consumed:
            continue
        key = assignment.group(1)
        if key in result:
            raise ValueError(f'Duplicate ADV attribute: {key}')
        end = _value_end(line, assignment.end())
        result[key] = line[assignment.end():end]
        consumed = end
    return tag.group(1), result


def extract(filename, csv_filename=None):
    path = script_path(filename)
    raw = path.read_bytes()
    script = raw.decode('utf-8-sig')
    items = fields(script)
    if csv_filename:
        verify_csv(Path(csv_filename), path, raw, items)
    lines = script.splitlines()
    dialogues, seen = [], set()
    for item in items:
        # Toolkit also exposes titles, names and nested choices, which do not
        # become timed message/narration events in this subtitle application.
        if item.key != 'text' or item.category not in ('text', 'narration'):
            continue
        line_number = int(item.identifier.split(':')[0])
        if line_number in seen:
            raise ValueError(f'{path}:{line_number}: multiple subtitle text fields')
        seen.add(line_number)
        line = lines[line_number - 1]
        try:
            get_clip(line)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f'{path}:{line_number}: invalid subtitle clip') from exc
        dialogues.append(line)
    dialogues.sort(key=lambda line: get_clip(line)['_startTime'])
    return dialogues


def verify_csv(csv_path, source_path, raw, items):
    """Use Toolkit's field IDs for both current and legacy Hoshimi CSVs."""
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['id', 'name', 'text', 'trans']:
            raise ValueError('CSV columns must be id,name,text,trans')
        rows = list(reader)
    if any(set(row) != {'id', 'name', 'text', 'trans'} or any(v is None for v in row.values()) for row in rows):
        raise ValueError('CSV has missing or extra cells')
    if (len(rows) < 2 or rows[-2]['id'] != 'info' or rows[-1]['id'] != '译者'
            or rows[-2]['name'] != source_path.name
            or rows[-2]['text'] != hashlib.sha256(raw).hexdigest()):
        raise ValueError('CSV source filename/checksum does not match the original TXT')
    items = [item for item in items if item.key != 'name']
    seen = set()
    for row in rows[:-2]:
        item = next((item for item in items if matches_csv_identifier(item, row['id'])), None)
        if item is None or item.identifier in seen:
            raise ValueError(f'Invalid or repeated CSV field ID: {row["id"]}')
        seen.add(item.identifier)
        if item.source != row['text']:
            raise ValueError(f'CSV original text differs from TXT: {row["id"]}')
    if len(seen) != len(items):
        raise ValueError('CSV is missing original script fields')


def video_references(filename, dialogues, offset=0):
    """定位非实时渲染视频引用在脚本台词序列中的位置。

    脚本时间只用于判断顺序，录屏范围由匹配后的帧号确定，
    支持开头、中间、结尾及连续多段非实时渲染视频。
    """
    clips = [get_clip(line) for line in dialogues]
    result = []
    for line_number, line in enumerate(script_path(filename).read_text(encoding='utf-8-sig').splitlines(), 1):
        tag = TAG.match(line)
        if tag is None or tag.group(1) != 'video':
            continue
        attrs, clip = command_attributes(line)[1], get_clip(line)
        if not attrs.get('video') or clip['_duration'] <= 0:
            continue
        start, end = clip['_startTime'], clip['_startTime'] + clip['_duration']
        result.append({
            'asset': attrs['video'], 'script': script_path(filename).name, 'line': line_number,
            'script_start': start, 'script_duration': clip['_duration'],
            'segment_start': offset, 'segment_end': offset + len(dialogues),
            'before': [offset + i for i, c in enumerate(clips)
                       if c['_startTime'] + c['_duration'] <= start + .1],
            'after': [offset + i for i, c in enumerate(clips) if c['_startTime'] >= end - .1],
        })
    return sorted(result, key=lambda ref: ref['script_start'])


def get_title(filename):
    script = script_path(filename).read_text(encoding='utf-8-sig')
    item = next((item for item in fields(script) if item.key == 'title'), None)
    if item is None:
        return script_path(filename).stem
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', item.source).replace(' ', '_')


def get_text(content):
    item = next((item for item in fields(content)
                 if item.key == 'text' and item.category in ('text', 'narration')), None)
    if item is None:
        raise ValueError('Not a nonempty message/narration command')
    player = config.get('Info', 'player_name').strip() or 'マネージャー'
    text = item.source.replace('{user}', player)
    gray = item.category == 'narration' or '（' in text or '）' in text
    if item.category != 'narration':
        text = text.replace('（', '').replace('）', '')
    return text.replace(r'\n', r'\N'), gray


def get_name(content):
    item = next((item for item in fields(content) if item.key == 'name'), None)
    return item.source if item is not None else ''


def get_clip(content):
    raw = command_attributes(content)[1]['clip']
    data = json.loads(raw.replace(r'\{', '{').replace(r'\}', '}'))
    for key in ('_startTime', '_duration'):
        value = float(data[key])
        if not 0 <= value < float('inf'):
            raise ValueError(f'Invalid clip {key}: {value}')
        data[key] = value
    return data


def to_time(clip_time):
    centiseconds = max(0, round(clip_time * 100))
    return format_centiseconds(centiseconds)


def frame_to_time(frame, fps):
    """Preserve legacy ASS centisecond truncation after CFR frame / FPS timing.

    Rational arithmetic avoids floating-point underflow at exact centiseconds.
    The frame index itself is never rounded or shifted to fit an old subtitle.
    """
    centiseconds = max(0, int(Fraction(frame * 100) / Fraction(str(fps))))
    return format_centiseconds(centiseconds)


def format_centiseconds(centiseconds):
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    seconds, fraction = divmod(remainder, 100)
    return f'{hours}:{minutes:02d}:{seconds:02d}.{fraction:02d}'


def end_time(startTime, duration):
    return to_time(startTime + duration)
