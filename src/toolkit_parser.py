"""Verbatim parsing subset from DreamGallery/HoshimiToolkit src/adv_csv.py.

Upstream commit: 6bac6a8ddc64b48b6818d8432c945908510a21c4
Source: https://github.com/DreamGallery/HoshimiToolkit/blob/6bac6a8ddc64b48b6818d8432c945908510a21c4/src/adv_csv.py
Only unrelated CSV writing/translation functions and imports were omitted.
Keep parsing functions unchanged; project-specific adapters live in adv_text.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


ASSIGNMENT = re.compile(r"(?:(?<=\[)|(?<= ))([A-Za-z_][A-Za-z_0-9]*)=")


TAG = re.compile(r"^\[([A-Za-z_][A-Za-z_0-9]*)\b")


FIELDS = {
    "message": {"text", "name"},
    "narration": {"text"},
    "title": {"title"},
    "choicegroup": {"text"},
    "choice": {"text"},
}


@dataclass(frozen=True)
class Field:
    identifier: str
    key: str
    category: str
    name: str
    source: str
    start: int
    end: int


def _value_end(line: str, start: int) -> int:
    depth = 0
    index = start
    while index < len(line):
        char = line[index]
        if char == "\\" and index + 1 < len(line):
            index += 2
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            if depth == 0:
                return index
            depth -= 1
        elif char == " " and depth == 0 and ASSIGNMENT.match(line, index + 1):
            return index
        index += 1
    raise ValueError("Unclosed adventure command")


def fields(script: str) -> list[Field]:
    result = []
    offset = 0
    for line_number, line in enumerate(script.splitlines(keepends=True), 1):
        command = line.rstrip("\r\n")
        match = TAG.match(command)
        if not match:
            offset += len(line)
            continue
        tag = match.group(1)
        wanted = FIELDS.get(tag, set())
        if not wanted:
            offset += len(line)
            continue
        assignments = list(ASSIGNMENT.finditer(command))
        name = ""
        for assignment in assignments:
            if assignment.group(1) == "name":
                name = command[assignment.end():_value_end(command, assignment.end())]
                break
        counts: dict[str, int] = {}
        for assignment in assignments:
            key = assignment.group(1)
            if key not in wanted:
                continue
            start = assignment.end()
            end = _value_end(command, start)
            value = command[start:end]
            if not value:
                continue
            counts[key] = counts.get(key, 0) + 1
            category = ("choice" if tag in {"choice", "choicegroup"} else
                        "narration" if tag == "narration" else key)
            result.append(Field(f"{line_number}:{key}:{counts[key]}", key, category, name,
                                value, offset + start, offset + end))
        offset += len(line)
    return result


def csv_identifier(item: Field) -> str:
    """Show the script command type while retaining the line and field ordinal."""
    line, _, ordinal = item.identifier.split(":")
    return f"{line}:{item.category}:{ordinal}"


def matches_csv_identifier(item: Field, identifier: str) -> bool:
    # Earlier CSVs used :text: for narration and choices. Both spellings locate
    # the same source field; patches continue to use the legacy stable key.
    return identifier in {item.identifier, csv_identifier(item)}
