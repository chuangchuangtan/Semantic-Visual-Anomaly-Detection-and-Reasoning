from __future__ import annotations

import re


def parse_object_descriptions(text: str) -> dict[str, str]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty object description text")

    pattern = re.compile(r"#([^#\n]+)#\s*:\s*")
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError("Expected '#Name#: Description' blocks, but none were found.")

    objects: dict[str, str] = {}
    for idx, match in enumerate(matches):
        name = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        desc = text[start:end].strip()
        if name:
            objects[name] = desc

    if not objects:
        raise ValueError("Parsed object description text, but got no objects.")
    return objects


def parse_anomaly_list(text: str) -> list[dict]:
    text = (text or "").strip()
    if not text:
        return []

    chunks = re.split(r"\n(?=@\d+\.)", text)
    if len(chunks) == 1:
        chunks = re.split(r"\n(?=\d+\.)", text)

    label_pattern = re.compile(
        r"\*\*(Name|Observed Phenomenon|Reasoning|Severity Score)\*\*\s*:\s*",
        re.IGNORECASE,
    )
    results: list[dict] = []

    for chunk in chunks:
        matches = list(label_pattern.finditer(chunk))
        if not matches:
            continue

        fields: dict[str, str] = {}
        for idx, match in enumerate(matches):
            label = match.group(1).strip().lower()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(chunk)
            value = chunk[start:end].strip()
            value = re.sub(r"^[\-\s]+", "", value)
            fields[label] = value.strip().strip("*").strip()

        item = {
            "Name": fields.get("name", ""),
            "Observed Phenomenon": fields.get("observed phenomenon", ""),
            "Reasoning": fields.get("reasoning", ""),
            "Severity Score": fields.get("severity score", ""),
        }
        if any(item.values()):
            results.append(item)

    return results
