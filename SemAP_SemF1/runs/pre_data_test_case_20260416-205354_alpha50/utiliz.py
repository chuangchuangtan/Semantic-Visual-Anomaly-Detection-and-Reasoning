#!/usr/bin/env python3
"""Utility helpers for the SemAP / SemF1 evaluation pipeline."""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

TEXT_FIELDS = ("Observed Phenomenon", "Reasoning")


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""
    with Path(path).open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(data: Any, path: str | Path) -> None:
    """Write JSON with UTF-8 encoding and pretty indentation."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)
    logging.info("Saved JSON: %s", output_path)


def setup_logging(save_dir: str | Path) -> Path:
    """Reset root logging handlers and log to both console and file."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "run.log"

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    logging.info("Log file: %s", log_path)
    return log_path


def normalize_image_key(key: str) -> str:
    """Normalize path-like image identifiers to their basename."""
    return Path(str(key)).name


def _normalize_text_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_entry(record: Any) -> Dict[str, str]:
    """Normalize one anomaly entry into the public metric schema."""
    if not isinstance(record, dict):
        raise TypeError(f"Each entry must be a dict, but received {type(record)!r}.")

    normalized = {
        "Name": _normalize_text_value(record.get("Name", "")),
        "Observed Phenomenon": _normalize_text_value(record.get("Observed Phenomenon", "")),
        "Reasoning": _normalize_text_value(record.get("Reasoning", "")),
        "Severity Score": _normalize_text_value(record.get("Severity Score", "")),
    }
    return normalized


def normalize_prediction_dict(raw_data: Any) -> Dict[str, List[Dict[str, str]]]:
    """Normalize prediction or ground-truth JSON into basename-indexed lists."""
    if not isinstance(raw_data, dict):
        raise TypeError("Prediction / ground-truth JSON must be a dict keyed by image id.")

    normalized: Dict[str, List[Dict[str, str]]] = {}
    source_by_key: Dict[str, str] = {}
    for raw_key, value in raw_data.items():
        normalized_key = normalize_image_key(raw_key)
        if normalized_key in normalized and source_by_key[normalized_key] != str(raw_key):
            raise ValueError(
                "Different original keys collapse to the same basename after normalization: "
                f"{source_by_key[normalized_key]!r} and {raw_key!r} -> {normalized_key!r}."
            )
        if value is None:
            entries: List[Dict[str, str]] = []
        elif isinstance(value, list):
            entries = [normalize_entry(item) for item in value]
        else:
            raise TypeError(
                f"Image {raw_key!r} must map to a list of anomaly entries, not {type(value)!r}."
            )

        normalized[normalized_key] = entries
        source_by_key[normalized_key] = str(raw_key)
    return normalized


def _deduplicate_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def extract_subset_keys(raw_subset: Any) -> List[str]:
    """Support several common split-file formats and return normalized keys."""
    if raw_subset is None:
        return []

    keys: List[str] = []
    if isinstance(raw_subset, dict):
        keys.extend(raw_subset.keys())
    elif isinstance(raw_subset, list):
        for item in raw_subset:
            if isinstance(item, str):
                keys.append(item)
                continue
            if isinstance(item, dict):
                if len(item) == 1:
                    keys.append(next(iter(item.keys())))
                    continue
                for candidate_field in ("image", "image_id", "image_path", "path", "file_name"):
                    if candidate_field in item:
                        keys.append(str(item[candidate_field]))
                        break
                else:
                    raise ValueError(
                        "Unsupported subset item format. Expected a string, a single-key dict, "
                        "or a dict containing one of: image, image_id, image_path, path, file_name."
                    )
                continue
            raise TypeError(f"Unsupported subset item type: {type(item)!r}.")
    else:
        raise TypeError("Subset JSON must be a dict or a list.")

    return _deduplicate_preserve_order(normalize_image_key(item) for item in keys)


def parse_float_csv(raw_value: str, *, field_name: str) -> List[float]:
    """Parse a comma-separated string of floats."""
    values: List[float] = []
    for piece in raw_value.split(","):
        stripped = piece.strip()
        if not stripped:
            continue
        try:
            values.append(float(stripped))
        except ValueError as exc:
            raise ValueError(f"Invalid float in `{field_name}`: {stripped!r}.") from exc

    if not values:
        raise ValueError(f"`{field_name}` must contain at least one float.")
    return values


def auto_find_largest_json(search_root: str | Path) -> Optional[Path]:
    """Find the largest JSON file below `search_root`."""
    search_root = Path(search_root)
    largest_path: Optional[Path] = None
    largest_size = -1

    for path in search_root.rglob("*.json"):
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > largest_size:
            largest_size = size
            largest_path = path

    return largest_path


def response_to_entries(entries_text: str) -> List[Dict[str, str]]:
    """
    Parse a plain-text model response into structured anomaly entries.

    Each entry is expected to contain:
    - Observed Phenomenon
    - Reasoning
    - Severity Score
    """
    if not entries_text or not entries_text.strip():
        return []

    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", entries_text.strip()) if chunk.strip()]
    parsed_entries: List[Dict[str, str]] = []

    for chunk in chunks:
        if any(keyword not in chunk for keyword in ("Observed Phenomenon", "Reasoning", "Severity Score")):
            continue

        cleaned = re.sub(r"\n\s*-\s*", "\n- ", chunk)
        cleaned = cleaned.replace("**:", ":")

        def _match(pattern: str) -> str:
            match = re.search(pattern, cleaned, flags=re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""

        name = _match(r"^(.*?)Observed Phenomenon\s*:")
        observed = _match(r"Observed Phenomenon\s*:(.*?)Reasoning\s*:")
        reasoning = _match(r"Reasoning\s*:(.*?)Severity Score\s*:")
        severity = _match(r"Severity Score\s*:(.*)$")

        parsed_entries.append(
            {
                "Name": name.strip("*- "),
                "Observed Phenomenon": observed.strip("*- "),
                "Reasoning": reasoning.strip("*- "),
                "Severity Score": severity.strip("*- "),
                "entry": cleaned,
            }
        )

    return parsed_entries
