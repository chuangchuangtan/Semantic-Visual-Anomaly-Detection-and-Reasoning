"""File IO helpers for JSON and NDJSON artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def load_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def append_ndjson(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_ndjson(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid NDJSON at {path}:{line_no}: {exc}") from exc


def strip_ndjson_prefix(value: str) -> Path:
    return Path(value.replace("ndjson:", "", 1))


def completed_paths(path: Path, success_only: bool = True) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_path = record.get("path")
            if not image_path:
                continue
            if not success_only or record.get("status") == "ok":
                done.add(str(image_path))
    return done

