"""JSON schemas (strict, OpenAI response_format) + validators. Stdlib only."""
from __future__ import annotations

import json
from typing import Any, Callable, Optional

OBJECT_LIST_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "object_list",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["name", "description"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["objects"],
            "additionalProperties": False,
        },
    },
}

GROUNDING_OBJECT_LIST_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "grounding_object_list",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "bbox": {
                                "type": "array",
                                "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                                "minItems": 4,
                                "maxItems": 4,
                                "description": "[x1, y1, x2, y2] tight box, normalized 0-1000 image coords (0,0 top-left; 1000,1000 bottom-right).",
                            },
                        },
                        "required": ["name", "description"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["objects"],
            "additionalProperties": False,
        },
    },
}

ANOMALY_CANDIDATES_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "anomaly_candidates",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "anomalies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "phenomenon": {"type": "string"},
                            "explanation": {"type": "string"},
                        },
                        "required": ["name", "phenomenon", "explanation"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["anomalies"],
            "additionalProperties": False,
        },
    },
}

FINAL_ANOMALY_LIST_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "final_anomaly_list",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "anomalies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "observed_phenomenon": {"type": "string"},
                            "reasoning": {"type": "string"},
                            "severity_score": {"type": "integer", "minimum": 0, "maximum": 100},
                        },
                        "required": ["name", "observed_phenomenon", "reasoning", "severity_score"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["anomalies"],
            "additionalProperties": False,
        },
    },
}


def _require(obj: Any, key: str, typ: type) -> Any:
    if not isinstance(obj, dict):
        raise ValueError(f"expected object, got {type(obj).__name__}")
    v = obj.get(key)
    if not isinstance(v, typ) or (typ is str and not v.strip()):
        raise ValueError(f"missing/invalid field {key!r} (need non-empty {typ.__name__})")
    return v


def validate_object_list(payload: dict) -> list[dict]:
    items = _require(payload, "objects", list)
    if not items:
        raise ValueError("objects list is empty")
    out = []
    for it in items:
        out.append({"name": _require(it, "name", str), "description": _require(it, "description", str)})
    return out


def _valid_bbox(bb) -> Optional[list[int]]:
    if not isinstance(bb, list) or len(bb) != 4:
        return None
    if not all(isinstance(v, int) and 0 <= v <= 1000 for v in bb):
        return None
    x1, y1, x2, y2 = bb
    if not (x1 < x2 and y1 < y2):
        return None
    return bb


def validate_grounding_object_list(payload: dict) -> list[dict]:
    items = _require(payload, "objects", list)
    if not items:
        raise ValueError("objects list is empty")
    out = []
    for it in items:
        entry = {"name": _require(it, "name", str), "description": _require(it, "description", str)}
        bbox = _valid_bbox(it.get("bbox"))
        if bbox is not None:
            entry["bbox"] = bbox
        out.append(entry)
    return out


def validate_anomaly_candidates(payload: dict) -> list[dict]:
    items = _require(payload, "anomalies", list)
    out = []
    for it in items:
        out.append(
            {
                "name": _require(it, "name", str),
                "phenomenon": _require(it, "phenomenon", str),
                "explanation": _require(it, "explanation", str),
            }
        )
    return out


def validate_final_anomaly_list(payload: dict) -> list[dict]:
    items = _require(payload, "anomalies", list)
    out = []
    for it in items:
        sev = it.get("severity_score")
        if not isinstance(sev, int) or not (0 <= sev <= 100):
            raise ValueError(f"severity_score must be int in [0,100], got {sev!r}")
        out.append(
            {
                "name": _require(it, "name", str),
                "observed_phenomenon": _require(it, "observed_phenomenon", str),
                "reasoning": _require(it, "reasoning", str),
                "severity_score": sev,
            }
        )
    return out


STEP_SCHEMAS: dict[str, tuple[dict, Callable[[dict], list[dict]]]] = {
    "object_list": (OBJECT_LIST_SCHEMA, validate_object_list),
    "grounding_object_list": (GROUNDING_OBJECT_LIST_SCHEMA, validate_grounding_object_list),
    "anomaly_candidates": (ANOMALY_CANDIDATES_SCHEMA, validate_anomaly_candidates),
    "final_anomaly_list": (FINAL_ANOMALY_LIST_SCHEMA, validate_final_anomaly_list),
}


_JSON_TAIL_HINTS: dict[str, str] = {
    "grounding_object_list": (
        " For every object, ALSO provide a tight bounding box 'bbox' as [x1, y1, x2, y2] in "
        "NORMALIZED image coordinates from 0 to 1000 (0,0 = top-left corner; 1000,1000 = bottom-right corner)."
    ),
}


def json_tail(kind: str, compact: bool = False) -> str:
    """Append this to the final user text when running in json mode."""
    schema, _ = STEP_SCHEMAS[kind]
    example = json.dumps(schema["json_schema"]["schema"], ensure_ascii=False, indent=None if compact else 1)
    hint = _JSON_TAIL_HINTS.get(kind, "")
    return (
        "\n\n**OUTPUT REQUIREMENT (STRICT)**: Respond with ONLY a single JSON object that "
        "exactly matches this JSON schema. No markdown, no code fences, no extra commentary."
        + hint + "\n"
        f"Schema:\n{example}\n"
    )


def parse_json_text(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text)
