"""Agent action protocol (dynamic schedule).

The controller emits ONE strict-JSON action per loop iteration (vLLM guided
decoding; no native tool-call dependency). The tool set is exactly 7:

    detect_objects / crop / inspect / check_relation / verify / global_scan / finish

The deterministic fixed schedule (default) executes the same tools in the
T+1+5N+2S+1 order (11+5N with T=S=3) without any controller LLM calls.
"""
from __future__ import annotations

import json

ACTIONS = (
    "detect_objects",
    "crop",
    "inspect",
    "check_relation",
    "verify",
    "global_scan",
    "finish",
)

FINAL_ITEM = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "observed_phenomenon": {"type": "string"},
        "reasoning": {"type": "string"},
        "severity_score": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["name", "observed_phenomenon", "reasoning", "severity_score"],
    "additionalProperties": False,
}

AGENT_ACTION_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_action",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "One short sentence: what you know and why this action next.",
                },
                "action": {"type": "string", "enum": list(ACTIONS)},
                "object": {"type": "string", "description": "Object name (EXACT, from OBJECTS table); empty string if not needed."},
                "members": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,
                    "maxItems": 3,
                    "description": "check_relation only: 2-3 EXACT object names; empty list otherwise.",
                },
                "depth": {"type": "string", "enum": ["full", "lite"], "description": "inspect only: 'lite' (describe) or 'full' (describe + anomaly candidates)."},
                "anomalies": {
                    "type": "array",
                    "items": FINAL_ITEM,
                    "minItems": 0,
                    "description": "finish only: pre-selected candidate anomalies (optional; empty = use all gathered evidence).",
                },
            },
            "required": ["thought", "action", "object", "members", "depth", "anomalies"],
            "additionalProperties": False,
        },
    },
}


class ActionError(ValueError):
    pass


def _fingerprint(action: dict) -> str:
    return json.dumps(
        {
            "action": action["action"],
            "object": action.get("object", ""),
            "members": sorted(action.get("members", []) or []),
            "depth": action.get("depth", ""),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def validate_agent_action(action: dict) -> dict:
    if not isinstance(action, dict):
        raise ActionError(f"action must be an object, got {type(action).__name__}")
    for field_name in ("thought", "action", "object", "members", "depth", "anomalies"):
        if field_name not in action:
            raise ActionError(f"missing field {field_name!r}")
    a = action["action"]
    if a not in ACTIONS:
        raise ActionError(f"unknown action {a!r}")
    if a in ("inspect", "crop") and not str(action.get("object") or "").strip():
        raise ActionError(f"action {a!r} requires 'object'")
    if a == "check_relation":
        members = [str(m).strip() for m in (action.get("members") or []) if str(m).strip()]
        if len(members) < 2:
            raise ActionError("check_relation requires 2-3 distinct members")
        if len(set(members)) != len(members):
            raise ActionError("check_relation members must be distinct")
        action["members"] = members[:3]
    if a == "finish" and action.get("anomalies") is None:
        action["anomalies"] = []
    return action


class ActionTracker:
    """Dedup + usage bookkeeping (local, zero cost)."""

    def __init__(self) -> None:
        self.seen: set = set()
        self.history: list = []
        self.inspected: set = set()
        self.detect_called = False
        self.verify_called = False
        self.global_scan_called = False

    def is_duplicate(self, action: dict) -> bool:
        fp = _fingerprint(action)
        if fp in self.seen:
            return True
        return False

    def record(self, action: dict, outcome: str) -> None:
        fp = _fingerprint(action)
        self.seen.add(fp)
        if action["action"] == "detect_objects":
            self.detect_called = True
        if action["action"] == "inspect":
            self.inspected.add((action.get("object", ""), action.get("depth", "")))
        if action["action"] == "verify":
            self.verify_called = True
        if action["action"] == "global_scan":
            self.global_scan_called = True
        args = ", ".join(
            filter(
                None,
                [
                    action.get("object") or "",
                    "+".join(sorted(action.get("members", []) or [])),
                    action.get("depth") if action["action"] == "inspect" else "",
                ],
            )
        )
        label = action["action"] + (f"({args})" if args else "")
        self.history.append(f"{label} -> {outcome[:100]}")
