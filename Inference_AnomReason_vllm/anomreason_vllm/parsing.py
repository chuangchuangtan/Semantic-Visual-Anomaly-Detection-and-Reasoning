"""Prompt and response parsing helpers for semantic anomaly reasoning."""

from __future__ import annotations

import re
from typing import Any


DEFAULT_PROMPT = (
    "Analyze the provided AI-generated image to detect all semantic anomalies.\n"
    "Provide a detailed list of anomalies using the following structure:\n"
    "- @1. Name: [Phenomenon Name].\n"
    "- Observed Phenomenon: [Description].\n"
    "- Reasoning: [Why it is unrealistic or illogical].\n"
    "- Severity Score: [0-100 realism].\n"
)


def normalize_response_text(text: Any) -> str:
    """Return a safe string for parser input."""

    return "" if text is None else str(text).strip()


def parse_anomaly_response(text: Any) -> list[dict[str, str]]:
    """Parse model output into the JSON schema used by SemAP/SemF1 evaluation.

    The parser intentionally accepts small formatting variations from VLMs:
    markdown bullets, bold labels, case differences, and blank-line separated
    anomaly blocks.
    """

    entries = normalize_response_text(text)
    if not entries:
        return []

    blocks = re.split(r"\n\s*\n", entries)
    parsed: list[dict[str, str]] = []

    for block in blocks:
        entry = block.strip()
        if not entry:
            continue
        if not re.search(r"Phenomenon", entry, re.IGNORECASE):
            continue
        if not re.search(r"Reasoning", entry, re.IGNORECASE):
            continue
        if not re.search(r"Severity\s*Score|Score", entry, re.IGNORECASE):
            continue

        cleaned = re.sub(r"\n\s*-\s*", "\n- ", entry)
        cleaned = cleaned.replace("**:", ":").replace("**", "")

        name = _match_until(cleaned, r"^(.*?)Observed\s+Phenomenon")
        observed = _match_until(cleaned, r"Observed\s+Phenomenon\s*:(.*?)Reasoning")
        reasoning = _match_until(cleaned, r"Reasoning\s*:(.*?)(?:Severity\s*Score|Score)")
        severity = _match_until(cleaned, r"(?:Severity\s*Score|Score)\s*:(.*)")

        parsed.append(
            {
                "Name": _clean_name(name),
                "Observed Phenomenon": _clean_field(observed),
                "Reasoning": _clean_field(reasoning),
                "Severity Score": _clean_field(severity),
                "entry": cleaned.strip(),
            }
        )

    return parsed


def _match_until(text: str, pattern: str) -> str:
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else ""


def _clean_field(value: str) -> str:
    return value.strip().strip("*").strip("-").strip()


def _clean_name(value: str) -> str:
    cleaned = _clean_field(value)
    cleaned = re.sub(r"^@?\d+[\).]?\s*", "", cleaned)
    cleaned = re.sub(r"^Name\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return _clean_field(cleaned)
