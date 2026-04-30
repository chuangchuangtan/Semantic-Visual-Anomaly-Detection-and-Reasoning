#!/usr/bin/env python3
"""Evaluate Deepfake Classification binary outputs and filter Anom Detection results accordingly."""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anomreason_vllm.io import atomic_write_json, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process two prompt runs: evaluate Deepfake Classification yes/no accuracy using "
            "folder-derived labels, then clear Anom Detection results for fake images that "
            "Deepfake Classification classified incorrectly."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--anom-detection-json", required=True, type=Path, help="Anom Detection predictions.json for fake images.")
    parser.add_argument(
        "--classification-json",
        required=True,
        type=Path,
        help="Deepfake Classification predictions.json for fake+real images, where values are raw yes/no responses.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for post-processing outputs.")
    parser.add_argument(
        "--filtered-anom-detection-out",
        type=Path,
        default=None,
        help="Optional output path for filtered Anom Detection predictions. Default: <output-dir>/anom_detection_filtered.json",
    )
    parser.add_argument(
        "--classification-metrics-out",
        type=Path,
        default=None,
        help="Optional output path for Deepfake Classification accuracy metrics. Default: <output-dir>/classification_accuracy.json",
    )
    parser.add_argument(
        "--classification-details-out",
        type=Path,
        default=None,
        help="Optional output path for per-image Deepfake Classification evaluation details. Default: <output-dir>/classification_details.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    anom_detection_data = expect_dict(load_json(args.anom_detection_json), args.anom_detection_json)
    classification_data = expect_dict(load_json(args.classification_json), args.classification_json)

    classification_index = build_path_index(classification_data)
    classification_details, classification_summary = evaluate_classification(classification_data)
    filtered_anom_detection, filter_summary = filter_anom_detection(anom_detection_data, classification_index)

    filtered_anom_detection_out = args.filtered_anom_detection_out or args.output_dir / "anom_detection_filtered.json"
    classification_metrics_out = args.classification_metrics_out or args.output_dir / "classification_accuracy.json"
    classification_details_out = args.classification_details_out or args.output_dir / "classification_details.json"

    atomic_write_json(filtered_anom_detection_out, filtered_anom_detection)
    atomic_write_json(
        classification_metrics_out,
        {
            "classification_accuracy": classification_summary,
            "anom_detection_filtering": filter_summary,
            "inputs": {
                "anom_detection_json": str(args.anom_detection_json),
                "classification_json": str(args.classification_json),
            },
            "outputs": {
                "filtered_anom_detection_json": str(filtered_anom_detection_out),
                "classification_details_json": str(classification_details_out),
            },
        },
    )
    atomic_write_json(classification_details_out, classification_details)

    print(f"Saved Deepfake Classification metrics: {classification_metrics_out}")
    print(f"Saved Deepfake Classification details: {classification_details_out}")
    print(f"Saved filtered Anom Detection results: {filtered_anom_detection_out}")
    return 0


def expect_dict(data: Any, path: Path) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must be a JSON object mapping image paths to predictions.")
    return {str(key): value for key, value in data.items()}


def evaluate_classification(classification_data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    details: list[dict[str, Any]] = []
    counts = init_counts()
    invalid_count = 0

    for image_path, raw_prediction in sorted(classification_data.items()):
        expected = infer_expected_answer(image_path)
        predicted = normalize_yes_no(raw_prediction)
        correct = predicted == expected if predicted is not None else False

        label_key = "fake" if expected == "yes" else "real"
        counts["total"] += 1
        counts[label_key]["total"] += 1
        if correct:
            counts["correct"] += 1
            counts[label_key]["correct"] += 1
        if predicted is None:
            invalid_count += 1

        details.append(
            {
                "path": image_path,
                "expected": expected,
                "predicted": predicted,
                "raw_prediction": raw_prediction,
                "correct": correct,
                "label_source": label_key,
            }
        )

    summary = {
        "total": counts["total"],
        "correct": counts["correct"],
        "accuracy": safe_ratio(counts["correct"], counts["total"]),
        "invalid_or_unrecognized_predictions": invalid_count,
        "fake": {
            "total": counts["fake"]["total"],
            "correct": counts["fake"]["correct"],
            "accuracy": safe_ratio(counts["fake"]["correct"], counts["fake"]["total"]),
        },
        "real": {
            "total": counts["real"]["total"],
            "correct": counts["real"]["correct"],
            "accuracy": safe_ratio(counts["real"]["correct"], counts["real"]["total"]),
        },
    }
    return details, summary


def filter_anom_detection(anom_detection_data: dict[str, Any], classification_index: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    filtered = copy.deepcopy(anom_detection_data)
    summary = {
        "total_anom_detection_entries": len(anom_detection_data),
        "matched_by_exact_path": 0,
        "matched_by_unique_basename": 0,
        "missing_classification_match": 0,
        "ambiguous_basename_match": 0,
        "kept_fake_entries": 0,
        "cleared_fake_entries": 0,
        "clear_reason": "Anom Detection results are replaced with [] when Deepfake Classification is not a correct 'yes' prediction for that fake image.",
    }

    for image_path in sorted(anom_detection_data.keys()):
        expected = infer_expected_answer(image_path)
        if expected != "yes":
            raise SystemExit(
                "Anom Detection input is expected to contain only fake-image results under the '1_fake' folder. "
                f"Found non-fake path: {image_path}"
            )

        match_type, classification_value = lookup_classification_prediction(image_path, classification_index)
        if match_type == "exact":
            summary["matched_by_exact_path"] += 1
        elif match_type == "basename":
            summary["matched_by_unique_basename"] += 1
        elif match_type == "ambiguous":
            summary["ambiguous_basename_match"] += 1
            filtered[image_path] = []
            summary["cleared_fake_entries"] += 1
            continue
        else:
            summary["missing_classification_match"] += 1
            filtered[image_path] = []
            summary["cleared_fake_entries"] += 1
            continue

        predicted = normalize_yes_no(classification_value)
        if predicted == "yes":
            summary["kept_fake_entries"] += 1
        else:
            filtered[image_path] = []
            summary["cleared_fake_entries"] += 1

    return filtered, summary


def build_path_index(classification_data: dict[str, Any]) -> dict[str, Any]:
    basename_map: dict[str, list[tuple[str, Any]]] = {}
    for image_path, value in classification_data.items():
        basename_map.setdefault(Path(image_path).name, []).append((image_path, value))
    return {
        "exact": classification_data,
        "basename": basename_map,
    }


def lookup_classification_prediction(image_path: str, classification_index: dict[str, Any]) -> tuple[str, Any]:
    exact = classification_index["exact"]
    if image_path in exact:
        return "exact", exact[image_path]

    matches = classification_index["basename"].get(Path(image_path).name, [])
    if len(matches) == 1:
        return "basename", matches[0][1]
    if len(matches) > 1:
        return "ambiguous", None
    return "missing", None


def infer_expected_answer(image_path: str) -> str:
    normalized = image_path.replace("\\", "/")
    path_parts = [part for part in normalized.split("/") if part]
    if "1_fake" in path_parts:
        return "yes"
    if "0_real" in path_parts:
        return "no"
    raise SystemExit(
        "Could not infer label from path. Expected the image path to include either "
        f"'1_fake' or '0_real': {image_path}"
    )


def normalize_yes_no(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    tokens = re.findall(r"[a-z]+", text.lower())
    if not tokens:
        return None
    if tokens[0] == "yes":
        return "yes"
    if tokens[0] == "no":
        return "no"
    return None


def init_counts() -> dict[str, Any]:
    return {
        "total": 0,
        "correct": 0,
        "fake": {"total": 0, "correct": 0},
        "real": {"total": 0, "correct": 0},
    }


def safe_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())

