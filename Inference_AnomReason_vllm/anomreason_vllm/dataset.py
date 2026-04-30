"""Dataset loading and image-path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


IMAGE_KEYS = (
    "image",
    "image_path",
    "path",
    "file_name",
    "filename",
    "img",
    "img_path",
)


def extract_image_references(data: Any) -> list[str]:
    """Extract image references from common JSON dataset layouts."""

    if isinstance(data, dict):
        return [str(key) for key in data.keys()]

    if not isinstance(data, list):
        raise TypeError("Dataset JSON must be a list or an object mapping image paths to annotations.")

    refs: list[str] = []
    for index, item in enumerate(data):
        if isinstance(item, str):
            refs.append(item)
            continue
        if not isinstance(item, dict):
            raise TypeError(f"Dataset item {index} must be a string or object, got {type(item).__name__}.")

        ref = next((item[key] for key in IMAGE_KEYS if key in item and item[key]), None)
        if ref is None and len(item) == 1:
            ref = next(iter(item.keys()))
        if ref is None:
            raise KeyError(
                f"Dataset item {index} does not contain an image key. "
                f"Supported keys: {', '.join(IMAGE_KEYS)}."
            )
        refs.append(str(ref))
    return refs


def resolve_image_paths(
    references: Iterable[str],
    image_root: Path,
    path_mode: str = "relative",
    skip_missing: bool = True,
) -> tuple[list[str], list[dict[str, str]]]:
    """Resolve dataset references to local image files.

    path_mode:
    - basename: image_root / basename(reference)
    - relative: image_root / reference
    - absolute: use reference as-is when absolute, otherwise image_root / reference
    """
    print(f"Resolving {len(references)} image references with path_mode={path_mode}...")
    images: list[str] = []
    missing: list[dict[str, str]] = []
    for reference in references:
        candidate = resolve_one_image_path(reference, image_root, path_mode)
        if candidate.exists():
            images.append(str(candidate))
            continue
        missing.append({"reference": reference, "resolved_path": str(candidate)})
        if not skip_missing:
            raise FileNotFoundError(f"Image not found for {reference}: {candidate}")
    return images, missing


def resolve_one_image_path(reference: str, image_root: Path, path_mode: str) -> Path:
    ref_path = Path(reference)
    if path_mode == "basename":
        return image_root / ref_path.name
    if path_mode == "relative":
        return image_root / ref_path
    if path_mode == "absolute":
        return ref_path if ref_path.is_absolute() else image_root / ref_path
    raise ValueError("path_mode must be one of: basename, relative, absolute")

