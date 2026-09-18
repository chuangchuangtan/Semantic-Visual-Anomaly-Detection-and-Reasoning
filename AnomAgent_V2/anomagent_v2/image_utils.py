from __future__ import annotations

import math
import base64
import io
import json
import os
from typing import Any, Optional

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")


def collect_image_paths(inputs) -> list[str]:
    """Shared image collector: files or directories (recursive). Sorted, deduped, absolute."""
    image_paths: list[str] = []
    for item in inputs:
        if os.path.isdir(item):
            for root, _, files in os.walk(item):
                for filename in files:
                    if filename.lower().endswith(IMAGE_EXTENSIONS):
                        image_paths.append(os.path.abspath(os.path.join(root, filename)))
        elif os.path.isfile(item):
            image_paths.append(os.path.abspath(item))
        else:
            raise FileNotFoundError(f"Input path not found: {item}")
    return sorted(set(image_paths))

def resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    if max(width, height) <= max_side:
        return image
    scale = max_side / float(max(width, height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resample = getattr(Image, "Resampling", Image).LANCZOS
    return image.resize((new_width, new_height), resample=resample)


def encode_image_to_data_url(
    image_path: str,
    *,
    max_side: int = 2048,
    image_format: str = "png",
    jpeg_quality: int = 85,
) -> tuple[str, dict[str, Any]]:
    image_format = (image_format or "png").lower()
    if image_format == "jpg":
        image_format = "jpeg"
    if image_format not in {"png", "jpeg"}:
        raise ValueError(f"Unsupported image_format: {image_format!r} (use 'png' or 'jpeg').")

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.copy()

    original_size = image.size
    if max_side:
        image = resize_max_side(image, int(max_side))

    buffered = io.BytesIO()
    if image_format == "jpeg":
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(buffered, format="JPEG", quality=int(jpeg_quality), optimize=True)
        mime_type = "image/jpeg"
    else:
        image.save(buffered, format="PNG")
        mime_type = "image/png"

    encoded_bytes = buffered.getvalue()
    encoded_str = base64.b64encode(encoded_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded_str}"

    info: dict[str, Any] = {
        "original_size": list(original_size),
        "encoded_size": list(image.size),
        "encoded_format": image_format,
        "encoded_bytes": len(encoded_bytes),
    }
    return data_url, info


def encode_image_variants(
    image_path: str,
    *,
    max_side: int = 2048,
    image_format: str = "png",
    jpeg_quality: int = 85,
) -> dict[str, str]:
    """Tiered encodings: global / medium (half) / local (quarter)."""
    out: dict[str, str] = {}
    base_info: dict[str, Any] = {}
    for tier, side in (("global", int(max_side)), ("medium", int(max_side) // 2), ("local", int(max_side) // 4)):
        url, info = encode_image_to_data_url(
            image_path, max_side=side, image_format=image_format, jpeg_quality=jpeg_quality
        )
        out[tier] = url
        if tier == "global":
            base_info = info
    out["info"] = json.dumps(base_info)
    return out

def bbox_area_frac(box: list[int]) -> float:
    """Fraction of the image covered by a normalized 0-1000 box."""
    x1, y1, x2, y2 = box
    return max(0.0, (x2 - x1) / 1000.0) * max(0.0, (y2 - y1) / 1000.0)


def union_bbox(boxes: list[list[int]]) -> list[int]:
    xs1 = [b[0] for b in boxes]
    ys1 = [b[1] for b in boxes]
    xs2 = [b[2] for b in boxes]
    ys2 = [b[3] for b in boxes]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def bbox_distance(b1: list[int], b2: list[int]) -> float:
    cx1, cy1 = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
    cx2, cy2 = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0
    return math.hypot(cx1 - cx2, cy1 - cy2)


def crop_bbox_data_url(
    image_path: str,
    bbox_norm: list[int],
    *,
    padding_frac: float = 0.15,
    min_side: int = 256,
    max_side: int = 1024,
    image_format: str = "png",
    jpeg_quality: int = 85,
    max_area_frac: float = 0.85,
) -> Optional[tuple[str, dict]]:
    """Crop a normalized (0-1000) bbox with padding and encode as a data URL.

    Returns (data_url, info) or None when the box is invalid/degenerate or
    covers too much of the image (caller then falls back to the full image).
    """
    from PIL import Image

    try:
        x1n, y1n, x2n, y2n = [int(v) for v in bbox_norm]
    except Exception:
        return None
    if not (0 <= x1n < x2n <= 1000 and 0 <= y1n < y2n <= 1000):
        return None
    try:
        with Image.open(image_path) as im:
            img_w, img_h = im.size
    except Exception:
        return None
    x1p, y1p = x1n / 1000.0 * img_w, y1n / 1000.0 * img_h
    x2p, y2p = x2n / 1000.0 * img_w, y2n / 1000.0 * img_h
    w, h = x2p - x1p, y2p - y1p
    if w * h / float(img_w * img_h) > max_area_frac:
        return None
    if max(w, h) < 0.04 * max(img_w, img_h):
        return None
    pad = padding_frac * max(w, h)
    x1p, y1p, x2p, y2p = x1p - pad, y1p - pad, x2p + pad, y2p + pad
    if x2p - x1p < min_side:
        grow = (min_side - (x2p - x1p)) / 2.0
        x1p, x2p = x1p - grow, x2p + grow
    if y2p - y1p < min_side:
        grow = (min_side - (y2p - y1p)) / 2.0
        y1p, y2p = y1p - grow, y2p + grow
    x1p, y1p = max(0.0, x1p), max(0.0, y1p)
    x2p, y2p = min(float(img_w), x2p), min(float(img_h), y2p)
    box = (int(round(x1p)), int(round(y1p)), int(round(x2p)), int(round(y2p)))
    if box[2] - box[0] < 32 or box[3] - box[1] < 32:
        return None
    try:
        with Image.open(image_path) as im:
            crop = im.crop(box)
            if max(crop.size) > max_side:
                ratio = max_side / float(max(crop.size))
                crop = crop.resize(
                    (max(1, int(crop.width * ratio)), max(1, int(crop.height * ratio))),
                    Image.LANCZOS,
                )
            buf = io.BytesIO()
            fmt = "JPEG" if image_format == "jpeg" else "PNG"
            if fmt == "JPEG" and crop.mode not in ("RGB", "L"):
                crop = crop.convert("RGB")
            save_kwargs = {"quality": jpeg_quality} if fmt == "JPEG" else {}
            crop.save(buf, format=fmt, **save_kwargs)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        mime = "image/jpeg" if fmt == "JPEG" else "image/png"
        return f"data:{mime};base64,{b64}", {"crop_box": list(box), "crop_size": list(crop.size)}
    except Exception:
        return None
