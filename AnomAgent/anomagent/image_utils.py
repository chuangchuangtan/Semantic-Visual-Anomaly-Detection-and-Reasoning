from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image, ImageOps


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
