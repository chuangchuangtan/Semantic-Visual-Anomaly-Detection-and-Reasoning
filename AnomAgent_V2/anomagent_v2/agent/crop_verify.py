"""Crop accuracy verification (P1-3).

Unreliable crops are replaced by the FULL image so that downstream steps
(inspect step2 / relation union crops) never analyze the wrong region.

Decision rules (all local, zero LLM cost):
  no_bbox          merged object has no bbox from any draft            -> full
  scene_level      bbox area > crop_area_max (default 60%) of image    -> full
  too_small        bbox area < crop_area_min (default 1%) of image     -> full
  drafts_disagree  mean pairwise IoU across >=2 same-name draft boxes  -> full
                   below crop_iou_min (default 0.30)
  crop             otherwise; a padded crop URL is precomputed

Optional post-hoc check (mention_rerun): if the descriptor text produced
from a crop does not mention the object name, the inspect step is re-run
once with the full image (at most one such rerun per object).
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from ..image_utils import bbox_area_frac, crop_bbox_data_url


@dataclass
class CropDecision:
    name: str
    status: str  # "crop" | "full"
    bbox: Optional[list] = None
    area_frac: float = 0.0
    iou: Optional[float] = None
    reason: str = ""
    crop_url: Optional[str] = None
    crop_info: dict = field(default_factory=dict)


def _norm(name: str) -> str:
    return " ".join(name.lower().split())


def iou(b1: list, b2: list) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def mean_pairwise_iou(boxes: list) -> Optional[float]:
    if len(boxes) < 2:
        return None
    vals = [iou(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:]]
    return statistics.fmean(vals) if vals else None


def match_draft_boxes(merged_names: list, drafts: list) -> dict:
    """Map merged object names to per-draft bbox lists (exact + fuzzy >= 0.75)."""
    draft_by_name: dict = {}
    draft_names: list = []
    for draft in drafts:
        for o in draft or []:
            if not isinstance(o, dict) or "name" not in o:
                continue
            key = _norm(o["name"])
            draft_names.append(key)
            if o.get("bbox"):
                draft_by_name.setdefault(key, []).append(list(o["bbox"]))

    out: dict = {}
    for name in merged_names:
        key = _norm(name)
        cands = draft_by_name.get(key)
        if not cands:
            best_key, best_ratio = None, 0.0
            for dn in draft_names:
                ratio = SequenceMatcher(None, key, dn).ratio()
                if ratio > best_ratio:
                    best_ratio, best_key = ratio, dn
            if best_key is not None and best_ratio >= 0.75:
                cands = draft_by_name.get(best_key)
        if cands:
            out[name] = cands
    return out


def assess_crops(
    image_path: str,
    img_w: int,
    img_h: int,
    merged_names: list,
    draft_boxes: dict,
    *,
    area_max: float = 0.60,
    area_min: float = 0.01,
    iou_min: float = 0.30,
    image_format: str = "png",
    jpeg_quality: int = 85,
) -> dict:
    """Return {object_name: CropDecision}. Objects without drafts are 'full/no_bbox'."""
    decisions: dict = {}
    for name in merged_names:
        boxes = draft_boxes.get(name)
        if not boxes:
            decisions[name] = CropDecision(name=name, status="full", reason="no_bbox")
            continue
        median_box = [int(statistics.median(c[i] for c in boxes)) for i in range(4)]
        frac = bbox_area_frac(median_box)
        p_iou = mean_pairwise_iou(boxes)
        decision = CropDecision(name=name, status="crop", bbox=median_box, area_frac=frac, iou=p_iou)
        if frac > area_max:
            decision.status, decision.reason = "full", "scene_level"
        elif frac < area_min:
            decision.status, decision.reason = "full", "too_small"
        elif p_iou is not None and p_iou < iou_min:
            decision.status, decision.reason = "full", "drafts_disagree"
        if decision.status == "crop":
            crop = crop_bbox_data_url(
                image_path, median_box, padding_frac=0.15, min_side=256, max_side=1024,
                image_format=image_format, jpeg_quality=jpeg_quality,
            )
            if crop is None:
                decision.status, decision.reason = "full", "crop_encode_failed"
            else:
                decision.crop_url, decision.crop_info = crop
        decisions[name] = decision
    return decisions


_STOPWORDS = {"a", "an", "the", "of", "in", "on", "with", "and", "or", "to", "at", "by", "for"}


def name_words(name: str) -> list:
    words = [w for w in _norm(name).split() if len(w) > 2 and w not in _STOPWORDS]
    return words


def mentions_object(text: str, name: str) -> bool:
    """Heuristic: does the description text plausibly refer to this object?
    Requires the full normalized name OR at least half of its significant words
    (word-token match: 'car' matches 'car'/'cars' but not 'scarf')."""
    if not text:
        return False
    low = _norm(text)
    full = _norm(name)
    if full and re.search(r"(?<![a-z0-9])" + re.escape(full) + r"(?![a-z0-9])", low):
        return True
    words = name_words(name)
    if not words:
        return True  # uninformative name; do not trigger a rerun
    tokens = re.findall(r"[a-z0-9]+", low)
    hits = sum(1 for w in words if any(t == w or t.startswith(w) for t in tokens))
    return hits / len(words) >= 0.5
