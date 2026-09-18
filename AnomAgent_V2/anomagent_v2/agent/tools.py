"""The 7 tools of the AnomAgent_V2 controller.

Every tool is a thin, stateless wrapper around one or more V1 analysis steps
(same prompts, same sampling parameters). Tools never talk to the API
directly - they go through P0Core.chat / P0Core.chat_json so that routing,
token elasticity, the step cache, the per-step ledger and the viz recorder
all apply uniformly.

    detect_objects  T discovery drafts (+1 merge) -> {name: desc} (+ bboxes)
    crop(object)    local crop-vs-full decision (crop accuracy check, zero LLM)
    inspect(object) descriptor step1 (+ step2 anomaly candidates when full)
    check_relation(object) relationship step1 + step2 + per-object summarizer step1
    verify          summarizer step2 x S
    global_scan     global single-shot reasoning x S
    finish          final structured step3
"""
from __future__ import annotations

import concurrent.futures
import json
import statistics
from dataclasses import dataclass, field
from typing import Optional

from .. import parsing
from ..image_utils import bbox_area_frac, bbox_distance, crop_bbox_data_url, union_bbox
from ..prompts import (
    SYSTEM_PROMPT,
    build_messages_analyze_all_objects,
    build_messages_descriptor_step1,
    build_messages_descriptor_step2,
    build_messages_multi_analyze_all_objects,
    build_messages_relationship_step1,
    build_messages_relationship_step2,
    build_messages_step1_reasoning,
    build_messages_summarizer_step1,
    build_messages_summarizer_step2,
    build_messages_summarizer_step3,
)
from .crop_verify import assess_crops, match_draft_boxes, mentions_object


@dataclass
class ToolResult:
    ok: bool = True
    summary: str = ""
    data: dict = field(default_factory=dict)


def _parallel(core, fn, items) -> dict:
    """Run fn(item) over items with the configured worker pool (V1 parity)."""
    out: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=core.config.max_workers) as executor:
        futures = {executor.submit(fn, item): item for item in items}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            out[item] = future.result()
    return out


class AgentTools:
    def __init__(self, core, ctx: dict) -> None:
        self.core = core
        self.ctx = ctx
        self.cfg = core.config
        self.system_prompt = SYSTEM_PROMPT
        self.expert_prompt = ""  # kept empty, same as V1 (reserved for extensions)

    @property
    def log(self):
        return self.core.logger

    def _img(self, step: str, object_name: Optional[str] = None) -> str:
        return self.core.image_for_step(self.ctx, step, object_name)

    # ---------------- 1. detect_objects ----------------

    def _objects_list_to_dict(self, objs: list) -> dict:
        """name->desc; duplicate names get _2/_3 suffixes (no silent overwrite)."""
        out: dict = {}
        seen: dict = {}
        for o in objs:
            name = str(o.get("name", "")).strip()
            if not name:
                continue
            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[name]}"
            else:
                seen[name] = 1
            out[name] = o.get("description", "")
        return out

    def detect_objects(self) -> ToolResult:
        cfg = self.cfg
        ctx = self.ctx
        if cfg.output_mode == "json":
            kind = "grounding_object_list" if cfg.grounding else "object_list"
            if cfg.num_get_objects > 1:
                drafts = _parallel(
                    self.core,
                    lambda i: self._analyze_all_objects_once(kind=kind, sample=i),
                    range(cfg.num_get_objects),
                )
                drafts = [drafts[i] for i in range(cfg.num_get_objects)]
            else:
                drafts = [self._analyze_all_objects_once(kind=kind)]
            ctx["drafts"] = drafts
            drafts_text = "\n".join(json.dumps(d, ensure_ascii=False) for d in drafts)
            merged = self._multi_analyze_all_objects(drafts_text=drafts_text, kind=kind)
        else:
            if cfg.num_get_objects > 1:
                drafts = _parallel(
                    self.core,
                    lambda i: self._analyze_all_objects_once_text(sample=i),
                    range(cfg.num_get_objects),
                )
                drafts = [drafts[i] for i in range(cfg.num_get_objects)]
            else:
                drafts = [self._analyze_all_objects_once_text()]
            ctx["drafts"] = drafts
            merged = self._multi_analyze_all_objects_text("\n------\n".join(drafts))
        objects = self._objects_list_to_dict(merged)
        if not objects:
            raise RuntimeError("object discovery produced no objects")
        if cfg.max_objects is not None:
            objects = {k: objects[k] for k in list(objects.keys())[: cfg.max_objects]}
        ctx["objects"] = objects
        self._build_crops(ctx)
        return ToolResult(
            ok=True,
            summary=f"{len(objects)} objects: " + ", ".join(list(objects)[:8]) + ("..." if len(objects) > 8 else ""),
            data={"objects": list(objects)},
        )

    def _analyze_all_objects_once(self, *, kind: str, sample: Optional[int] = None) -> list:
        messages = build_messages_analyze_all_objects(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=self._img("object_draft"),
        )
        return self.core.chat_json(
            self.ctx, step="object_draft", kind=kind, tool="detect_objects",
            messages=messages, temperature=0.3,
            sample=sample,
        )

    def _analyze_all_objects_once_text(self, sample: Optional[int] = None) -> str:
        messages = build_messages_analyze_all_objects(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=self._img("object_draft"),
        )
        response = self.core.chat(
            self.ctx, step="object_draft", tool="detect_objects",
            messages=messages, temperature=0.3,
            sample=sample,
        )
        return response.choices[0].message.content

    def _multi_analyze_all_objects(self, *, drafts_text: str, kind: str) -> list:
        messages = build_messages_multi_analyze_all_objects(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=self._img("object_merge"),
            multi_get_objects=drafts_text,
        )
        return self.core.chat_json(
            self.ctx, step="object_merge", kind=kind, tool="detect_objects",
            messages=messages, temperature=0.3,
        )

    def _multi_analyze_all_objects_text(self, drafts_text: str) -> str:
        messages = build_messages_multi_analyze_all_objects(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=self._img("object_merge"),
            multi_get_objects=drafts_text,
        )
        response = self.core.chat(
            self.ctx, step="object_merge", tool="detect_objects",
            messages=messages, temperature=0.3,
        )
        return response.choices[0].message.content

    # ---------------- 2. crop ----------------

    def _build_crops(self, ctx: dict) -> None:
        """Match merged objects to draft bboxes, verify crop reliability (crop
        accuracy check), precompute per-object crops and union relation crops."""
        from PIL import Image

        enc = self.cfg.image
        if self.cfg.output_mode == "json" and self.cfg.grounding:
            draft_boxes = match_draft_boxes(list(ctx["objects"]), ctx["drafts"])
        else:
            draft_boxes = {}
        ctx["bboxes"] = {
            name: [int(statistics.median(c[i] for c in boxes)) for i in range(4)]
            for name, boxes in draft_boxes.items()
        }
        try:
            with Image.open(ctx["image_path"]) as im:
                img_w, img_h = im.size
        except Exception:
            img_w = img_h = 0
        if self.cfg.grounding and self.cfg.output_mode == "json" and img_w and img_h:
            if self.cfg.crop_verify:
                decisions = assess_crops(
                    ctx["image_path"], img_w, img_h, list(ctx["objects"]), draft_boxes,
                    image_format=enc.image_format, jpeg_quality=enc.jpeg_quality,
                )
            else:
                from .crop_verify import CropDecision
                decisions = {
                    name: (
                        CropDecision(name=name, status="crop", bbox=box,
                                     area_frac=bbox_area_frac(box),
                                    crop_url=crop_bbox_data_url(
                                         ctx["image_path"], box, padding_frac=0.15, min_side=256,
                                         max_side=1024, image_format=enc.image_format,
                                         jpeg_quality=enc.jpeg_quality)[0],
                                     crop_info={},
                                     reason="crop")
                        if (box := ctx["bboxes"].get(name))
                        else CropDecision(name=name, status="full", reason="no_bbox")
                    )
                    for name in ctx["objects"]
                }
            ctx["crop_decisions"] = decisions
            for name, dec in decisions.items():
                if dec.status == "crop" and dec.crop_url:
                    ctx["object_crops"][name] = dec.crop_url
        # union crops: object + 2 nearest neighbors (relationship step2)
        bboxes = ctx["bboxes"]
        ctx["rel_crops"] = {}
        ctx["rel_crop_members"] = {}
        if img_w and img_h and len(bboxes) > 1:
            for name, box in bboxes.items():
                neighbor_names = sorted(
                    (other for other, other_box in bboxes.items() if other_box != box),
                    key=lambda other: bbox_distance(box, bboxes[other]),
                )[:2]
                union = union_bbox([box] + [bboxes[other] for other in neighbor_names])
                if bbox_area_frac(union) <= 0.75:
                    rel = crop_bbox_data_url(
                        ctx["image_path"], union, padding_frac=0.10, min_side=256, max_side=1024,
                        image_format=enc.image_format, jpeg_quality=enc.jpeg_quality, max_area_frac=0.95,
                    )
                    if rel is not None:
                        ctx["rel_crops"][name] = rel[0]
                        ctx["rel_crop_members"][name] = neighbor_names
        n_bbox = len(bboxes)
        n_obj = len(ctx["objects"])
        self.log.info("[%s] grounding: %d/%d objects matched to bboxes; %d object crops, %d relation crops.",
                      ctx["image_name"], n_bbox, n_obj, len(ctx["object_crops"]), len(ctx["rel_crops"]))

    def crop(self, object_name: str) -> ToolResult:
        dec = self.ctx["crop_decisions"].get(object_name)
        if dec is None:
            return ToolResult(ok=False, summary=f"no crop decision for {object_name!r}")
        extra = f" area={dec.area_frac:.1%}" if dec.bbox else ""
        iou = f" iou={dec.iou:.2f}" if dec.iou is not None else ""
        return ToolResult(ok=True, summary=f"{object_name}: {dec.status}{extra}{iou} ({dec.reason or 'ok'})")

    # ---------------- 3. inspect ----------------

    def inspect(self, object_name: str, depth: str = "full") -> ToolResult:
        ctx = self.ctx
        evidence = ctx["evidence"].setdefault(object_name, {})

        messages1 = build_messages_descriptor_step1(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=self._img("descriptor_step1"),
            object_name=object_name,
        )
        response1 = self.core.chat(
            ctx, step="descriptor_step1", tool="inspect", messages=messages1,
            temperature=0.3, object_name=object_name,
        )
        desc1 = response1.choices[0].message.content or ""
        evidence["desc1"] = desc1
        evidence["inspected_depth"] = depth

        if depth != "full":
            return ToolResult(ok=True, summary=f"{object_name}: described (lite)")

        messages2 = build_messages_descriptor_step2(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=self._img("descriptor_step2", object_name),
            object_name=object_name,
            descriptor_step1_response=desc1,
        )
        used_crop = bool(ctx["object_crops"].get(object_name))
        if self.cfg.output_mode == "json":
            cands = self.core.chat_json(
                ctx, step="descriptor_step2", kind="anomaly_candidates", tool="inspect",
                messages=messages2, temperature=0.3, object_name=object_name,
            )
            desc2 = json.dumps({"object": object_name, "anomalies": cands}, ensure_ascii=False)
            evidence["cands2"] = cands
        else:
            response2 = self.core.chat(
                ctx, step="descriptor_step2", tool="inspect", messages=messages2,
                temperature=0.3, object_name=object_name,
            )
            desc2 = response2.choices[0].message.content or ""
        evidence["desc2"] = desc2

        # crop accuracy post-check: description must plausibly refer to the object,
        # otherwise re-run the candidate step once with the full image (max 1/object)
        if (
            used_crop
            and self.cfg.crop_verify
            and not mentions_object(desc1 + " " + desc2, object_name)
            and not evidence.get("mention_rerun")
        ):
            self.log.warning("[%s] %s: descriptor from crop does not mention the object; re-running with full image.",
                             ctx["image_name"], object_name)
            evidence["mention_rerun"] = True
            messages2f = build_messages_descriptor_step2(
                system_prompt=self.system_prompt,
                expert_prompt=self.expert_prompt,
                image_data_url=self._img("global_reasoning"),
                object_name=object_name,
                descriptor_step1_response=desc1,
            )
            if self.cfg.output_mode == "json":
                cands = self.core.chat_json(
                    ctx, step="descriptor_step2_fullimage", kind="anomaly_candidates", tool="inspect",
                    messages=messages2f, temperature=0.3, object_name=object_name,
                )
                desc2 = json.dumps({"object": object_name, "anomalies": cands}, ensure_ascii=False)
                evidence["cands2"] = cands
            else:
                response2 = self.core.chat(
                    ctx, step="descriptor_step2_fullimage", tool="inspect", messages=messages2f,
                    temperature=0.3, object_name=object_name,
                )
                desc2 = response2.choices[0].message.content or ""
            evidence["desc2"] = desc2

        n_cand = len(evidence.get("cands2") or [])
        return ToolResult(ok=True, summary=f"{object_name}: described + {n_cand} anomaly candidates (full)")

    # ---------------- 4. check_relation ----------------

    def check_relation(self, object_name: str, members: Optional[list] = None) -> ToolResult:
        ctx = self.ctx
        if object_name not in ctx["objects"]:
            return ToolResult(ok=False, summary=f"unknown object {object_name!r}")
        all_objects = list(ctx["objects"])
        other_objects = [o for o in all_objects if o != object_name]
        if members:
            other_objects = [m for m in members if m in all_objects and m != object_name]
        elif self.cfg.grounding and object_name in ctx["rel_crops"]:
            visible = [m for m in ctx["rel_crop_members"].get(object_name, []) if m in all_objects]
            if visible:
                other_objects = visible
        all_other_objects_csv = ", ".join(other_objects)

        ev2_prev = ctx["evidence"].get(object_name, {}).get("desc2", "")
        messages1 = build_messages_relationship_step1(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=self._img("relationship_step1"),
            object_name=object_name,
            all_other_objects_csv=all_other_objects_csv,
            descriptor_step2_response=ev2_prev,
        )
        response1 = self.core.chat(
            ctx, step="relationship_step1", tool="check_relation", messages=messages1,
            temperature=0.3, object_name=object_name,
        )
        rel1 = response1.choices[0].message.content or ""

        messages2 = build_messages_relationship_step2(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=self._img("relationship_step2", object_name),
            object_name=object_name,
            all_other_objects_csv=all_other_objects_csv,
            relationship_step1_response=rel1,
        )
        if self.cfg.output_mode == "json":
            cands = self.core.chat_json(
                ctx, step="relationship_step2", kind="anomaly_candidates", tool="check_relation",
                messages=messages2, temperature=0.3, object_name=object_name,
            )
            rel2 = json.dumps({"object": object_name, "anomalies": cands}, ensure_ascii=False)
            evidence_rel = cands
        else:
            response2 = self.core.chat(
                ctx, step="relationship_step2", tool="check_relation", messages=messages2,
                temperature=0.3, object_name=object_name,
            )
            rel2 = response2.choices[0].message.content or ""
            evidence_rel = []

        # per-object summarizer step1 (V1: 5th per-object call)
        messages3 = build_messages_summarizer_step1(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=self._img("summarizer_step1", object_name),
            object_name=object_name,
            descriptor_step2_response=ev2_prev,
            relationship_step2_response=rel2,
        )
        if self.cfg.output_mode == "json":
            sum1_cands = self.core.chat_json(
                ctx, step="summarizer_step1", kind="anomaly_candidates", tool="check_relation",
                messages=messages3, temperature=0.3, object_name=object_name,
            )
            sum1 = json.dumps({"object": object_name, "anomalies": sum1_cands}, ensure_ascii=False)
        else:
            response3 = self.core.chat(
                ctx, step="summarizer_step1", tool="check_relation", messages=messages3,
                temperature=0.3, object_name=object_name,
            )
            sum1 = response3.choices[0].message.content or ""
        ctx["relationship"][object_name] = {"step1": rel1, "step2": rel2, "candidates": evidence_rel}
        ctx["summarizer_step1"][object_name] = sum1
        return ToolResult(ok=True, summary=f"{object_name}: relations vs {len(other_objects)} objects consolidated")

    # ---------------- 5. verify ----------------

    @staticmethod
    def _format_summarizer_step1(by_object: dict) -> str:
        parts = []
        for object_name, text in by_object.items():
            parts.append(f"List of the semantically unnatural expressions on {object_name}: {text}\n{'%' * 20}\n")
        return "".join(parts)

    @staticmethod
    def _merge_texts(texts, separator: str = "\n------\n") -> str:
        return separator.join([t for t in texts if isinstance(t, str) and t.strip()])

    def verify(self) -> ToolResult:
        cfg = self.cfg
        ctx = self.ctx
        # canonical object order (discovery order), NOT as_completed order: keeps the
        # step2 input deterministic across runs (cache-replayable, reproducible)
        step1_text = self._format_summarizer_step1(
            {n: ctx["summarizer_step1"][n] for n in ctx["objects"] if n in ctx["summarizer_step1"]}
        )
        if cfg.num_summarizer_step2 > 1:
            texts = _parallel(
                self.core,
                lambda i: self._summarize_step2_once(summarizer_everyone_step1=step1_text, sample=i),
                range(cfg.num_summarizer_step2),
            )
            texts = [texts[i] for i in range(cfg.num_summarizer_step2)]
        else:
            texts = [self._summarize_step2_once(summarizer_everyone_step1=step1_text)]
        ctx["verify_texts"] = texts
        return ToolResult(ok=True, summary=f"verify done: {len(texts)} samples")

    def _summarize_step2_once(self, *, summarizer_everyone_step1: str, sample: Optional[int] = None) -> str:
        messages = build_messages_summarizer_step2(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=self._img("summarizer_step2"),
            summarizer_everyone_step1=summarizer_everyone_step1,
        )
        if self.cfg.output_mode == "json":
            cands = self.core.chat_json(
                self.ctx, step="summarizer_step2", kind="anomaly_candidates", tool="verify",
                messages=messages, temperature=0.3,
                sample=sample,
            )
            return json.dumps({"anomalies": cands}, ensure_ascii=False)
        response = self.core.chat(
            self.ctx, step="summarizer_step2", tool="verify", messages=messages, temperature=0.3,
            sample=sample,
        )
        return response.choices[0].message.content or ""

    # ---------------- 6. global_scan ----------------

    def global_scan(self) -> ToolResult:
        cfg = self.cfg
        ctx = self.ctx
        if cfg.num_global_reasoning > 1:
            texts = _parallel(
                self.core,
                lambda i: self._step1_reasoning_once(sample=i),
                range(cfg.num_global_reasoning),
            )
            texts = [texts[i] for i in range(cfg.num_global_reasoning)]
        else:
            texts = [self._step1_reasoning_once()]
        ctx["global_scan_texts"] = texts
        if ctx["global_scan_items"] is None:
            items: list = []
            for text in texts:
                if self.cfg.output_mode == "json":
                    try:
                        items.extend(json.loads(text)["anomalies"])
                    except Exception:
                        pass
                else:
                    items.extend(parsing.parse_anomaly_list(text))
            ctx["global_scan_items"] = items
        return ToolResult(ok=True, summary=f"global scan done: {len(ctx['global_scan_items'])} anomalies in pool")

    def _step1_reasoning_once(self, sample: Optional[int] = None) -> str:
        messages = build_messages_step1_reasoning(image_data_url=self._img("global_reasoning"))
        if self.cfg.output_mode == "json":
            items = self.core.chat_json(
                self.ctx, step="global_reasoning", kind="final_anomaly_list", tool="global_scan",
                messages=messages, temperature=0.7,
                sample=sample,
            )
            return json.dumps({"anomalies": items}, ensure_ascii=False)
        response = self.core.chat(
            self.ctx, step="global_reasoning", tool="global_scan", messages=messages, temperature=0.7,
            sample=sample,
        )
        return response.choices[0].message.content or ""

    # ---------------- 7. finish ----------------

    def finish(self, preselected: Optional[list] = None) -> tuple[str, list]:
        """Final structured step3 (V1: up to 5 parse retries when the list comes back empty)."""
        ctx = self.ctx
        # V1 parity: pool = step2 texts + global reasoning texts (+ optional preselection)
        pool = list(ctx["verify_texts"])
        if ctx.get("global_scan_texts"):
            pool.extend(ctx["global_scan_texts"])
        if preselected:
            preselected_text = json.dumps({"anomalies": preselected}, ensure_ascii=False)
            pool.append(preselected_text)
        anomalies_text = self._merge_texts(pool)

        last_error: Optional[Exception] = None
        max_parse_retries = 5
        for attempt in range(1, max_parse_retries + 1):
            try:
                step3_text, items = self._summarize_step3_once(anomalies_text=anomalies_text)
                if not items:
                    raise ValueError("Empty parsed anomaly list")
                ctx["_final"] = {"anomalies": items, "step3_raw": step3_text}
                return step3_text, items
            except Exception as e:
                last_error = e
                self.log.warning("[%s] finish step3 parse failed (attempt %s/%s): %s",
                                 ctx["image_name"], attempt, max_parse_retries, e)
        raise RuntimeError("Failed to generate final anomalies") from last_error

    def _summarize_step3_once(self, *, anomalies_text: str) -> tuple[str, list]:
        messages = build_messages_summarizer_step3(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=self._img("summarizer_step3"),
            anomalies_text=anomalies_text,
        )
        if self.cfg.output_mode == "json":
            items = self.core.chat_json(
                self.ctx, step="summarizer_step3", kind="final_anomaly_list", tool="finish",
                messages=messages, temperature=0.3,
            )
            step3_text = json.dumps({"anomalies": items}, ensure_ascii=False)
            return step3_text, [
                {
                    "Name": it["name"],
                    "Observed Phenomenon": it["observed_phenomenon"],
                    "Reasoning": it["reasoning"],
                    "Severity Score": str(it["severity_score"]),
                }
                for it in items
            ]
        response = self.core.chat(
            self.ctx, step="summarizer_step3", tool="finish", messages=messages, temperature=0.3
        )
        step3_text = response.choices[0].message.content or ""
        return step3_text, parsing.parse_anomaly_list(step3_text)
