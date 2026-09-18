"""P0Core: per-run state, model routing, token-elastic chat, ledgers, viz recording.

One P0Core per worker (image-parallel workers each own a core; they share the
run dir, the step cache, and the run-level ledger file).
"""
from __future__ import annotations

import base64
import copy
import json
import os
import re
import threading
import time
from typing import Any, Optional

from .config import AnalyzerConfig, OpenAIConfig
from .image_utils import encode_image_to_data_url, encode_image_variants
from .openai_client import OpenAIChatClient
from .schemas import STEP_SCHEMAS, json_tail, parse_json_text
from .step_cache import StepCache
from .logging_utils import build_logger

# step -> default max_tokens (json mode); text mode uses V1 values (2000 / step3 5000)
_JSON_STEP_MAX_TOKENS = {
    "object_draft": "discovery",
    "object_merge": "discovery",
    "descriptor_step1": "base",
    "descriptor_step2": "base",
    "relationship_step1": "base",
    "relationship_step2": "base",
    "summarizer_step1": "base",
    "summarizer_step2": "step2",
    "global_reasoning": "reasoning",
    "summarizer_step3": "step3",
}

_VISION_TIERS = {
    "object_draft": "global",
    "object_merge": "global",
    "global_reasoning": "global",
    "summarizer_step3": "global",
    "summarizer_step2": "medium",
    "descriptor_step1": "medium",
    "relationship_step1": "medium",
    "descriptor_step2": "local",
    "relationship_step2": "local",
    "summarizer_step1": "local",
}


def _data_url_to_bytes(data_url: str) -> Optional[bytes]:
    m = re.match(r"^data:(?P<mime>[\w/+.-]+);base64,(?P<payload>.*)$", data_url or "", re.S)
    if not m:
        return None
    try:
        return base64.b64decode(m.group("payload"))
    except Exception:
        return None


class VizRecorder:
    """Collects per-step input/output (incl. image asset references) for one image."""

    def __init__(self, assets_dir: str) -> None:
        self.assets_dir = assets_dir
        self.entries: list = []
        self._seq = 0
        os.makedirs(assets_dir, exist_ok=True)

    def _save_image_asset(self, step: str, label: str, data_url: str) -> Optional[str]:
        raw = _data_url_to_bytes(data_url)
        if not raw:
            return None
        self._seq += 1
        ext = "jpg" if (data_url.startswith("data:image/jpeg") or data_url.startswith("data:image/jpg")) else "png"
        rel = f"{self._seq:03d}_{label.replace(' ', '_')}.{ext}"
        with open(os.path.join(self.assets_dir, rel), "wb") as f:
            f.write(raw)
        return rel

    def record(
        self,
        *,
        step: str,
        tool: str,
        client: str,
        model: str,
        object_name: Optional[str],
        messages: list,
        input_images: list,
        response: Any,
        status: str,
        usage: dict,
        finish_reason: Optional[str],
        latency_ms: float,
        attempts: int,
        max_tokens: int,
        temperature: float,
        parsed: Any = None,
        error: Optional[str] = None,
    ) -> None:
        system_text, user_text = "", ""
        for msg in messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                system_text = msg["content"]
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    user_text = content
                elif isinstance(content, list):
                    user_text = "\n".join(
                        p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                    )
        raw_content = None
        try:
            raw_content = response.choices[0].message.content if response is not None else None
        except Exception:
            raw_content = None
        entry = {
            "step": step,
            "tool": tool,
            "client": client,
            "model": model,
            "object": object_name,
            "status": status,
            "attempts": attempts,
            "latency_ms": round(latency_ms, 1),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "finish_reason": finish_reason,
            "usage": usage,
            "input": {
                "system": system_text[:8000],
                "user_text": user_text,
                "images": input_images,
            },
            "output": {
                "raw": raw_content,
                "parsed": parsed,
            },
            "error": error,
            "ts": time.time(),
        }
        self.entries.append(entry)


class P0Core:
    def __init__(
        self,
        save_root_dir: str,
        *,
        config: AnalyzerConfig,
        worker_id: int = 0,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        step_cache_dir: Optional[str] = None,
    ) -> None:
        self.config = config
        self.save_root_dir = save_root_dir
        self.every_steps_dir = os.path.join(save_root_dir, "every_steps")
        self.assets_root = os.path.join(self.every_steps_dir, "assets")
        os.makedirs(self.every_steps_dir, exist_ok=True)

        self.logger, self.log_file_path = build_logger(
            name=f"anomagent_v2.core.{worker_id}",
            log_level=config.log_level,
            log_to_file=config.log_to_file,
            log_dir=save_root_dir,
        )

        self._ledger_lock = threading.Lock()
        self._run_ledger_path = os.path.join(save_root_dir, "ledger.jsonl")
        self.step_cache = StepCache(step_cache_dir or os.path.join(save_root_dir, "step_cache"),
                                    enabled=config.step_cache)

        self.clients: dict = {}
        main_cfg = OpenAIConfig(
            api_key=config.openai.api_key,
            base_url=base_url or config.openai.base_url,
            model=model or config.openai.model,
        )
        if not main_cfg.api_key or not main_cfg.model:
            raise ValueError("Missing model/api_key config for the main client.")
        self.clients["main"] = self._build_client(main_cfg)
        if config.expert.enabled:
            expert_cfg = OpenAIConfig(
                api_key=config.expert.api_key or config.openai.api_key,
                base_url=config.expert.base_url,
                model=config.expert.model,
            )
            self.clients["expert"] = self._build_client(expert_cfg)
            self.logger.info("model routing enabled: expert=%s @ %s; routes=%s",
                             config.expert.model, config.expert.base_url, config.route_steps)

        self.current_image_name = ""

    def _build_client(self, openai_config: OpenAIConfig) -> OpenAIChatClient:
        return OpenAIChatClient(
            openai_config=openai_config,
            retry_config=self.config.retry,
            logger=self.logger,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            if self.config.no_thinking
            else None,
            on_call=self._on_call,
            step_cache=self.step_cache,
        )

    # ---------- ledger ----------

    def _append_run_ledger(self, record: dict) -> None:
        record = dict(record)
        record["image"] = self.current_image_name
        with self._ledger_lock:
            try:
                with open(self._run_ledger_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                pass

    def _on_call(self, meta: dict) -> None:
        self._append_run_ledger(meta)

    def write_image_ledger(self, image_name: str, records: list) -> None:
        path = os.path.join(self.every_steps_dir, f"{image_name}.ledger.jsonl")
        try:
            with open(path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:  # pragma: no cover
            self.logger.warning("Failed to write image ledger for %s: %s", image_name, e)

    # ---------- per-image context ----------

    def begin_image(self, image_path: str) -> dict:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        enc = self.config.image
        if self.config.vision_mode == "tiered":
            variants = encode_image_variants(
                image_path,
                max_side=enc.max_side,
                image_format=enc.image_format,
                jpeg_quality=enc.jpeg_quality,
            )
            urls = {"global": variants["global"], "medium": variants["medium"], "local": variants["local"]}
            image_info = json.loads(variants["info"])
        else:
            url, image_info = encode_image_to_data_url(
                image_path, max_side=enc.max_side, image_format=enc.image_format, jpeg_quality=enc.jpeg_quality
            )
            urls = {"global": url, "medium": url, "local": url}
        for client in self.clients.values():
            client.reset_usage()
        self.current_image_name = image_name
        ctx = {
            "image_path": image_path,
            "image_name": image_name,
            "image_info": image_info,
            "image_urls": urls,
            "image_url": urls["global"],
            "objects": {},
            "drafts": [],
            "bboxes": {},
            "crop_decisions": {},
            "object_crops": {},
            "rel_crops": {},
            "rel_crop_members": {},
            "evidence": {},
            "relationship": {},
            "summarizer_step1": {},
            "verify_texts": [],
            "global_scan_items": None,
            "image_ledger": [],
            "viz": VizRecorder(os.path.join(self.assets_root, image_name)),
            "start_time": time.perf_counter(),
        }
        self.logger.info("[%s] Start (encoded_size=%s, encoded_bytes=%s).",
                         image_name, image_info.get("encoded_size"), image_info.get("encoded_bytes"))
        return ctx

    # ---------- image selection (grounding/tiers) ----------

    def image_for_step(self, ctx: dict, step: str, object_name: Optional[str] = None) -> str:
        if self.config.grounding and object_name is not None:
            if step in ("descriptor_step2", "summarizer_step1"):
                crop = ctx["object_crops"].get(object_name)
                if crop:
                    return crop
            elif step == "relationship_step2":
                crop = ctx["rel_crops"].get(object_name)
                if crop:
                    return crop
        tier = _VISION_TIERS.get(step, "global")
        return ctx["image_urls"].get(tier, ctx["image_urls"]["global"])

    def describe_image(self, ctx: dict, data_url: str, label: str) -> dict:
        """Classify which variant a data URL is (for viz bookkeeping)."""
        for name, url in ctx["image_urls"].items():
            if url == data_url:
                return {"label": label or name, "kind": name, "file": None}
        return {"label": label or "crop", "kind": "crop", "file": None}

    # ---------- chat with elasticity / routing / ledger / viz ----------

    def step_max_tokens(self, step: str) -> int:
        key = _JSON_STEP_MAX_TOKENS.get(step)
        if self.config.output_mode == "json":
            if key == "discovery":
                return self.config.agent.discovery_grounding_max_tokens if self.config.grounding \
                    else self.config.agent.discovery_max_tokens
            if key == "step2":
                return self.config.agent.step2_max_tokens
            if key == "reasoning":
                return self.config.agent.reasoning_max_tokens
            if key == "step3":
                return self.config.agent.step3_max_tokens
            return self.config.agent.max_tokens
        return 5000 if step == "summarizer_step3" else 2000

    def chat(
        self,
        ctx: dict,
        *,
        step: str,
        tool: str = "direct",
        messages: list,
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
        top_p: float = 0.95,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        stop=None,
        response_format: Optional[dict] = None,
        client_name: Optional[str] = None,
        object_name: Optional[str] = None,
        sample: Optional[int] = None,
    ):
        client_name = client_name or self.config.route_for(step)
        client = self.clients.get(client_name)
        if client is None:
            raise ValueError(f"Unknown client {client_name!r} for step {step!r}")
        base_max = max_tokens if max_tokens is not None else self.step_max_tokens(step)
        agent_cfg = self.config.agent
        cur_messages = messages
        cur_max = base_max
        attempts = 0
        truncated_retries = 0
        while True:
            t0 = time.perf_counter()
            try:
                response = client.create_chat_completion(
                    step=step,
                    messages=cur_messages,
                    max_tokens=cur_max,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    response_format=response_format,
                    sample=sample,
                )
            except Exception as e:
                self._record_chat(ctx, step=step, tool=tool, client_name=client_name,
                                  model=client.model, object_name=object_name, messages=cur_messages,
                                  response=None, status="error", latency_ms=(time.perf_counter() - t0) * 1000.0,
                                  attempts=attempts + 1, max_tokens=cur_max, temperature=temperature,
                                  error=str(e)[:300])
                raise
            attempts += 1
            finish_reason = None
            try:
                finish_reason = response.choices[0].finish_reason
            except Exception:
                pass
            usage = {}
            try:
                usage = (response.to_dict() or {}).get("usage") or {}
            except Exception:
                try:
                    usage = response.usage.to_dict() or {}
                except Exception:
                    usage = {}
            is_cache = bool(getattr(response, "from_cache", False))
            self._record_chat(ctx, step=step, tool=tool, client_name=client_name,
                              model=client.model, object_name=object_name, messages=cur_messages,
                              response=response, status="cache_hit" if is_cache else "ok",
                              latency_ms=(time.perf_counter() - t0) * 1000.0,
                              attempts=attempts, max_tokens=cur_max, temperature=temperature,
                              usage=usage, finish_reason=finish_reason)
            if finish_reason == "length" and cur_max * 2 <= agent_cfg.max_tokens_cap:
                new_max = cur_max * 2
                truncated_retries += 1
                self.logger.warning("[%s] step=%s truncated at max_tokens=%s; elastic retry with %s.",
                                    ctx["image_name"], step, cur_max, new_max)
                cur_max = new_max
                continue
            return response

    def _record_chat(
        self,
        ctx: dict,
        *,
        step: str,
        tool: str,
        client_name: str,
        model: str,
        object_name: Optional[str],
        messages: list,
        response: Any,
        status: str,
        latency_ms: float,
        attempts: int,
        max_tokens: int,
        temperature: float,
        usage: Optional[dict] = None,
        finish_reason: Optional[str] = None,
        parsed: Any = None,
        error: Optional[str] = None,
    ) -> None:
        usage = usage or {}
        record = {
            "step": step,
            "object": object_name,
            "tool": tool,
            "client": client_name,
            "model": model,
            "attempt": attempts,
            "status": status,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "finish_reason": finish_reason,
            "latency_ms": round(latency_ms, 1),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "error": error,
        }
        ctx["image_ledger"].append(record)
        try:
            viz_images = []
            for msg in messages:
                content = msg.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "image_url":
                            url = (part.get("image_url") or {}).get("url", "")
                            label = part.get("label", "")
                            desc = self.describe_image(ctx, url, label)
                            if desc["file"] is None and desc["kind"] != "global":
                                rel = ctx["viz"]._save_image_asset(step, desc["label"], url)
                                desc["file"] = rel
                            viz_images.append(desc)
            ctx["viz"].record(
                step=step, tool=tool, client=client_name, model=model, object_name=object_name,
                messages=messages, input_images=viz_images, response=response, status=status,
                usage=usage, finish_reason=finish_reason, latency_ms=latency_ms, attempts=attempts,
                max_tokens=max_tokens, temperature=temperature, parsed=parsed, error=error,
            )
        except Exception as e:  # pragma: no cover - viz must never break the run
            self.logger.debug("viz record failed: %s", e)

    def chat_json(
        self,
        ctx: dict,
        *,
        step: str,
        kind: str,
        tool: str = "direct",
        messages: list,
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
        object_name: Optional[str] = None,
        sample: Optional[int] = None,
    ) -> list:
        """JSON-schema step: append strict schema tail, validate, one feedback retry on failure."""
        schema, validator = STEP_SCHEMAS[kind]
        msgs = self._with_json_tail(messages, kind)
        last_error: Optional[Exception] = None
        cur_msgs = msgs
        for attempt in range(1, 3):
            if attempt == 2:
                cur_msgs = copy.deepcopy(msgs)
                feedback = (
                    f"\n\nYour previous response was invalid: {last_error}. "
                    "Respond again with ONLY the valid JSON object."
                )
                content = cur_msgs[-1].get("content")
                if isinstance(content, list):
                    content.append({"type": "text", "text": feedback})
                elif isinstance(content, str):
                    cur_msgs[-1]["content"] = content + feedback
            response = self.chat(
                ctx, step=step, tool=tool,
                messages=cur_msgs, max_tokens=max_tokens, temperature=temperature,
                top_p=0.95, frequency_penalty=0, presence_penalty=0, stop=None,
                response_format=schema, object_name=object_name,
                sample=sample,
            )
            text = response.choices[0].message.content
            try:
                items = validator(parse_json_text(text))
                self._annotate_last_viz(ctx, parsed=items)
                return items
            except Exception as e:
                last_error = e
                self.logger.warning("[%s] %s json parse/validate failed (attempt %s/2): %s",
                                    ctx["image_name"], step, attempt, e)
        raise RuntimeError(f"Failed to parse/validate JSON for step {step}") from last_error

    def _annotate_last_viz(self, ctx: dict, parsed: Any) -> None:
        try:
            if ctx["viz"].entries:
                ctx["viz"].entries[-1]["output"]["parsed"] = parsed
        except Exception:
            pass

    @staticmethod
    def _with_json_tail(messages: list, kind: str) -> list:
        msgs = copy.deepcopy(messages)
        last = msgs[-1]
        content = last.get("content")
        if isinstance(content, list):
            for part in reversed(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] += json_tail(kind)
                    break
        elif isinstance(content, str):
            last["content"] = content + json_tail(kind)
        return msgs

    # ---------- run outputs (V1-compatible) ----------

    def save_image_log(self, image_name: str, log_data: dict) -> None:
        path = os.path.join(self.every_steps_dir, f"{image_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

    def load_image_log(self, image_name: str) -> Optional[dict]:
        path = os.path.join(self.every_steps_dir, f"{image_name}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def save_viz(self, image_name: str, ctx: dict) -> None:
        path = os.path.join(self.every_steps_dir, f"{image_name}.viz.json")
        payload = {
            "image_name": image_name,
            "image_path": ctx["image_path"],
            "image_info": ctx["image_info"],
            "assets_dir": f"assets/{image_name}",
            "crop_decisions": {
                name: {
                    "status": getattr(dec, "status", None),
                    "bbox": getattr(dec, "bbox", None),
                    "area_frac": getattr(dec, "area_frac", None),
                    "iou": getattr(dec, "iou", None),
                    "reason": getattr(dec, "reason", None),
                }
                for name, dec in ctx.get("crop_decisions", {}).items()
            },
            "final": ctx.get("_final", {}),
            "entries": ctx["viz"].entries,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1)
        except Exception as e:  # pragma: no cover
            self.logger.warning("Failed to write viz json for %s: %s", image_name, e)
