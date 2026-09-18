"""Controllers.

FixedScheduleController (default): deterministic paper pipeline - the 7 tools
executed in the exact T+1+5N+2S+1 call pattern (T drafts + 1 merge + N x
[inspect 2 + check_relation 3] + S verify + S global_scan + 1 finish;
11+5N with T=S=3). No controller LLM calls, no budgets, no relation-pair
dedup: the same schedule as V1, so results are reproducible while all
engineering gains apply.

DynamicController (opt-in, --schedule dynamic): the controller model emits one
strict-JSON tool action per loop iteration under a token budget (ported from
the AnomAgent_plusplus pilot; 7-tool set, no expert judge tool).
"""
from __future__ import annotations

import json
import os
import statistics
import time
from typing import Optional

from .. import schemas as step_schemas
from .actions import AGENT_ACTION_SCHEMA, ActionTracker, validate_agent_action
from .tools import AgentTools


class FixedScheduleController:
    def __init__(self, core) -> None:
        self.core = core
        self.log = core.logger

    def run_image(self, image_path: str) -> tuple[str, list, dict]:
        core = self.core
        ctx = core.begin_image(image_path)
        tools = AgentTools(core, ctx)
        image_name = ctx["image_name"]
        t0 = time.perf_counter()
        self.log.info("[%s] fixed schedule start (T=%s S2=%s Sg=%s, mode=%s, vision=%s, grounding=%s)",
                      image_name, core.config.num_get_objects, core.config.num_summarizer_step2,
                      core.config.num_global_reasoning, core.config.output_mode,
                      core.config.vision_mode, core.config.grounding)
        status = "ok"
        try:
            # 1. detect_objects: T drafts (parallel) + 1 merge
            tools.detect_objects()
            # 3. inspect: descriptor step1+step2 per object (parallel)
            from ..agent.tools import _parallel
            _parallel(core, lambda n: tools.inspect(n, "full"), list(ctx["objects"]))
            # 4. check_relation: relationship step1+step2 + per-object summarizer step1 (parallel)
            _parallel(core, lambda n: tools.check_relation(n), list(ctx["objects"]))
            # 5. verify: summarizer step2 x S (parallel inside)
            tools.verify()
            # 6. global_scan: global reasoning x S (parallel inside)
            tools.global_scan()
            # 7. finish: final structured step3
            step3_text, final_items = tools.finish()
        except Exception:
            status = "failed"
            raise
        finally:
            self._write_outputs(core, ctx, tools, status, time.perf_counter() - t0)
        meta = {
            "image_name": image_name,
            "schedule": "fixed",
            "status": status,
            "n_objects": len(ctx["objects"]),
            "n_crops": sum(1 for d in ctx["crop_decisions"].values() if d.status == "crop"),
            "n_mention_reruns": sum(1 for e in ctx["evidence"].values() if e.get("mention_rerun")),
            "n_anomalies": len(final_items) if status == "ok" else 0,
            "elapsed_seconds": round(time.perf_counter() - t0, 1),
        }
        self.log.info("[%s] fixed schedule done: %s in %.1fs (%d anomalies).",
                      image_name, status, meta["elapsed_seconds"], meta["n_anomalies"])
        return step3_text, final_items, meta

    def _write_outputs(self, core, ctx, tools: AgentTools, status: str, elapsed: float) -> None:
        usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for client in core.clients.values():
            u = client.usage.to_dict()
            for k in usage:
                usage[k] += u.get(k, 0)
        log_data = {
            "image_path": ctx["image_path"],
            "image_name": ctx["image_name"],
            "image_info": ctx["image_info"],
            "objects": ctx["objects"],
            "descriptor": {n: e.get("desc2", "") for n, e in ctx["evidence"].items()},
            "relationship": {n: r.get("step2", "") for n, r in ctx["relationship"].items()},
            "summarizer": {
                "step1": dict(ctx["summarizer_step1"]),
                "step3_raw": (ctx.get("_final") or {}).get("step3_raw", ""),
            },
            "final": {
                "anomalies": (ctx.get("_final") or {}).get("anomalies", []),
                "elapsed_minutes": round(elapsed / 60.0, 3),
                "status": status,
            },
            "usage": usage,
        }
        try:
            core.save_image_log(ctx["image_name"], log_data)
            core.write_image_ledger(ctx["image_name"], ctx["image_ledger"])
            core.save_viz(ctx["image_name"], ctx)
        except Exception as e:  # pragma: no cover
            core.logger.warning("[%s] failed to write outputs: %s", ctx["image_name"], e)


# ---------------------------------------------------------------- dynamic ---

CONTROLLER_SYSTEM_PROMPT = """\
You are the CONTROLLER of an anomaly-analysis agent for AI-generated images.
You decide, step by step, which tool to call next to find SEMANTIC anomalies
(physical/logical/common-sense violations, anatomical or structural defects, implausible object interactions).

Available tools:
- detect_objects: discover all salient objects in the image (with bounding boxes when available). Always call it first.
- crop(object): resolve whether a reliable cropped view of the object exists (informational; automatic after detection).
- inspect(object, depth): analyze one object. "full" = description + anomaly candidates (thorough, more tokens); "lite" = description only.
- check_relation(members): check the physical/logical relations between 2-3 named objects and consolidate per-object findings.
- verify: aggregate all per-object evidence into a classified anomaly list (one pooled pass; call once, near the end).
- global_scan: single-pass whole-image anomaly scan (safety net; call once before finish).
- finish: emit the final structured anomaly list. Optional pre-selection of the strongest candidates.

Policy:
1. Call detect_objects first.
2. Prefer thorough coverage: inspect every non-background object with depth "full" before verify. Background/sky/wall/ground items may be skipped.
3. Call check_relation on the most interaction-critical groups (persons and worn/held items, objects in contact or support, floating objects, scale-sensitive pairs).
4. Call verify once your evidence feels sufficient, then global_scan (if not done), then finish.
5. Watch the BUDGET: when little budget remains, stop inspecting and finish.
6. Never repeat a tool call you already made with the same arguments.
Emit exactly ONE action per turn as JSON. Set unused fields to "" / [] / "lite" defaults.
"""


def baseline_tokens_from_run(run_dir: str) -> Optional[int]:
    ledger = os.path.join(run_dir, "ledger.jsonl")
    if not os.path.exists(ledger):
        return None
    per_img: dict = {}
    try:
        with open(ledger, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                img = r.get("image") or "?"
                if img != "?":
                    per_img[img] = per_img.get(img, 0) + int(r.get("total_tokens") or 0)
    except Exception:
        return None
    vals = list(per_img.values())
    return int(statistics.median(vals)) if vals else None


class DynamicController:
    def __init__(
        self,
        core,
        *,
        budget_tokens: Optional[int] = None,
        budget_ref_run: Optional[str] = None,
    ) -> None:
        self.core = core
        cfg = core.config
        self.cfg = cfg
        self.log = core.logger
        if budget_tokens:
            self.baseline_tokens = int(budget_tokens)
        else:
            base = baseline_tokens_from_run(budget_ref_run) if budget_ref_run else None
            self.baseline_tokens = base if base else cfg.agent.fallback_baseline_tokens
        self.budget = int(self.baseline_tokens * cfg.agent.budget_factor)
        self.hard_stop = int(self.baseline_tokens * cfg.agent.hard_stop_factor)
        self.system_prompt = CONTROLLER_SYSTEM_PROMPT

    def _spent(self) -> int:
        total = 0
        for client in self.core.clients.values():
            total += client.usage.total_tokens
        return total

    def _render_state(self, tools: AgentTools) -> str:
        lines = []
        if tools.ctx["objects"]:
            lines.append(f"OBJECTS ({len(tools.ctx['objects'])}):")
            for i, (name, _) in enumerate(tools.ctx["objects"].items(), 1):
                d = tools.ctx["crop_decisions"].get(name)
                status = d.status if d else "?"
                area = f" area={d.area_frac:.1%}" if d and d.bbox else ""
                ev = tools.ctx["evidence"].get(name) or {}
                depth = ev.get("inspected_depth") or ""
                ncand = len(ev.get("cands2") or [])
                extra = f" [{depth} {ncand}cand]" if depth else ""
                lines.append(f" {i}. {name} | crop={status}{area}{extra}")
        else:
            lines.append("OBJECTS: none yet (call detect_objects)")
        gs = tools.ctx["global_scan_items"]
        lines.append(f"GLOBAL SCAN: done ({len(gs)} anomalies)" if gs is not None else "GLOBAL SCAN: pending")
        vdone = bool(tools.ctx["verify_texts"])
        lines.append("VERIFY: done" if vdone else "VERIFY: pending")
        spent = self._spent()
        lines.append(f"BUDGET: {spent} / {self.budget} tokens used ({spent / max(1, self.budget):.0%}); hard stop at {self.hard_stop}")
        lines.append("ACTIONS SO FAR: " + "; ".join(self.tracker.history[-12:]))
        if getattr(self, "_last_error", None):
            lines.append(f"LAST ACTION REJECTED: {self._last_error}")
        if getattr(self, "_budget_warning", False):
            lines.append("BUDGET EXCEEDED: only check_relation and finish are allowed now; finish soon.")
        return "\n".join(lines)

    def _controller_call(self, state: str) -> dict:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": state + "\n\nChoose the next action (JSON)."},
        ]
        response = self.core.chat(
            self._ctx,
            step="controller", tool="controller", messages=messages,
            max_tokens=self.cfg.agent.max_tokens, temperature=self.cfg.agent.controller_temperature,
            response_format=AGENT_ACTION_SCHEMA,
        )
        text = response.choices[0].message.content or ""
        return step_schemas.parse_json_text(text)

    def _dispatch(self, tools: AgentTools, action: dict) -> str:
        a = action["action"]
        if a == "detect_objects":
            res = tools.detect_objects()
        elif a == "crop":
            res = tools.crop(action["object"])
        elif a == "inspect":
            res = tools.inspect(action["object"], action.get("depth") or "full")
        elif a == "check_relation":
            res = tools.check_relation(action["members"][0], members=action["members"])
        elif a == "verify":
            res = tools.verify()
        elif a == "global_scan":
            res = tools.global_scan()
        elif a == "finish":
            res = None
        else:
            res = None
        if res is None:
            return "ok: finish"
        return ("ok: " if res.ok else "rejected: ") + res.summary

    def run_image(self, image_path: str) -> tuple[str, list, dict]:
        core = self.core
        ctx = core.begin_image(image_path)
        self._ctx = ctx
        tools = AgentTools(core, ctx)
        self.tracker = ActionTracker()
        image_name = ctx["image_name"]
        max_actions_cfg = self.cfg.agent.max_actions_per_image
        n_actions = 0
        consecutive_invalid = 0
        finish_reason = ""
        preselected = None
        t0 = time.perf_counter()
        self.log.info("[%s] dynamic schedule start (budget=%d, hard_stop=%d, baseline=%d)",
                      image_name, self.budget, self.hard_stop, self.baseline_tokens)

        while True:
            spent = self._spent()
            n_objects = len(ctx["objects"])
            cap = max_actions_cfg if max_actions_cfg else 3 * max(1, n_objects) + 8
            if spent > self.hard_stop:
                self.log.warning("[%s] hard budget stop (%d tokens); auto global_scan+finish", image_name, spent)
                finish_reason = "hard_budget_stop"
                break
            if n_actions >= cap:
                self.log.warning("[%s] max actions reached (%d); finishing", image_name, cap)
                finish_reason = "max_actions"
                break
            if ctx["global_scan_items"] is None and not ctx["verify_texts"] and spent > 0.85 * self.budget:
                finish_reason = "budget_preempt"
                break
            state = self._render_state(tools)
            try:
                action = validate_agent_action(self._controller_call(state))
                consecutive_invalid = 0
            except Exception as e:
                consecutive_invalid += 1
                self._last_error = str(e)[:200]
                self.log.warning("[%s] invalid controller action (%d in a row): %s",
                                 image_name, consecutive_invalid, self._last_error)
                n_actions += 1
                if consecutive_invalid >= 3:
                    finish_reason = "invalid_actions"
                    break
                continue
            self._last_error = None
            if spent > self.budget and action["action"] not in ("check_relation", "finish"):
                self.log.warning("[%s] over budget; action %s blocked", image_name, action["action"])
                self._budget_warning = True
                n_actions += 1
                if n_actions % 3 == 0:
                    finish_reason = "over_budget"
                    break
                continue
            else:
                self._budget_warning = False
            if self.tracker.is_duplicate(action):
                self.log.info("[%s] duplicate action %s rejected", image_name, action["action"])
                self.tracker.record(action, "duplicate-rejected")
                n_actions += 1
                continue
            if action["action"] == "finish":
                preselected = action.get("anomalies") or None
                self.tracker.record(action, "finish")
                n_actions += 1
                finish_reason = "finish"
                break
            try:
                outcome = self._dispatch(tools, action)
            except Exception as e:
                self.log.exception("[%s] tool %s crashed: %s", image_name, action["action"], e)
                outcome = f"error: {e}"
            self.tracker.record(action, outcome)
            n_actions += 1
            self.log.info("[%s] action %d: %s | %s", image_name, n_actions, action["action"], outcome[:160])

        # safety net + final step3
        if ctx["objects"]:
            if ctx["global_scan_items"] is None and finish_reason != "finish":
                try:
                    tools.global_scan()
                except Exception as e:
                    self.log.warning("[%s] auto global_scan failed: %s", image_name, e)
            try:
                step3_text, final_items = tools.finish(preselected)
            except Exception:
                pool_items = list(ctx["global_scan_items"] or [])
                final_items = [
                    {
                        "Name": it.get("name", ""),
                        "Observed Phenomenon": it.get("observed_phenomenon", ""),
                        "Reasoning": it.get("reasoning", ""),
                        "Severity Score": str(it.get("severity_score", 50)),
                    }
                    for it in pool_items
                ]
                step3_text = json.dumps({"anomalies": final_items}, ensure_ascii=False)
                finish_reason += "/step3-fallback"
        else:
            raise RuntimeError("object detection failed (detect_failed)")

        fixed = FixedScheduleController(core)
        fixed._write_outputs(core, ctx, tools, "ok" if final_items is not None else "failed",
                             time.perf_counter() - t0)
        meta = {
            "image_name": image_name,
            "schedule": "dynamic",
            "finish_reason": finish_reason,
            "n_actions": n_actions,
            "n_objects": len(ctx["objects"]),
            "n_anomalies": len(final_items),
            "tokens_spent": self._spent(),
            "budget": self.budget,
            "elapsed_seconds": round(time.perf_counter() - t0, 1),
            "history": self.tracker.history,
        }
        self.log.info("[%s] dynamic schedule done: %s (tokens=%d/%d, actions=%d, %.1fs, %d anomalies)",
                      image_name, finish_reason, meta["tokens_spent"], self.budget,
                      n_actions, meta["elapsed_seconds"], meta["n_anomalies"])
        return step3_text, final_items, meta
