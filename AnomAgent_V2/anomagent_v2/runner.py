"""Multi-image runner: inter-image parallelism, multi-instance round-robin,
resume (--continue), batch-API fan-out hook, aggregated outputs (V1 format)."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

import tqdm

from .config import AnalyzerConfig
from .core import P0Core
from .image_utils import collect_image_paths
from .agent.controller import DynamicController, FixedScheduleController


class ImageRunner:
    def __init__(self, save_root_dir: str, config: AnalyzerConfig, *, step_cache_dir: Optional[str] = None) -> None:
        self.save_root_dir = save_root_dir
        self.config = config
        self.step_cache_dir = step_cache_dir or (
            os.path.join(save_root_dir, "step_cache") if config.step_cache else None
        )
        os.makedirs(os.path.join(save_root_dir, "every_steps"), exist_ok=True)
        self._ledger_lock = threading.Lock()
        self.responses_text: list = []
        self.responses_structured: list = []
        self.failed_files: list = []
        self._results_lock = threading.Lock()

    def _controller_for(self, core: P0Core, budget_ref_run: Optional[str]) -> object:
        if self.config.agent.schedule == "dynamic":
            return DynamicController(core, budget_ref_run=budget_ref_run)
        return FixedScheduleController(core)

    def _worker_analyze(self, core: P0Core, image_path: str, budget_ref_run: Optional[str]) -> None:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        controller = self._controller_for(core, budget_ref_run)
        try:
            step3_text, items, meta = controller.run_image(image_path)
            with self._results_lock:
                self.responses_text.append({"path": image_path, "response": step3_text})
                self.responses_structured.append({image_path: items})
            core.logger.info("Done %s (schedule=%s).", image_name, meta.get("schedule"))
        except Exception as e:
            core.logger.exception("Failed to analyze %s: %s", image_path, e)
            with self._results_lock:
                self.failed_files.append({"path": image_path, "error": str(e)[:500]})
                self.responses_text.append({"path": image_path, "response": f"ERROR: {e}"})
                self.responses_structured.append({image_path: []})

    def run(
        self,
        inputs,
        *,
        start: str = "",
        end: str = "",
        is_continue: bool = False,
        budget_ref_run: Optional[str] = None,
    ) -> None:
        image_paths = collect_image_paths(inputs)
        if start:
            image_paths = [p for p in image_paths if start in os.path.basename(p)]
        if end:
            image_paths = [p for p in image_paths if end in os.path.basename(p)]
        if not image_paths:
            raise ValueError(f"No images found under inputs: {inputs}")

        if is_continue:
            keep = []
            for p in image_paths:
                name = os.path.splitext(os.path.basename(p))[0]
                existing, raw = self._load_final(name)
                if existing:
                    with self._results_lock:
                        self.responses_text.append({"path": p, "response": raw})
                        self.responses_structured.append({p: existing})
                    continue
                keep.append(p)
            image_paths = keep
            if not image_paths:
                self._finalize()
                return

        total = len(image_paths)
        concurrency = self.config.image_concurrency
        base_urls = list(self.config.base_urls or [])
        if concurrency == 1:
            core = P0Core(
                self.save_root_dir, config=self.config,
                base_url=base_urls[0] if base_urls else None,
                step_cache_dir=self.step_cache_dir,
            )
            for path in tqdm.tqdm(image_paths, desc="images", unit="img"):
                self._worker_analyze(core, path, budget_ref_run)
        else:
            self._run_parallel(image_paths, concurrency, base_urls, budget_ref_run, total)
        self._finalize()

    def _load_final(self, image_name: str) -> tuple[Optional[list], str]:
        path = os.path.join(self.save_root_dir, "every_steps", f"{image_name}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            final = (data.get("final") or {}).get("anomalies")
            if final:
                return final, (data.get("summarizer") or {}).get("step3_raw", "")
        except Exception:
            pass
        return None, ""

    def _run_parallel(self, image_paths, concurrency: int, base_urls: list, budget_ref_run, total: int) -> None:
        from collections import deque

        queue: deque = deque(image_paths)
        q_lock = threading.Lock()
        pbar = tqdm.tqdm(total=total, desc=f"images x{concurrency}", unit="img")

        def worker(wid: int) -> None:
            base_url = base_urls[wid % len(base_urls)] if base_urls else None
            core = P0Core(
                self.save_root_dir, config=self.config, worker_id=wid,
                base_url=base_url, step_cache_dir=self.step_cache_dir,
            )
            while True:
                with q_lock:
                    if not queue:
                        return
                    path = queue.popleft()
                try:
                    self._worker_analyze(core, path, budget_ref_run)
                finally:
                    pbar.update(1)

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        pbar.close()

    def _finalize(self) -> None:
        with self._results_lock:
            responses_text = list(self.responses_text)
            responses_structured = list(self.responses_structured)
            failed_files = list(self.failed_files)
        self._write_json(os.path.join(self.save_root_dir, "responses.json"), responses_text)
        self._write_json(os.path.join(self.save_root_dir, "responses_structured.json"), responses_structured)
        self._write_json(os.path.join(self.save_root_dir, "failed_files.json"), failed_files)
        self._write_run_summary()

    @staticmethod
    def _write_json(path: str, data) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    def _write_run_summary(self) -> None:
        ledger_path = os.path.join(self.save_root_dir, "ledger.jsonl")
        per_image: dict = {}
        calls: dict = {}
        cache_hits = 0
        truncated = 0
        if os.path.exists(ledger_path):
            with open(ledger_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    img = r.get("image") or "?"
                    if r.get("status") == "ok":
                        per_image[img] = per_image.get(img, 0) + int(r.get("total_tokens") or 0)
                    elif r.get("status") == "cache_hit":
                        cache_hits += 1
                    if r.get("finish_reason") == "length":
                        truncated += 1
                    calls[img] = calls.get(img, 0) + 1
        summary = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_images": len(per_image),
            "tokens_per_image": dict(sorted(per_image.items())),
            "calls_per_image": dict(sorted(calls.items())),
            "step_cache_hits": cache_hits,
            "truncated_length_retries": truncated,
            "failed": [f.get("path") for f in self.failed_files],
        }
        self._write_json(os.path.join(self.save_root_dir, "run_summary.json"), summary)
