from __future__ import annotations

import concurrent.futures
import glob
import json
import os
import sys
import time
from collections.abc import Iterable
from typing import Optional, Union

from tqdm import tqdm

from .config import AnalyzerConfig, OpenAIConfig
from .image_utils import encode_image_to_data_url
from .logging_utils import build_logger
from .openai_client import OpenAIChatClient
from .parsing import parse_anomaly_list, parse_object_descriptions
from .prompts import (
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


class AIImageAnalyzer:
    def __init__(
        self,
        save_root_dir: str,
        *,
        config: Optional[AnalyzerConfig] = None,
        is_continue: bool = False,
        save_root_dir_suffix: str = "",
        start: str = "",
        end: str = "",
    ) -> None:
        self.config = config or AnalyzerConfig()
        self.is_continue = bool(is_continue)

        self.save_root_dir = self._resolve_run_dir(
            save_root_dir=save_root_dir,
            is_continue=self.is_continue,
            suffix=save_root_dir_suffix,
            start=start,
            end=end,
        )
        self.every_steps_dir = os.path.join(self.save_root_dir, "every_steps")
        os.makedirs(self.every_steps_dir, exist_ok=True)

        self.logger, self.log_file_path = build_logger(
            name=f"anomagent.analyzer.{id(self)}",
            log_level=self.config.log_level,
            log_to_file=self.config.log_to_file,
            log_dir=self.save_root_dir,
        )

        self.openai_config = self._resolve_openai_config(self.config.openai)
        self.chat = OpenAIChatClient(
            openai_config=self.openai_config,
            retry_config=self.config.retry,
            logger=self.logger,
        )

        self.system_prompt = SYSTEM_PROMPT
        self.expert_prompt = ""  # keep for later extensions

        self._write_run_metadata()

    def _resolve_openai_config(self, cfg: OpenAIConfig) -> OpenAIConfig:
        api_key = cfg.api_key or os.getenv("OPENAI_API_KEY")
        base_url = cfg.base_url or os.getenv("OPENAI_BASE_URL")
        model = cfg.model or os.getenv("OPENAI_MODEL") or "gpt-4o"
        return OpenAIConfig(api_key=api_key, base_url=base_url, model=model)

    def _write_run_metadata(self) -> None:
        metadata = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "config": self.config.to_dict(),
            "openai": {
                "base_url": self.openai_config.base_url,
                "model": self.openai_config.model,
            },
        }
        path = os.path.join(self.save_root_dir, "run_metadata.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning("Failed to write run metadata: %s", e)

    def _resolve_run_dir(
        self, *, save_root_dir: str, is_continue: bool, suffix: str, start: str, end: str
    ) -> str:
        if is_continue:
            return self.get_latest_run_dir(save_root_dir)

        run_dir = f"{save_root_dir}_{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())}_{suffix}"
        if start and end:
            run_dir = f"{run_dir}start_{start}_end_{end}"
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def get_latest_run_dir(self, save_root_dir: str) -> str:
        prefix = os.path.basename(save_root_dir)
        parent = os.path.dirname(save_root_dir) or "."
        if not os.path.isdir(parent):
            run_dir = f"{save_root_dir}_{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())}"
            os.makedirs(run_dir, exist_ok=True)
            return run_dir

        pattern = rf"{prefix}_(\d{{4}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}})"
        candidates = []
        for name in os.listdir(parent):
            full_path = os.path.join(parent, name)
            if not os.path.isdir(full_path):
                continue
            if not name.startswith(prefix):
                continue
            match = None
            try:
                import re

                match = re.search(pattern, name)
            except Exception:
                match = None
            if match:
                candidates.append((name, match.group(1)))

        if not candidates:
            run_dir = f"{save_root_dir}_{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())}"
            os.makedirs(run_dir, exist_ok=True)
            return run_dir

        import datetime

        latest = max(
            candidates,
            key=lambda x: datetime.datetime.strptime(x[1], "%Y_%m_%d_%H_%M_%S"),
        )[0]
        return os.path.join(parent, latest)

    def _save_image_log(self, image_name: str, log_data: dict) -> None:
        path = os.path.join(self.every_steps_dir, f"{image_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

    def _load_image_log(self, image_name: str) -> Optional[dict]:
        path = os.path.join(self.every_steps_dir, f"{image_name}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _collect_images_from_dir(self, data_dir: str) -> list[str]:
        patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp", "**/*.gif", "**/*.tiff", "**/*.webp"]
        image_paths: list[str] = []
        for ext in patterns:
            image_paths.extend(glob.glob(os.path.join(data_dir, ext), recursive=True))
        return [os.path.abspath(p) for p in sorted(set(image_paths))]

    def analyze_images(self, inputs: Union[str, list[str]]) -> None:
        if isinstance(inputs, list):
            image_paths = [os.path.abspath(p) for p in inputs]
        else:
            image_paths = self._collect_images_from_dir(inputs)

        failed_files: list[dict] = []
        responses_text: list[dict] = []
        responses_structured: list[dict] = []

        for image_path in tqdm(image_paths, file=sys.stdout):
            try:
                text, parsed = self.analyze_image(image_path)
                if text == "skip" or parsed == "skip":
                    continue
                responses_text.append({image_path: text})
                responses_structured.append({image_path: parsed})
            except Exception as e:
                failed_files.append({image_path: str(e)})
                self.logger.exception("Failed to analyze %s: %s", image_path, e)

        self._write_json(os.path.join(self.save_root_dir, "responses.json"), responses_text)
        self._write_json(
            os.path.join(self.save_root_dir, "responses_structured.json"), responses_structured
        )
        self._write_json(os.path.join(self.save_root_dir, "failed_files.json"), failed_files)

    def _write_json(self, path: str, data) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def analyze_image(self, image_path: str) -> tuple[str, Union[list[dict], str]]:
        image_path = os.path.abspath(image_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        existing = self._load_image_log(image_name)
        if self.is_continue and existing and existing.get("final", {}).get("anomalies"):
            return "skip", "skip"

        image_data_url, image_info = encode_image_to_data_url(
            image_path,
            max_side=self.config.image.max_side,
            image_format=self.config.image.image_format,
            jpeg_quality=self.config.image.jpeg_quality,
        )

        self.chat.reset_usage()

        encoded_size = image_info.get("encoded_size")
        encoded_bytes = image_info.get("encoded_bytes")
        self.logger.info(
            "[%s] Start (encoded_size=%s, encoded_bytes=%s).",
            image_name,
            encoded_size,
            encoded_bytes,
        )

        log_data = {
            "image_path": image_path,
            "image_name": image_name,
            "image_info": image_info,
            "objects": {},
            "descriptor": {},
            "relationship": {},
            "summarizer": {},
            "final": {},
            "usage": {},
        }

        start_time = time.perf_counter()

        self.logger.info(
            "[%s] Discovering objects (num_get_objects=%s)...",
            image_name,
            self.config.num_get_objects,
        )
        objects = self._get_objects(image_data_url=image_data_url)
        self.logger.info("[%s] Discovered %s objects.", image_name, len(objects))
        if self.config.max_objects is not None:
            objects = {k: objects[k] for k in list(objects.keys())[: self.config.max_objects]}
            self.logger.info("[%s] Limiting to max_objects=%s.", image_name, self.config.max_objects)
        log_data["objects"] = objects
        self._save_image_log(image_name, log_data)

        descriptor_step2_by_object: dict[str, str] = {}
        relationship_step2_by_object: dict[str, str] = {}

        self.logger.info("[%s] Descriptor step (objects=%s)...", image_name, len(objects))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._analyze_descriptor, obj, image_data_url): obj for obj in objects
            }
            total = len(futures)
            done = 0
            last_log_at = time.monotonic()
            for future in concurrent.futures.as_completed(futures):
                obj = futures[future]
                descriptor_step2_by_object[obj] = future.result()
                done += 1
                last_log_at = self._log_progress(
                    image_name=image_name,
                    stage="Descriptor",
                    done=done,
                    total=total,
                    last_log_at=last_log_at,
                )

        log_data["descriptor"] = descriptor_step2_by_object
        self._save_image_log(image_name, log_data)

        self.logger.info("[%s] Relationship step (objects=%s)...", image_name, len(objects))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    self._analyze_relationship,
                    obj,
                    list(objects.keys()),
                    descriptor_step2_by_object[obj],
                    image_data_url,
                ): obj
                for obj in objects
            }
            total = len(futures)
            done = 0
            last_log_at = time.monotonic()
            for future in concurrent.futures.as_completed(futures):
                obj = futures[future]
                relationship_step2_by_object[obj] = future.result()
                done += 1
                last_log_at = self._log_progress(
                    image_name=image_name,
                    stage="Relationship",
                    done=done,
                    total=total,
                    last_log_at=last_log_at,
                )

        log_data["relationship"] = relationship_step2_by_object
        self._save_image_log(image_name, log_data)

        summarizer_step1_by_object: dict[str, str] = {}
        self.logger.info("[%s] Summarizer step1 (objects=%s)...", image_name, len(objects))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    self._summarize_object_step1,
                    obj,
                    descriptor_step2_by_object[obj],
                    relationship_step2_by_object[obj],
                    image_data_url,
                ): obj
                for obj in objects
            }
            total = len(futures)
            done = 0
            last_log_at = time.monotonic()
            for future in concurrent.futures.as_completed(futures):
                obj = futures[future]
                summarizer_step1_by_object[obj] = future.result()
                done += 1
                last_log_at = self._log_progress(
                    image_name=image_name,
                    stage="Summarizer step1",
                    done=done,
                    total=total,
                    last_log_at=last_log_at,
                )

        log_data["summarizer"]["step1"] = summarizer_step1_by_object
        self._save_image_log(image_name, log_data)

        self.logger.info(
            "[%s] Summarizer step2/3 (runs=%s)...",
            image_name,
            self.config.num_summarizer_step2,
        )
        summarizer_everyone_step1 = self._format_summarizer_step1(summarizer_step1_by_object)
        step3_text, step3_items = self._summarize_step23(
            summarizer_everyone_step1=summarizer_everyone_step1,
            image_data_url=image_data_url,
            image_name=image_name,
        )

        end_time = time.perf_counter()
        elapsed_minutes = (end_time - start_time) / 60.0

        log_data["summarizer"]["step3_raw"] = step3_text
        log_data["final"] = {"anomalies": step3_items, "elapsed_minutes": elapsed_minutes}
        log_data["usage"] = self.chat.usage.to_dict()
        self._save_image_log(image_name, log_data)

        self.logger.info(
            "Done %s in %.2f min (tokens=%s).",
            image_name,
            elapsed_minutes,
            self.chat.usage.total_tokens,
        )

        return step3_text, step3_items

    def _get_objects(self, *, image_data_url: str) -> dict[str, str]:
        last_error: Optional[Exception] = None
        max_parse_retries = 5
        for attempt in range(1, max_parse_retries + 1):
            try:
                if self.config.num_get_objects > 1:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=self.config.max_workers
                    ) as executor:
                        drafts = list(
                            executor.map(
                                lambda _: self._analyze_all_objects_once(
                                    image_data_url=image_data_url
                                ),
                                range(self.config.num_get_objects),
                            )
                        )

                    merged = self._multi_analyze_all_objects(
                        image_data_url=image_data_url, drafts=drafts
                    )
                    return parse_object_descriptions(merged)

                text = self._analyze_all_objects_once(image_data_url=image_data_url)
                return parse_object_descriptions(text)
            except Exception as e:
                last_error = e
                self.logger.warning(
                    "Object parsing failed (attempt %s/%s): %s",
                    attempt,
                    max_parse_retries,
                    e,
                )

        raise RuntimeError("Failed to analyze and parse objects") from last_error

    def _analyze_all_objects_once(self, *, image_data_url: str) -> str:
        messages = build_messages_analyze_all_objects(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=image_data_url,
        )
        response = self.chat.create_chat_completion(
            messages=messages,
            max_tokens=2000,
            temperature=0.3,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        text = response.choices[0].message.content
        self.logger.debug("analyze_all_objects:\n%s", text)
        return text

    def _multi_analyze_all_objects(self, *, image_data_url: str, drafts: list[str]) -> str:
        messages = build_messages_multi_analyze_all_objects(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=image_data_url,
            multi_get_objects=drafts,
        )
        response = self.chat.create_chat_completion(
            messages=messages,
            max_tokens=2000,
            temperature=0.3,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        text = response.choices[0].message.content
        self.logger.debug("_multi_analyze_all_objects:\n%s", text)
        return text

    def _analyze_descriptor(self, object_name: str, image_data_url: str) -> str:
        messages1 = build_messages_descriptor_step1(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=image_data_url,
            object_name=object_name,
        )
        r1 = self.chat.create_chat_completion(
            messages=messages1,
            max_tokens=2000,
            temperature=0.3,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        step1_text = r1.choices[0].message.content
        self.logger.debug("%s: Descriptor Step1:\n%s", object_name, step1_text)

        messages2 = build_messages_descriptor_step2(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=image_data_url,
            object_name=object_name,
            descriptor_step1_response=step1_text,
        )
        r2 = self.chat.create_chat_completion(
            messages=messages2,
            max_tokens=2000,
            temperature=0.3,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        step2_text = r2.choices[0].message.content
        self.logger.debug("%s: Descriptor Step2:\n%s", object_name, step2_text)
        return step2_text

    def _analyze_relationship(
        self,
        object_name: str,
        all_objects: list[str],
        descriptor_step2_response: str,
        image_data_url: str,
    ) -> str:
        other_objects = [o for o in all_objects if o != object_name]
        all_other_objects_csv = ", ".join(other_objects)

        messages1 = build_messages_relationship_step1(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=image_data_url,
            object_name=object_name,
            all_other_objects_csv=all_other_objects_csv,
            descriptor_step2_response=descriptor_step2_response,
        )
        r1 = self.chat.create_chat_completion(
            messages=messages1,
            max_tokens=2000,
            temperature=0.3,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        step1_text = r1.choices[0].message.content
        self.logger.debug("%s: Relationship Step1:\n%s", object_name, step1_text)

        messages2 = build_messages_relationship_step2(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=image_data_url,
            object_name=object_name,
            all_other_objects_csv=all_other_objects_csv,
            relationship_step1_response=step1_text,
        )
        r2 = self.chat.create_chat_completion(
            messages=messages2,
            max_tokens=2000,
            temperature=0.3,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        step2_text = r2.choices[0].message.content
        self.logger.debug("%s: Relationship Step2:\n%s", object_name, step2_text)
        return step2_text

    def _summarize_object_step1(
        self,
        object_name: str,
        descriptor_step2_response: str,
        relationship_step2_response: str,
        image_data_url: str,
    ) -> str:
        messages = build_messages_summarizer_step1(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=image_data_url,
            object_name=object_name,
            descriptor_step2_response=descriptor_step2_response,
            relationship_step2_response=relationship_step2_response,
        )
        r = self.chat.create_chat_completion(
            messages=messages,
            max_tokens=2000,
            temperature=0.3,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        text = r.choices[0].message.content
        self.logger.debug("%s: Summarizer Step1:\n%s", object_name, text)
        return text

    def _format_summarizer_step1(self, by_object: dict[str, str]) -> str:
        parts = []
        for object_name, text in by_object.items():
            parts.append(
                f"List of the semantically unnatural expressions on {object_name}: {text}\n{'%' * 20}\n"
            )
        return "".join(parts)

    def _summarize_step23(
        self, *, summarizer_everyone_step1: str, image_data_url: str, image_name: str
    ) -> tuple[str, list[dict]]:
        step2_texts: list[str] = []
        reasoning_texts: list[str] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(
                    self._summarize_step2_once,
                    summarizer_everyone_step1=summarizer_everyone_step1,
                    image_data_url=image_data_url,
                )
                for _ in range(self.config.num_summarizer_step2)
            ]
            total = len(futures)
            done = 0
            last_log_at = time.monotonic()
            for future in concurrent.futures.as_completed(futures):
                step2_texts.append(future.result())
                done += 1
                last_log_at = self._log_progress(
                    image_name=image_name,
                    stage="Summarizer step2",
                    done=done,
                    total=total,
                    last_log_at=last_log_at,
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(self._step1_reasoning_once, image_data_url=image_data_url)
                for _ in range(self.config.num_summarizer_step2)
            ]
            total = len(futures)
            done = 0
            last_log_at = time.monotonic()
            for future in concurrent.futures.as_completed(futures):
                reasoning_texts.append(future.result())
                done += 1
                last_log_at = self._log_progress(
                    image_name=image_name,
                    stage="Step1 reasoning",
                    done=done,
                    total=total,
                    last_log_at=last_log_at,
                )

        anomalies_text = self._merge_texts(step2_texts + reasoning_texts)

        last_error: Optional[Exception] = None
        max_parse_retries = 5
        for attempt in range(1, max_parse_retries + 1):
            try:
                step3_text = self._summarize_step3_once(
                    anomalies_text=anomalies_text, image_data_url=image_data_url
                )
                items = parse_anomaly_list(step3_text)
                if not items:
                    raise ValueError("Empty parsed anomaly list")
                return step3_text, items
            except Exception as e:
                last_error = e
                self.logger.warning(
                    "Summarizer step3 parsing failed (attempt %s/%s): %s",
                    attempt,
                    max_parse_retries,
                    e,
                )

        raise RuntimeError("Failed to generate Step3 anomalies") from last_error

    def _summarize_step2_once(self, *, summarizer_everyone_step1: str, image_data_url: str) -> str:
        messages = build_messages_summarizer_step2(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=image_data_url,
            summarizer_everyone_step1=summarizer_everyone_step1,
        )
        r = self.chat.create_chat_completion(
            messages=messages,
            max_tokens=2000,
            temperature=0.3,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        text = r.choices[0].message.content
        self.logger.debug("Summarizer Step2:\n%s", text)
        return text

    def _step1_reasoning_once(self, *, image_data_url: str) -> str:
        messages = build_messages_step1_reasoning(image_data_url=image_data_url)
        r = self.chat.create_chat_completion(
            messages=messages,
            max_tokens=2000,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        text = r.choices[0].message.content
        self.logger.debug("Step1 reasoning:\n%s", text)
        return text

    def _summarize_step3_once(self, *, anomalies_text: str, image_data_url: str) -> str:
        messages = build_messages_summarizer_step3(
            system_prompt=self.system_prompt,
            expert_prompt=self.expert_prompt,
            image_data_url=image_data_url,
            anomalies_text=anomalies_text,
        )
        r = self.chat.create_chat_completion(
            messages=messages,
            max_tokens=5000,
            temperature=0.3,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        text = r.choices[0].message.content
        self.logger.debug("Summarizer Step3:\n%s", text)
        return text

    def _merge_texts(self, texts: Iterable[str], *, separator: str = "\n------\n") -> str:
        return separator.join([t for t in texts if isinstance(t, str) and t.strip()])

    def _log_progress(
        self,
        *,
        image_name: str,
        stage: str,
        done: int,
        total: int,
        last_log_at: float,
        min_interval_seconds: float = 20.0,
    ) -> float:
        if total <= 0:
            return last_log_at

        now = time.monotonic()
        step = max(1, total // 10)
        should_log = done in {1, total} or done % step == 0 or (now - last_log_at) >= min_interval_seconds
        if should_log:
            self.logger.info("[%s] %s progress: %s/%s", image_name, stage, done, total)
            return now
        return last_log_at
