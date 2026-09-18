"""AnomAgent V2: agent-ified semantic anomaly detection & reasoning for AI-generated images.

Controller + 7 tools (detect_objects / crop / inspect / check_relation /
verify / global_scan / finish) with strict JSON-schema outputs, image token
governance (tiered resolution + grounding crops + JPEG), elastic max_tokens,
step-level caching, per-step token ledgers, model routing, inter-image
parallelism and per-sample HTML visualization.
"""

__version__ = "2.0.0"
