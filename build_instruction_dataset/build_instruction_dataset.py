#!/usr/bin/env python3
"""Public CLI entrypoint for building instruction-style training datasets.

Examples:
    python3 build_instruction_dataset.py \
        --recipe authenticity_reasoning \
        --fake-input fake_img_reasoning_100_case.json \
        --real-input real_img_caption_100case.json \
        --output outputs/authenticity_reasoning.json
"""

from instruction_dataset_builder import RECIPE_AUTHENTICITY_REASONING, run_cli


if __name__ == "__main__":
    raise SystemExit(run_cli(default_recipe=RECIPE_AUTHENTICITY_REASONING))
