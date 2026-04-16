from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Union


def build_logger(
    *,
    name: str,
    log_level: Union[str, int] = "INFO",
    log_to_file: bool = True,
    log_dir: Optional[str] = None,
    log_filename: str = "run.log",
) -> tuple[logging.Logger, Optional[str]]:
    if isinstance(log_level, str):
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        level = int(log_level)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = None
    if log_to_file and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger, log_path
