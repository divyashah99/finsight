"""Minimal stdlib logging config — single setup function called from app startup."""

from __future__ import annotations

import logging
import sys

from finsight.settings import settings


def configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        level=level,
    )


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or "finsight")
