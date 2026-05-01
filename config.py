"""Project-wide configuration helpers."""

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure a consistent logging format and level for the project."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
