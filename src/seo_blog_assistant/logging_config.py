"""Application logging configuration."""

import logging


def configure_logging(level: str) -> None:
    """Configure a concise root logger without replacing existing handlers."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger().setLevel(level)
