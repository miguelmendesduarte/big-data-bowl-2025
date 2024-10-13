"""Logging configuration."""

import logging
import logging.config

from .settings import get_settings

settings = get_settings()

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "style": "{",
            "format": settings.LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "level": settings.LOG_LEVEL,
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": settings.LOG_LEVEL,
            "propagate": False,
        },
    },
}


def configure_logging() -> None:
    """Configure custom logging."""
    logging.config.dictConfig(LOGGING_CONFIG)
