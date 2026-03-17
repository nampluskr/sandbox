# defectvad/common/logger.py

import logging
from typing import Optional


class Logger:
    def __init__(self, name=None, level=None):
        self._logger = logging.getLogger(name or "defectvad")
        if level is not None:
            self._logger.setLevel(level)

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        self._logger.exception(msg, *args, exc_info=exc_info, **kwargs)

    @property
    def raw(self) -> logging.Logger:
        """Return underlying logging.Logger"""
        return self._logger


