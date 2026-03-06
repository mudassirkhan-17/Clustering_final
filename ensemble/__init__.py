import logging

class _SimpleLogger:
    def __init__(self, level="info"):
        self._logger = logging.getLogger("pipeline")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        if not self._logger.handlers:
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)
        self._level = level

    def log(self, msg):
        if self._level == "info":
            self._logger.info(msg)
        elif self._level == "error":
            self._logger.error(msg)
        elif self._level == "warning":
            self._logger.warning(msg)

logger_object = {
    "info":    _SimpleLogger("info"),
    "error":   _SimpleLogger("error"),
    "warning": _SimpleLogger("warning"),
}