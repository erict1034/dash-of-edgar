import logging
import sys
from pathlib import Path


_CONFIGURED = False


def _ensure_error_handler():
    global _CONFIGURED
    if _CONFIGURED:
        return
    data_dir = Path(__file__).with_name("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "central_errors.log"
    handler = logging.FileHandler(str(log_path))
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    def _excepthook(exc_type, exc, tb):
        logger = logging.getLogger("uncaught")
        logger.error("Uncaught exception", exc_info=(exc_type, exc, tb))
        if sys.__excepthook__:
            sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _excepthook
    _CONFIGURED = True


def get_error_logger(name: str) -> logging.Logger:
    _ensure_error_handler()
    return logging.getLogger(name)
