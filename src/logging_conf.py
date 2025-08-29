import logging
import logging.handlers
from pathlib import Path
from .config import LOG_DIR

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fh = logging.handlers.RotatingFileHandler(
        Path(LOG_DIR) / "app.log", maxBytes=2_000_000, backupCount=3
    )
    ch = logging.StreamHandler()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
