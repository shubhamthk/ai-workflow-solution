from pathlib import Path
from src.logging_conf import get_logger
from src.config import LOG_DIR

def test_logging_written(tmp_path, monkeypatch):
    # redirect logs to temp folder to isolate
    monkeypatch.setenv("TEST_MODE", "1")
    logger = get_logger("test")
    logger.info("hello world")
    files = list(Path(LOG_DIR).glob("app.log*"))
    assert len(files) >= 1
