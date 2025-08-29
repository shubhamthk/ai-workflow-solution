import time
from typing import Callable, Dict
from .logging_conf import get_logger

logger = get_logger("monitor")

class SimpleMonitor:
    def __init__(self):
        self.counters: Dict[str, int] = {"requests_total": 0, "errors_total": 0}
        self.latency_ms: float = 0.0

    def track(self, fn: Callable):
        def wrapped(*args, **kwargs):
            start = time.monotonic()
            self.counters["requests_total"] += 1
            try:
                res = fn(*args, **kwargs)
                return res
            except Exception:
                self.counters["errors_total"] += 1
                logger.exception("API error")
                raise
            finally:
                self.latency_ms = (time.monotonic() - start) * 1000
        return wrapped

    def metrics(self):
        return {
            "requests_total": self.counters["requests_total"],
            "errors_total": self.counters["errors_total"],
            "last_latency_ms": round(self.latency_ms, 2),
        }

monitor = SimpleMonitor()
