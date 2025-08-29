from pathlib import Path
import joblib
from typing import Any, Dict
from .config import MODEL_DIR

def save_model(name: str, model: Any, meta: Dict) -> Path:
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump({"model": model, "meta": meta}, path)
    return path

def load_model(name: str):
    data = joblib.load(MODEL_DIR / f"{name}.joblib")
    return data["model"], data["meta"]
