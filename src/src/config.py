from pathlib import Path
import os

# Toggle to safely isolate tests from production artifacts
TEST_MODE = os.getenv("TEST_MODE", "0") == "1"

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = (BASE_DIR / "artifacts_test") if TEST_MODE else (BASE_DIR / "artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = ARTIFACT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = ARTIFACT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = ARTIFACT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_COL = "target"
COUNTRY_COL = "country"
