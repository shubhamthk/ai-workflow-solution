"""
Reusable ingestion function/script for automation.
Generates or loads a simple tabular dataset with a 'country' column.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from .config import DATA_DIR, COUNTRY_COL, TARGET_COL, RANDOM_STATE
from .logging_conf import get_logger

logger = get_logger(__name__)

def create_synthetic(path: Path = DATA_DIR / "data.csv", n=1500, n_countries=5) -> Path:
    rng = np.random.default_rng(RANDOM_STATE)
    countries = [f"C{i}" for i in range(1, n_countries + 1)]
    df = pd.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(2, 1.5, n),
        COUNTRY_COL: rng.choice(countries, n),
    })
    # synthetic target varies by country
    logits = 0.8*df["feature_a"] - 0.4*df["feature_b"] + df[COUNTRY_COL].map(
        {c:i*0.3 for i,c in enumerate(countries)}
    )
    p = 1 / (1 + np.exp(-logits))
    df[TARGET_COL] = (rng.random(n) < p).astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved dataset: %s rows -> %s", len(df), path)
    return path

def load_dataset(path: Path = DATA_DIR / "data.csv") -> pd.DataFrame:
    if not Path(path).exists():
        create_synthetic(path)
    df = pd.read_csv(path)
    logger.info("Loaded dataset: %s rows from %s", len(df), path)
    return df

if __name__ == "__main__":
    create_synthetic()
