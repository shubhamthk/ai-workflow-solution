import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from src.data_ingestion import load_dataset
from src.config import ARTIFACT_DIR, TARGET_COL, COUNTRY_COL

def run_eda():
    df = load_dataset()
    fig1 = plt.figure(figsize=(5,5))
    df.boxplot(column="feature_a", by=COUNTRY_COL)
    plt.title("Feature A by Country"); plt.suptitle("")
    plt.savefig(ARTIFACT_DIR / "eda_feature_a_by_country.png", bbox_inches="tight")

    fig2 = plt.figure(figsize=(5,5))
    df.groupby(COUNTRY_COL)[TARGET_COL].mean().plot(kind="bar")
    plt.title("Target rate by Country"); plt.ylabel("Rate")
    plt.savefig(ARTIFACT_DIR / "eda_target_rate_by_country.png", bbox_inches="tight")

if __name__ == "__main__":
    run_eda()
