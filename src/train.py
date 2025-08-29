"""
Train & compare multiple models, save the best, and produce a baseline comparison plot.
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.pipeline import Pipeline

from .data_ingestion import load_dataset
from .features import train_val_split, build_preprocessor
from .model_store import save_model
from .config import ARTIFACT_DIR, TARGET_COL, COUNTRY_COL, RANDOM_STATE
from .logging_conf import get_logger

logger = get_logger(__name__)

def train_and_compare() -> Path:
    df = load_dataset()
    train_df, val_df = train_val_split(df)

    X_tr, y_tr = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
    X_va, y_va = val_df.drop(columns=[TARGET_COL]), val_df[TARGET_COL]

    pre = build_preprocessor(df)

    models = {
        "baseline_logreg": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }

    scores = {}
    roc_fig = plt.figure(figsize=(5,5))
    ax = plt.gca()

    best_name, best_score, best_pipe = None, -1, None
    for name, estimator in models.items():
        pipe = Pipeline([("pre", pre), ("clf", estimator)])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]
        score = roc_auc_score(y_va, proba)
        scores[name] = float(score)
        RocCurveDisplay.from_predictions(y_va, proba, name=name, ax=ax)
        if score > best_score:
            best_name, best_score, best_pipe = name, score, pipe
        logger.info("Model %s ROC AUC=%.4f", name, score)

    ax.set_title("Baseline vs Candidate ROC")
    plot_path = ARTIFACT_DIR / "model_comparison_roc.png"
    roc_fig.savefig(plot_path, bbox_inches="tight")

    meta = {"best_model": best_name, "scores": scores, "country_column": COUNTRY_COL}
    model_path = save_model("current", best_pipe, meta)
    with open(ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved best model: %s -> %s", best_name, model_path)
    return model_path

if __name__ == "__main__":
    train_and_compare()
