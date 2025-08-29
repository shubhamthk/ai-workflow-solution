import pandas as pd
from src.data_ingestion import load_dataset
from src.features import build_preprocessor
from src.config import TARGET_COL, COUNTRY_COL
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def test_model_train_and_predict():
    df = load_dataset()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    pipe = Pipeline([("pre", build_preprocessor(df)), ("clf", LogisticRegression(max_iter=200))])
    pipe.fit(X, y)
    preds = pipe.predict(X.head(5))
    assert len(preds) == 5
