from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
from typing import List, Optional
from .model_store import load_model
from .config import COUNTRY_COL
from .monitoring import monitor

app = FastAPI(title="Country Prediction API", version="1.0.0")
model, meta = load_model("current")

class Item(BaseModel):
    feature_a: float
    feature_b: float
    country: str

@monitor.track
@app.post("/predict")
def predict(items: List[Item]):
    df = pd.DataFrame([i.model_dump() for i in items])
    proba = model.predict_proba(df)[:, 1].tolist()
    preds = (model.predict(df)).tolist()
    return {"predictions": preds, "probabilities": proba}

@monitor.track
@app.get("/predict/country")
def predict_by_country(
    country: Optional[str] = Query(default=None, description="Specific country or None for all")
):
    # Return a template showing the accepted country list and example probabilities.
    ohe_countries = [c for c in model.named_steps["pre"].transformers_[1][1].categories_[0]]
    if country and country not in ohe_countries:
        return {"error": "Unknown country", "available_countries": ohe_countries}

    inputs = []
    for c in (ohe_countries if country is None else [country]):
        inputs.append({"feature_a": 0.0, "feature_b": 0.0, "country": c})
    df = pd.DataFrame(inputs)
    proba = model.predict_proba(df)[:, 1].tolist()
    return {"countries": [r[COUNTRY_COL] for r in inputs], "probabilities": proba}

@app.get("/metrics")
def metrics():
    return monitor.metrics()
