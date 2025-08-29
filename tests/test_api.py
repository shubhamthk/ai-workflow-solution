import os
os.environ["TEST_MODE"] = "1"  # isolate artifacts
from fastapi.testclient import TestClient
from src.train import train_and_compare
from src.api import app  # after model exists

def setup_module(module):
    train_and_compare()  # ensure a model is present for API

client = TestClient(app)

def test_predict_single_and_batch():
    body = [{"feature_a": 0.1, "feature_b": -0.2, "country": "C1"}]
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    out = r.json()
    assert "predictions" in out and len(out["predictions"]) == 1

def test_country_endpoint_specific_and_all():
    r1 = client.get("/predict/country", params={"country": "C1"})
    assert r1.status_code == 200
    r2 = client.get("/predict/country")
    assert r2.status_code == 200
    assert len(r2.json()["countries"]) > 1

def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    m = r.json()
    assert "requests_total" in m
