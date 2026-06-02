# uvicorn app.main:app --reload --port 8080
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ── Root ──────────────────────────────────────────────────────
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


# ── Options ───────────────────────────────────────────────────
def test_options_returns_areas_and_crops():
    response = client.get("/options")
    assert response.status_code == 200
    data = response.json()
    assert "areas" in data
    assert "crops" in data
    assert "Sri Lanka" in data["areas"]
    assert "Maize" in data["crops"]


# ── Benchmark ─────────────────────────────────────────────────
def test_benchmark_has_all_models():
    response = client.get("/benchmark")
    assert response.status_code == 200
    models = response.json()["models"]
    assert "LinearRegression" in models
    assert "RandomForest" in models
    assert "XGBoost" in models


def test_benchmark_random_forest_is_best():
    response = client.get("/benchmark")
    models = response.json()["models"]
    best_r2 = max(m["R2"] for m in models.values())
    assert models["RandomForest"]["R2"] == best_r2


# ── Predict — happy path ──────────────────────────────────────
def test_predict_valid_input():
    payload = {
        "area": "Albania",
        "item": "Maize",
        "year": 2013,
        "average_rain_fall_mm_per_year": 1485.0,
        "pesticides_tonnes": 121.0,
        "avg_temp": 16.37,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["area"] == "Albania"
    assert data["crop"] == "Maize"
    assert data["predicted_yield_hg_per_ha"] > 0
    assert data["model_used"] == "RandomForest"


def test_predict_sri_lanka_rice():
    payload = {
        "area": "Sri Lanka",
        "item": "Rice, paddy",
        "year": 2010,
        "average_rain_fall_mm_per_year": 1712.0,
        "pesticides_tonnes": 500.0,
        "avg_temp": 28.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predicted_yield_hg_per_ha"] > 0


# ── Predict — edge cases ──────────────────────────────────────
def test_predict_invalid_area():
    payload = {
        "area": "Wakanda",
        "item": "Maize",
        "year": 2013,
        "average_rain_fall_mm_per_year": 1485.0,
        "pesticides_tonnes": 121.0,
        "avg_temp": 16.37,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Unknown area" in response.json()["detail"]


def test_predict_invalid_crop():
    payload = {
        "area": "Albania",
        "item": "Dragon Fruit",
        "year": 2013,
        "average_rain_fall_mm_per_year": 1485.0,
        "pesticides_tonnes": 121.0,
        "avg_temp": 16.37,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Unknown crop" in response.json()["detail"]


def test_predict_missing_field_returns_422():
    payload = {
        "area": "Albania",
        "item": "Maize",
        # missing all other fields
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
