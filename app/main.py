import joblib
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Load saved model & encoders ───────────────────────────────
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
le_area = joblib.load("model/le_area.pkl")
le_item = joblib.load("model/le_item.pkl")

with open("model/benchmark.json") as f:
    benchmark_results = json.load(f)

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Crop Yield Predictor",
    description="Predicts crop yield using a Random Forest model trained on global agricultural data.",
    version="1.0.0",
)


# ── Input schema ──────────────────────────────────────────────
class CropInput(BaseModel):
    area: str
    item: str
    year: int
    average_rain_fall_mm_per_year: float
    pesticides_tonnes: float
    avg_temp: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "area": "Albania",
                "item": "Maize",
                "year": 2013,
                "average_rain_fall_mm_per_year": 1485.0,
                "pesticides_tonnes": 121.0,
                "avg_temp": 16.37,
            }
        }
    }


# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Crop Yield Predictor API is running"}


@app.get("/options")
def get_options():
    """Returns valid values for area and item fields."""
    return {
        "areas": sorted(le_area.classes_.tolist()),
        "crops": sorted(le_item.classes_.tolist()),
    }


@app.post("/predict")
def predict(data: CropInput):
    """Predicts crop yield in hg/ha given farming conditions."""

    # validate area and item against known labels
    if data.area not in le_area.classes_:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown area '{data.area}'. Call /options to see valid values.",
        )
    if data.item not in le_item.classes_:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown crop '{data.item}'. Call /options to see valid values.",
        )

    # encode and scale
    area_enc = le_area.transform([data.area])[0]
    item_enc = le_item.transform([data.item])[0]

    features = pd.DataFrame(
        [
            [
                area_enc,
                item_enc,
                data.year,
                data.average_rain_fall_mm_per_year,
                data.pesticides_tonnes,
                data.avg_temp,
            ]
        ],
        columns=[
            "Area_encoded",
            "Item_encoded",
            "Year",
            "average_rain_fall_mm_per_year",
            "pesticides_tonnes",
            "avg_temp",
        ],
    )

    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]

    return {
        "area": data.area,
        "crop": data.item,
        "year": data.year,
        "predicted_yield_hg_per_ha": round(float(prediction), 2),
        "predicted_yield_tonnes_per_ha": round(float(prediction) / 10000, 4),
        "model_used": "RandomForest",
        "model_r2_score": benchmark_results["RandomForest"]["R2"],
    }


@app.get("/benchmark")
def benchmark():
    """Returns performance comparison of all 3 trained models."""
    return {
        "models": benchmark_results,
        "best_model": "RandomForest",
        "metric": "R² score (higher is better)",
        "note": "Linear Regression performs poorly — confirms non-linear relationships in crop data",
    }
