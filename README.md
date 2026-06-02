# 🌾 Crop Yield Predictor API

A production-ready machine learning API that predicts agricultural crop yields based on environmental and farming conditions. Built with FastAPI and deployed on Railway.

**Live API:** [crop-yield-predictor-production-f9b6.up.railway.app](https://crop-yield-predictor-production-f9b6.up.railway.app/)  
**Interactive Docs:** [/docs](https://crop-yield-predictor-production-f9b6.up.railway.app/docs)

---

## Overview

This project trains and serves a Random Forest regression model on a global agricultural dataset covering 101 countries and 10 crop types (1990–2013). Given farming conditions, the API predicts crop yield in hg/ha with 98.57% accuracy (R²).

The project includes a full ML pipeline — data processing, model benchmarking, a REST API backend, edge case handling, automated tests, and CI/CD.

---

## Results

| Model | RMSE | R² Score |
|---|---|---|
| Linear Regression | 81,501 | 0.084 |
| XGBoost | 13,451 | 0.975 |
| **Random Forest** | **10,178** | **0.986** ✅ |

Linear Regression's poor performance confirms non-linear relationships in crop yield data — tree-based models capture interactions between rainfall, temperature, and pesticide use that linear models miss entirely.

---

## API Endpoints

### `POST /predict`
Predict crop yield given farming conditions.

**Request body:**
```json
{
  "area": "Sri Lanka",
  "item": "Rice, paddy",
  "year": 2010,
  "average_rain_fall_mm_per_year": 1712.0,
  "pesticides_tonnes": 500.0,
  "avg_temp": 28.0
}
```

**Response:**
```json
{
  "area": "Sri Lanka",
  "crop": "Rice, paddy",
  "year": 2010,
  "predicted_yield_hg_per_ha": 38245.60,
  "predicted_yield_tonnes_per_ha": 3.8246,
  "model_used": "RandomForest",
  "model_r2_score": 0.9857
}
```

### `GET /benchmark`
Returns performance comparison of all 3 trained models.

### `GET /options`
Returns all valid values for `area` (101 countries) and `item` (10 crops).

### `GET /`
Health check.

---

## Project Structure

```
crop-yield-predictor/
│
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI app with all routes
│
├── model/
│   ├── explore.py           # EDA and dataset visualisation
│   ├── train.py             # Train & benchmark 3 models, save artifacts
│   ├── model.pkl            # Saved Random Forest model
│   ├── scaler.pkl           # StandardScaler
│   ├── le_area.pkl          # LabelEncoder for country
│   ├── le_item.pkl          # LabelEncoder for crop type
│   └── benchmark.json       # Model comparison results
│
├── data/                    # Dataset (downloaded via Kaggle API)
│
├── tests/
│   ├── __init__.py
│   └── test_api.py          # 9 pytest tests covering all endpoints + edge cases
│
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI pipeline
│
├── railway.toml             # Railway deployment config
├── requirements.txt
├── Makefile
├── conftest.py
└── README.md
```

---

## Local Setup

**Prerequisites:** Python 3.11+, Git

```bash
# 1. Clone the repo
git clone https://github.com/AmnaBarzan/crop-yield-predictor.git
cd crop-yield-predictor

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
make install

# 4. Download dataset (requires Kaggle API token)
export KAGGLE_API_TOKEN=your_token_here
pip install kaggle
kaggle datasets download -d patelris/crop-yield-prediction-dataset -p data/ --unzip

# 5. Train the model
python model/train.py

# 6. Run the API
uvicorn app.main:app --reload --port 8080
```

Open [http://localhost:8080/docs](http://localhost:8080/docs) for the interactive Swagger UI.

---

## Running Tests

```bash
# Run all tests with verbose output
make test

# Format code
make format

# Lint code
make lint

# Run everything
make all
```

**Test coverage:**
- Root health check
- `/options` returns valid areas and crops including Sri Lanka
- `/benchmark` contains all 3 models and Random Forest wins
- `/predict` with valid input returns positive yield
- `/predict` with Sri Lanka + Rice, paddy
- `/predict` with unknown country returns 400 with clear error
- `/predict` with unknown crop returns 400 with clear error
- `/predict` with missing fields returns 422 validation error

---

## CI/CD Pipeline

Every push to `main` triggers a GitHub Actions workflow that:

1. Installs all dependencies
2. Checks code formatting with Black
3. Downloads the dataset from Kaggle
4. Trains the model from scratch
5. Runs all 9 pytest tests

Railway auto-deploys on every push with the same build pipeline — dataset download → model training → API startup.

---

## Dataset

**Source:** [Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset) via Kaggle  
**Size:** 28,242 records · 8 features  
**Coverage:** 101 countries · 10 crop types · 1990–2013  
**Features:** Country, crop type, year, rainfall (mm/year), pesticides (tonnes), average temperature  
**Target:** Crop yield in hg/ha  
**Missing values:** None

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI |
| ML models | Scikit-Learn, XGBoost |
| Data processing | pandas, NumPy |
| Input validation | Pydantic v2 |
| Testing | pytest, httpx |
| Code quality | Black, Pylint |
| CI/CD | GitHub Actions |
| Deployment | Railway |

---

## Author

**Amna Barzan**    
[GitHub](https://github.com/AmnaBarzan)
