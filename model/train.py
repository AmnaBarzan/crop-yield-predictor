import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# ── 1. Load data ──────────────────────────────────────────────
df = pd.read_csv("data/yield_df.csv")
df = df.drop(columns=["Unnamed: 0"])

print("Dataset loaded:", df.shape)

# ── 2. Encode categorical columns ─────────────────────────────
# Area and Item are text — we convert them to numbers
le_area = LabelEncoder()
le_item = LabelEncoder()

df["Area_encoded"] = le_area.fit_transform(df["Area"])
df["Item_encoded"] = le_item.fit_transform(df["Item"])

# ── 3. Define features and target ─────────────────────────────
FEATURES = [
    "Area_encoded",
    "Item_encoded",
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]
TARGET = "hg/ha_yield"

X = df[FEATURES]
y = df[TARGET]

# ── 4. Split into train and test sets ─────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# ── 5. Scale features ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 6. Train 3 models ─────────────────────────────────────────
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results[name] = {"RMSE": round(rmse, 2), "R2": round(r2, 4)}
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  R²   : {r2:.4f}")

# ── 7. Pick best model (highest R²) ───────────────────────────
best_name = max(results, key=lambda k: results[k]["R2"])
best_model = models[best_name]
print(f"\n✅ Best model: {best_name} (R²={results[best_name]['R2']})")

# ── 8. Save everything ────────────────────────────────────────
joblib.dump(best_model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(le_area, "model/le_area.pkl")
joblib.dump(le_item, "model/le_item.pkl")

with open("model/benchmark.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved: model.pkl, scaler.pkl, le_area.pkl, le_item.pkl, benchmark.json")
