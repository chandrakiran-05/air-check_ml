"""Data loading, cleaning and model training for India AQI prediction."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "city_day.csv"
MODEL_DIR = ROOT / "models"
POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
              "Benzene", "Toluene", "Xylene"]
FEATURES = POLLUTANTS + ["Month"]
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer",
           6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
           10: "Post-monsoon", 11: "Post-monsoon"}
BUCKETS = [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
           (300, "Poor"), (400, "Very Poor"), (np.inf, "Severe")]


def aqi_bucket(aqi: float) -> str:
    return next(name for limit, name in BUCKETS if aqi <= limit)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW, parse_dates=["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Season"] = df["Month"].map(SEASONS)
    return df


def train(df: pd.DataFrame | None = None) -> dict:
    df = load_data() if df is None else df
    data = df.dropna(subset=["AQI"]).copy()
    # Impute missing pollutant values with the city median, then global median
    for col in POLLUTANTS:
        data[col] = data[col].fillna(data.groupby("City")[col].transform("median"))
        data[col] = data[col].fillna(data[col].median())
    X, y = data[FEATURES], data["AQI"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, max_depth=20, n_jobs=-1, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "r2": round(float(r2_score(y_te, pred)), 4),
        "mae": round(float(mean_absolute_error(y_te, pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_te, pred))), 2),
        "bucket_accuracy": round(float(np.mean(
            [aqi_bucket(a) == aqi_bucket(b) for a, b in zip(y_te, pred)])), 4),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "importance": dict(sorted(zip(FEATURES, model.feature_importances_.round(4).tolist()),
                                  key=lambda kv: -kv[1])),
        "defaults": X.median().round(2).to_dict(),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "aqi_model.joblib", compress=3)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    sample = pd.DataFrame({"actual": y_te.values, "predicted": pred}).sample(1500, random_state=0)
    sample.to_csv(MODEL_DIR / "test_predictions.csv", index=False)
    return metrics


if __name__ == "__main__":
    m = train()
    print({k: v for k, v in m.items() if k not in ("importance", "defaults")})
