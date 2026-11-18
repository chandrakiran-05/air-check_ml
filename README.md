# 🌫️ AirSense India — Air Quality Analysis & AQI Prediction (2015–2020)

An end-to-end machine learning project on daily air-quality data from **26 Indian cities** (CPCB data, Kaggle *Air Quality Data in India 2015–2020*). It combines interactive exploratory analysis with a **Random Forest model that predicts AQI** from pollutant readings, served through a Streamlit web app.

## ✨ Features
| Tab | What it shows |
|---|---|
| 📊 Overview | KPIs, median pollutant by city, AQI category share, correlation heatmap |
| 🗓️ Trends & Seasons | Monthly trends, seasonal box plots, COVID-19 lockdown impact (Apr–May 2020 vs 2019) |
| 🤖 AQI Predictor | Enter pollutant values → predicted AQI + colour-coded category |
| 📈 Model Performance | R², MAE, RMSE, category accuracy, feature importance, actual-vs-predicted plot |

Sidebar filters (cities, years, pollutant) update every chart.

## 🧠 Model
- **Target:** AQI · **Features:** PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, Month
- **Preprocessing:** rows without AQI dropped; missing pollutants imputed with city median, then global median
- **Algorithm:** RandomForestRegressor (150 trees, 80/20 split)

| Metric | Score |
|---|---|
| R² | 0.909 |
| MAE | 20.8 |
| RMSE | 40.8 |
| AQI category accuracy | 79.5% |

PM2.5 and PM10 are the dominant drivers of AQI.

## 📁 Structure
```
├── public/                # static web app for Vercel (index.html + JSON data/model)
├── app.py                 # Streamlit dashboard
├── src/export_web.py      # exports web model + data to public/
├── src/pipeline.py        # data loading, cleaning, training, evaluation
├── data/raw/city_day.csv  # dataset
├── models/                # metrics.json, test predictions (model .joblib generated)
├── .streamlit/config.toml # theme
└── requirements.txt
```

## 🌐 Web version (Vercel)
`public/` is a static site (HTML + Plotly) with the same four tabs. The AQI predictor runs a
gradient-boosting model (R² 0.90) exported to `public/model.json`, so predictions happen in the browser with no server.
Regenerate it with `python -m src.export_web`. Vercel deploys `public/` directly (see `vercel.json`).

## 🚀 Run locally
```bash
pip install -r requirements.txt
python -m src.pipeline      # trains model (optional — app auto-trains on first run)
streamlit run app.py
```

## 📌 Key insights
- Northern cities (Delhi, Patna, Lucknow) have the worst air, especially in **winter**.
- **Monsoon** months are the cleanest across all cities.
- The **2020 lockdown** caused sharp drops in PM2.5 and NO2 in most cities.

## 📜 Data source
Vopani (Rohan Rao), *Air Quality Data in India (2015–2020)*, Kaggle — from Central Pollution Control Board (CPCB).
