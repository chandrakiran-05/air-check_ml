"""AirSense India — Streamlit dashboard for India air quality (2015-2020) + AQI prediction."""
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline import (FEATURES, MODEL_DIR, POLLUTANTS, aqi_bucket, load_data, train)

st.set_page_config(page_title="AirSense India", page_icon="🌫️", layout="wide")

BUCKET_COLORS = {"Good": "#2e9d5b", "Satisfactory": "#9ccc3c", "Moderate": "#f2c12e",
                 "Poor": "#f08a24", "Very Poor": "#d7372f", "Severe": "#7d1f3c"}
ORDER = list(BUCKET_COLORS)

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.hero {background: linear-gradient(120deg,#0f4c75,#3282b8); color:#fff; padding:1.4rem 1.8rem;
       border-radius:16px; margin-bottom:1rem;}
.hero h1 {margin:0; font-size:2rem; color:#fff;} .hero p {margin:.3rem 0 0; opacity:.9;}
div[data-testid="stMetric"] {background:rgba(50,130,184,.08); border:1px solid rgba(50,130,184,.25);
       border-radius:12px; padding:.8rem 1rem;}
.badge {display:inline-block; padding:.5rem 1.2rem; border-radius:999px; color:#fff;
        font-weight:700; font-size:1.3rem;}
</style>""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model():
    if not (MODEL_DIR / "aqi_model.joblib").exists():
        with st.spinner("Training model for the first time (≈30s)…"):
            train()
    return (joblib.load(MODEL_DIR / "aqi_model.joblib"),
            json.loads((MODEL_DIR / "metrics.json").read_text()))


df = get_data()
model, metrics = get_model()

st.markdown('<div class="hero"><h1>🌫️ AirSense India</h1><p>Exploring and predicting air quality '
            'across 26 Indian cities, 2015–2020 · CPCB data via Kaggle</p></div>',
            unsafe_allow_html=True)

# ---------- Sidebar filters ----------
with st.sidebar:
    st.header("Filters")
    cities = sorted(df["City"].unique())
    default = [c for c in ["Delhi", "Mumbai", "Bengaluru", "Kolkata", "Chennai", "Lucknow"] if c in cities]
    sel_cities = st.multiselect("Cities", cities, default=default)
    years = st.slider("Years", 2015, 2020, (2015, 2020))
    pollutant = st.selectbox("Pollutant", POLLUTANTS, index=0)
    st.caption("Data: Kaggle *Air Quality Data in India (2015–2020)*")

view = df[df["City"].isin(sel_cities or cities) & df["Year"].between(*years)]
if view.empty:
    st.warning("No data for these filters.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗓️ Trends & Seasons", "🤖 AQI Predictor", "📈 Model Performance"])

# ---------- Overview ----------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Daily records", f"{len(view):,}")
    c2.metric("Median AQI", f"{view['AQI'].median():.0f}")
    c3.metric(f"Median {pollutant}", f"{view[pollutant].median():.1f}")
    poor = view["AQI_Bucket"].isin(["Poor", "Very Poor", "Severe"]).mean() * 100
    c4.metric("Days Poor or worse", f"{poor:.1f}%")

    a, b = st.columns([3, 2])
    city_med = view.groupby("City")[pollutant].median().sort_values().reset_index()
    a.plotly_chart(px.bar(city_med, x=pollutant, y="City", orientation="h", color=pollutant,
                          color_continuous_scale="OrRd", title=f"Median {pollutant} by city"),
                   use_container_width=True)
    share = view.dropna(subset=["AQI_Bucket"])["AQI_Bucket"].value_counts().reindex(ORDER).dropna().reset_index()
    b.plotly_chart(px.pie(share, names="AQI_Bucket", values="count", hole=.5, color="AQI_Bucket",
                          color_discrete_map=BUCKET_COLORS, title="AQI category share"),
                   use_container_width=True)
    corr = view[POLLUTANTS + ["AQI"]].corr().round(2)
    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Pollutant correlation"), use_container_width=True)

# ---------- Trends ----------
with tab2:
    monthly = (view.set_index("Date").groupby("City")[pollutant].resample("MS").mean().reset_index())
    st.plotly_chart(px.line(monthly, x="Date", y=pollutant, color="City",
                            title=f"Monthly mean {pollutant}"), use_container_width=True)
    a, b = st.columns(2)
    season_order = ["Winter", "Summer", "Monsoon", "Post-monsoon"]
    a.plotly_chart(px.box(view, x="Season", y=pollutant, color="Season", points=False,
                          category_orders={"Season": season_order}, title=f"{pollutant} by season"),
                   use_container_width=True)
    # COVID-19 lockdown: Apr-May 2020 vs Apr-May 2019
    am = df[df["City"].isin(sel_cities or cities) & df["Month"].isin([4, 5]) & df["Year"].isin([2019, 2020])]
    lock = am.pivot_table(index="City", columns="Year", values=pollutant, aggfunc="mean").dropna()
    if not lock.empty:
        lock["Change %"] = (lock[2020] / lock[2019] - 1) * 100
        b.plotly_chart(px.bar(lock.reset_index().sort_values("Change %"), x="Change %", y="City",
                              orientation="h", color="Change %", color_continuous_scale="RdYlGn_r",
                              title=f"Lockdown effect: {pollutant}, Apr–May 2020 vs 2019"),
                       use_container_width=True)

# ---------- Predictor ----------
with tab3:
    st.subheader("Predict AQI from pollutant readings")
    st.caption("Defaults are dataset medians. Adjust values and the Random Forest model estimates AQI.")
    d = metrics["defaults"]
    with st.form("predict"):
        cols = st.columns(4)
        inputs = {}
        for i, f in enumerate(POLLUTANTS):
            inputs[f] = cols[i % 4].number_input(f, min_value=0.0, value=float(d[f]), step=1.0)
        inputs["Month"] = cols[0].selectbox("Month", list(range(1, 13)), index=0)
        go = st.form_submit_button("Predict AQI", type="primary", use_container_width=True)
    if go:
        aqi = float(model.predict(pd.DataFrame([inputs])[FEATURES])[0])
        bucket = aqi_bucket(aqi)
        r1, r2 = st.columns([1, 2])
        r1.metric("Predicted AQI", f"{aqi:.0f}")
        r2.markdown(f'<br><span class="badge" style="background:{BUCKET_COLORS[bucket]}">{bucket}</span>',
                    unsafe_allow_html=True)

# ---------- Model performance ----------
with tab4:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² score", metrics["r2"])
    c2.metric("MAE", metrics["mae"])
    c3.metric("RMSE", metrics["rmse"])
    c4.metric("Category accuracy", f"{metrics['bucket_accuracy']*100:.1f}%")
    a, b = st.columns(2)
    imp = pd.DataFrame(metrics["importance"].items(), columns=["Feature", "Importance"])
    a.plotly_chart(px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                          title="Feature importance"), use_container_width=True)
    preds = pd.read_csv(MODEL_DIR / "test_predictions.csv")
    fig = px.scatter(preds, x="actual", y="predicted", opacity=.4, title="Actual vs predicted AQI (test sample)")
    m = preds.max().max()
    fig.add_shape(type="line", x0=0, y0=0, x1=m, y1=m, line=dict(dash="dash", color="grey"))
    b.plotly_chart(fig, use_container_width=True)
    st.caption(f"Random Forest · {metrics['n_train']:,} train / {metrics['n_test']:,} test rows · 80/20 split")
