import joblib
import pandas as pd
import streamlit as st
from datetime import date

st.set_page_config(page_title="Lahore AQI Forecast", page_icon="🌫️")

FEATURES = ["aqi", "aqi_lag1", "aqi_lag2", "aqi_roll3", "aqi_roll7", "is_smog_season"]


@st.cache_resource
def load_models():
    return joblib.load("aqi_models.pkl")


def category(a):
    if a <= 100:
        return "Good/Moderate"
    elif a <= 150:
        return "Unhealthy for Sensitive"
    elif a <= 200:
        return "Unhealthy"
    return "Very Unhealthy/Hazardous"


ADVICE = {
    "Good/Moderate": "Air is acceptable for most people.",
    "Unhealthy for Sensitive": "Children, older adults and people with asthma or heart/lung conditions should limit long outdoor activity.",
    "Unhealthy": "Everyone should reduce prolonged outdoor exertion. Sensitive groups should stay indoors where possible.",
    "Very Unhealthy/Hazardous": "Avoid outdoor activity. A mask (N95/KN95) is advisable if you must go out.",
}

st.title("🌫️ Lahore AQI Forecast")
st.write(
    "A machine-learning model (Linear Regression) forecasts Lahore's US-EPA AQI "
    "for the next 1 to 3 days from the last 7 days of AQI readings."
)

models = load_models()

st.subheader("1. Enter the last 7 days of AQI")
st.caption("Use US AQI for Lahore (for example from IQAir or AQI.in). Day 0 is today.")

cols = st.columns(7)
defaults = [180, 175, 190, 170, 165, 185, 200]
values = []
for i, col in enumerate(cols):
    label = "Today" if i == 0 else f"-{i}d"
    values.append(col.number_input(label, min_value=0, max_value=500, value=defaults[i], key=f"d{i}"))

today = st.date_input("Today's date", value=date.today())

if st.button("Forecast"):
    aqi0, aqi1, aqi2 = values[0], values[1], values[2]
    row = pd.DataFrame([{
        "aqi": aqi0,
        "aqi_lag1": aqi1,
        "aqi_lag2": aqi2,
        "aqi_roll3": sum(values[:3]) / 3,
        "aqi_roll7": sum(values[:7]) / 7,
        "is_smog_season": int(today.month in [10, 11, 12, 1, 2]),
    }])[FEATURES]

    st.subheader("2. Model forecast")
    out = st.columns(3)
    for h, col in zip([1, 2, 3], out):
        scaler, model = models[h]
        pred = aqi0 + model.predict(scaler.transform(row))[0]
        pred = int(round(min(max(pred, 0), 500)))
        col.metric(f"In {h} day{'s' if h > 1 else ''}", pred)
        col.write(category(pred))

    worst = max(
        aqi0 + models[h][1].predict(models[h][0].transform(row))[0] for h in [1, 2, 3]
    )
    st.info(ADVICE[category(worst)])

st.divider()
st.caption(
    "Educational project. Trained on daily PM2.5 from one monitoring station in Lahore "
    "(Dec 2022 to Mar 2025) converted to AQI with the US EPA formula. "
    "Forecasts are estimates, not official health guidance."
)
