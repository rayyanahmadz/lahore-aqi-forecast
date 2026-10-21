import joblib
import math
import requests
import altair as alt
import pandas as pd
import streamlit as st
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Lahore AQI Forecast", page_icon="🌫️", layout="wide")
F = ["aqi", "aqi_lag1", "aqi_lag2", "aqi_roll3", "aqi_roll7", "is_smog_season"]
CATS = {"Good/Moderate": ("#8BC34A", "#1b1b1b"), "Unhealthy for Sensitive": ("#FF9800", "#1b1b1b"),
        "Unhealthy": ("#E53935", "#ffffff"), "Very Unhealthy/Hazardous": ("#7E0023", "#ffffff")}
BASE = {"Good/Moderate": "Air is acceptable for most people.",
        "Unhealthy for Sensitive": "Sensitive groups should limit long outdoor activity.",
        "Unhealthy": "Everyone should reduce prolonged outdoor exertion.",
        "Very Unhealthy/Hazardous": "Avoid outdoor activity where possible."}
EXTRA = {  # (for Unhealthy for Sensitive, for Unhealthy and worse)
    "General public": ("Watch for coughing or eye irritation.", "Keep windows closed and limit time outside."),
    "Child / student": ("Keep outdoor games short.", "Skip outdoor sports; stay indoors at break."),
    "Elderly": ("Limit long walks outdoors.", "Stay indoors and avoid exertion."),
    "Asthma / heart / lung condition": ("Keep medication handy and limit outdoor time.", "Stay indoors and contact your doctor if symptoms worsen."),
    "Outdoor worker / athlete": ("Take more breaks and shorten hard workouts.", "Train indoors; wear an N95/KN95 mask if you must work outside."),
}

def cat(a):
    return "Good/Moderate" if a <= 100 else "Unhealthy for Sensitive" if a <= 150 else "Unhealthy" if a <= 200 else "Very Unhealthy/Hazardous"

CALIB = 1.0  # live model PM2.5 is divided by this (set from the Colab check)

def pm25_to_aqi(c):
    c = math.floor(c * 10) / 10
    for lo, hi, ilo, ihi in [(0, 12, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200),
                             (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)]:
        if c <= hi:
            return round((ihi - ilo) / (hi - lo) * (c - lo) + ilo)
    return 500

@st.cache_data(ttl=3600)
def fetch_live():
    r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", timeout=15, params={
        "latitude": 31.5497, "longitude": 74.3436, "hourly": "pm2_5",
        "past_days": 7, "forecast_days": 1, "timezone": "Asia/Karachi"})
    r.raise_for_status()
    h = r.json()["hourly"]
    s = pd.Series(h["pm2_5"], index=pd.to_datetime(h["time"])).dropna()
    s = s[s.index <= pd.Timestamp.now(tz="Asia/Karachi").tz_localize(None)]
    daily = s.groupby(s.index.date).mean().tail(7)
    if len(daily) < 7:
        raise ValueError("not enough days")
    return [min(500, max(0, pm25_to_aqi(v / CALIB))) for v in daily.values[::-1]], str(daily.index[-1])

def load_live():
    try:
        v, last = fetch_live()
        for i in range(7):
            st.session_state[f"d{i}"] = int(v[i])
        st.session_state["src"] = f"Live PM2.5 for Lahore from Open-Meteo, converted to AQI. Latest day: {last}."
    except Exception:
        st.session_state["src"] = None

def refresh():
    fetch_live.clear()
    load_live()

@st.cache_resource
def models():
    return joblib.load("aqi_models.pkl")

@st.cache_data
def history():
    try:
        return pd.read_csv("lahore_clean.csv", index_col="date", parse_dates=True)
    except Exception:
        return None

def feats(d):
    d = d.copy()
    d["aqi_lag1"], d["aqi_lag2"] = d.aqi.shift(1), d.aqi.shift(2)
    d["aqi_roll3"], d["aqi_roll7"] = d.aqi.rolling(3).mean(), d.aqi.rolling(7).mean()
    d["is_smog_season"] = d.index.month.isin([10, 11, 12, 1, 2]).astype(int)
    return d

@st.cache_data
def backtest(h):  # same 80/20 time split used in training
    d = feats(history()); d["t"] = d.aqi.shift(-h); d = d.dropna(subset=F + ["t"])
    s = int(len(d) * 0.8)
    Xa, Xb, ya, yb = d[F].iloc[:s], d[F].iloc[s:], d.t.iloc[:s], d.t.iloc[s:]
    sc = StandardScaler().fit(Xa)
    m = LinearRegression().fit(sc.transform(Xa), ya - Xa.aqi)
    return yb, Xb.aqi + m.predict(sc.transform(Xb)), Xb.aqi

st.title("🌫️ Lahore AQI Forecast")
st.caption("Machine-learning forecast of Lahore's US-EPA AQI, 1 to 3 days ahead")
t1, t2, t3, t4 = st.tabs(["🔮 Forecast", "📊 Model evidence", "🗓️ Smog pattern", "ℹ️ About"])

with t1:
    if "src" not in st.session_state:
        for i, v in enumerate([180, 175, 190, 170, 165, 185, 200]):
            st.session_state[f"d{i}"] = v
        load_live()
    st.write("The last 7 days of AQI are fetched automatically. You can edit any value.")
    st.button("🔄 Refresh live data", on_click=refresh)
    if st.session_state["src"]:
        st.caption("✅ " + st.session_state["src"])
    else:
        st.warning("Live data is unavailable, so example values are shown. Please enter the last 7 days manually.")
    cols = st.columns(7)
    vals = [cols[i].number_input("Today" if i == 0 else f"-{i}d", 0, 500, key=f"d{i}") for i in range(7)]
    c1, c2 = st.columns(2)
    today = c1.date_input("Today's date", value=date.today())
    who = c2.selectbox("Who is this advice for?", list(EXTRA))
    if st.button("Forecast"):
        row = pd.DataFrame([{"aqi": vals[0], "aqi_lag1": vals[1], "aqi_lag2": vals[2],
                             "aqi_roll3": sum(vals[:3]) / 3, "aqi_roll7": sum(vals) / 7,
                             "is_smog_season": int(today.month in [10, 11, 12, 1, 2])}])[F]
        preds = []
        for h in (1, 2, 3):
            sc, m = models()[h]
            preds.append(int(round(min(max(vals[0] + m.predict(sc.transform(row))[0], 0), 500))))
        for col, h, p in zip(st.columns(3), (1, 2, 3), preds):
            bg, fg = CATS[cat(p)]
            col.markdown(f"<div style='background:{bg};color:{fg};padding:16px;border-radius:12px;text-align:center'>"
                         f"<div>In {h} day{'s' if h > 1 else ''}</div><div style='font-size:42px;font-weight:700'>{p}</div>"
                         f"<div>{cat(p)}</div></div>", unsafe_allow_html=True)
        worst = cat(max(preds)); msg = BASE[worst]
        if worst != "Good/Moderate":
            msg += " " + EXTRA[who][0 if worst == "Unhealthy for Sensitive" else 1]
        st.info(f"**{who}:** {msg}")
        a = pd.DataFrame({"date": pd.to_datetime([today - timedelta(days=i) for i in range(6, -1, -1)]), "AQI": vals[::-1], "kind": "Entered"})
        f = pd.DataFrame({"date": pd.to_datetime([today + timedelta(days=i) for i in range(4)]), "AQI": [vals[0]] + preds, "kind": "Forecast"})
        st.altair_chart(alt.Chart(pd.concat([a, f])).mark_line(point=True).encode(
            x="date:T", y="AQI:Q", color="kind:N", strokeDash="kind:N").properties(width="container", height=280))

with t2:
    if history() is None:
        st.warning("Upload lahore_clean.csv to the GitHub repository to enable this tab.")
    else:
        st.write("Tested on the **last 20% of days** the model never saw, against a lazy baseline: *tomorrow = today*.")
        rows = []
        for h in (1, 2, 3):
            y, p, n = backtest(h)
            rows.append({"Days ahead": h, "Baseline error (MAE)": round(mean_absolute_error(y, n), 1),
                         "My model error (MAE)": round(mean_absolute_error(y, p), 1),
                         "Category accuracy, baseline": round((n.apply(cat) == y.apply(cat)).mean(), 2),
                         "Category accuracy, model": round((p.apply(cat) == y.apply(cat)).mean(), 2)})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        h = st.radio("Show chart for forecast distance (days)", [1, 2, 3], index=1, horizontal=True)
        y, p, n = backtest(h)
        long = pd.concat([pd.DataFrame({"date": y.index, "AQI": y.values, "line": "Actual"}),
                          pd.DataFrame({"date": y.index, "AQI": p.values, "line": "My model"}),
                          pd.DataFrame({"date": y.index, "AQI": n.values, "line": "Baseline"})])
        st.altair_chart(alt.Chart(long).mark_line().encode(x="date:T", y="AQI:Q", color="line:N").properties(width="container", height=300))
        st.markdown("**Findings from my experiments**\n"
                    "- Past AQI does most of the work; the model beats the baseline at 2 and 3 days, but not at 1 day.\n"
                    "- Adding daily weather (temperature, humidity, wind) did **not** improve the forecast.\n"
                    "- A Random Forest did worse than Linear Regression, probably because there are only about 790 days of data.\n"
                    "- The test period is one winter, so these gaps are modest and may not hold in other years.")

with t3:
    if history() is None:
        st.warning("Upload lahore_clean.csv to the GitHub repository to enable this tab.")
    else:
        d = history().copy(); d["month"] = d.index.month
        mean = d.groupby("month").aqi.mean().round().reset_index()
        share = ((d.aqi > 150).groupby(d.month).mean() * 100).round().reset_index(name="pct")
        l, r = st.columns(2)
        l.write("**Average AQI by month**")
        l.altair_chart(alt.Chart(mean).mark_bar().encode(x="month:O", y="aqi:Q").properties(width="container", height=260))
        r.write("**% of days with AQI above 150 (Unhealthy)**")
        r.altair_chart(alt.Chart(share).mark_bar(color="#E53935").encode(x="month:O", y="pct:Q").properties(width="container", height=260))
        st.caption("Data covers Dec 2022 to Mar 2025 from one monitoring station.")

with t4:
    st.markdown("""
**Pipeline:** daily PM2.5 (OpenAQ, US Diplomatic Post station, Lahore) + daily weather (Open-Meteo) → fill 120 missing days →
convert PM2.5 to AQI with the US EPA formula → build features (today, 1 and 2 days ago, 3 and 7 day averages, smog-season flag) →
train Linear Regression to predict the *change* in AQI → time-based 80/20 test against a naive baseline → this app.

**Limitations:** one station, about 790 days, simple model, modest gains over the baseline, and 120 filled-in days.
Live inputs come from Open-Meteo's air-quality model for Lahore, not the training station, so they can differ from ground readings. Forecasts are estimates, not official health guidance.
""")
