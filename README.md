# Lahore AQI Forecast

Machine-learning forecast of Lahore's US-EPA AQI 1 to 3 days ahead, with a health-risk label and advice by group.

**Data:** daily PM2.5 from OpenAQ (one Lahore station) and daily weather from Open-Meteo, Dec 2022 to Mar 2025 (about 790 days, 120 gaps filled by interpolation).
**Method:** PM2.5 converted to AQI (US EPA), lag and rolling-average features, Linear Regression predicting the change in AQI, time-based 80/20 split, compared with a "tomorrow = today" baseline.
**Findings:** the model beats the baseline at 2 and 3 days ahead, not at 1 day; weather features did not help; Random Forest did worse than Linear Regression on this small dataset.
**Limitations:** one station, short history, modest gains. Not official health guidance.

Files: `app.py` (Streamlit app), `aqi_models.pkl` (trained models), `lahore_clean.csv` (dataset), `requirements.txt`.
