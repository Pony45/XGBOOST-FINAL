import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="XGBoost M&V", layout="wide")
st.title("🚀 XGBoost M&V Dashboard")
st.markdown("*AI-based Measurement & Verification using XGBoost*")

# Load model with pickle
@st.cache_resource
def load_model():
    # Try multiple paths
    paths = ['models/xgboost_model.pkl', 'xgboost_model.pkl']
    for path in paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                model = pickle.load(f)
            return model
    
    st.error("Model not found!")
    return None

model = load_model()
if model is None:
    st.stop()

st.success("✅ XGBoost loaded!")

# Features (hardcoded to avoid file reading)
FEATURES = [
    'temperature', 'humidity', 'hour', 'dayofweek', 'month',
    'floor_area', 'occupants', 'retrofit',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'is_weekend', 'temp_humidity', 'occ_per_area'
]

SCALING = 56

# Sidebar
with st.sidebar:
    st.header("📋 Building Parameters")
    temp = st.slider("Temperature (°C)", 22, 35, 28)
    humidity = st.slider("Humidity (%)", 60, 95, 80)
    hour = st.slider("Hour", 0, 23, 14)
    day = st.selectbox("Day", range(7), format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
    month = st.selectbox("Month", range(1,13), format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
    area = st.number_input("Floor Area (m²)", 50, 300, 120)
    occ = st.number_input("Occupants", 1, 8, 4)
    retro = st.selectbox("Retrofit", [0,1], format_func=lambda x: "✅ Yes" if x else "❌ No")

# Feature engineering
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
is_weekend = 1 if day >= 5 else 0
temp_hum = temp * humidity / 100
occ_area = occ / area

X = np.array([[
    temp, humidity, hour, day, month, area, occ, retro,
    hour_sin, hour_cos, month_sin, month_cos, is_weekend, temp_hum, occ_area
]])

if st.button("🔮 Predict", type="primary"):
    pred = model.predict(X)[0] / SCALING
    st.metric("⚡ Energy", f"{pred:.2f} kWh")
    
    if retro == 1:
        X_base = X.copy()
        X_base[0][7] = 0
        base = model.predict(X_base)[0] / SCALING
        savings = base - pred
        pct = (savings / base) * 100
        st.success(f"💰 Savings: {savings:.2f} kWh ({pct:.1f}%)")
        st.info(f"Monthly Savings: RM {savings * 24 * 30 * 0.52:.2f}")
        
        # Simple chart
        fig, ax = plt.subplots()
        ax.bar(['Baseline', 'Retrofitted'], [base, pred], color=['red', 'green'])
        ax.set_ylabel('kWh')
        st.pyplot(fig)
    else:
        X_retro = X.copy()
        X_retro[0][7] = 1
        retro_pred = model.predict(X_retro)[0] / SCALING
        potential = pred - retro_pred
        st.info(f"💡 If retrofitted: Save ~{potential:.2f} kWh")

st.markdown("---")
st.caption("🚀 XGBoost M&V | Thesis Project")
