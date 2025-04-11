import streamlit as st
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("pcos_model.h5")
scaler = joblib.load("scaler.save")

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

st.title("🧠 PCOS Prediction App")

user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", step=1.0)
    user_input.append(value)

if st.button("Predict PCOS"):
    input_array = np.array([user_input])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0][0]

    if prediction > 0.5:
        st.error("⚠️ Likely PCOS")
    else:
        st.success("✅ Unlikely PCOS")
