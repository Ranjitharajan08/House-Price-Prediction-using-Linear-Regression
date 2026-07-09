import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load trained model
model = joblib.load(Path("models") / "linear_regression_model.pkl")

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction using Linear Regression")

st.write("Enter the house details below.")

MedInc = st.number_input("Median Income", value=5.0)
HouseAge = st.number_input("House Age", value=20.0)
AveRooms = st.number_input("Average Rooms", value=5.0)
AveBedrms = st.number_input("Average Bedrooms", value=1.0)
Population = st.number_input("Population", value=1000.0)
AveOccup = st.number_input("Average Occupancy", value=3.0)
Latitude = st.number_input("Latitude", value=37.0)
Longitude = st.number_input("Longitude", value=-122.0)

if st.button("Predict House Price"):

    new_house = pd.DataFrame({
        "MedInc": [MedInc],
        "HouseAge": [HouseAge],
        "AveRooms": [AveRooms],
        "AveBedrms": [AveBedrms],
        "Population": [Population],
        "AveOccup": [AveOccup],
        "Latitude": [Latitude],
        "Longitude": [Longitude]
    })

    prediction = model.predict(new_house)

    st.success(f"🏠 Predicted House Value: {prediction[0]:.3f}")