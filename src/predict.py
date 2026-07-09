import joblib
import pandas as pd
from pathlib import Path

# Load the trained model
model_path = Path("models") / "linear_regression_model.pkl"
model = joblib.load(model_path)

print("\n===== House Price Prediction =====\n")

MedInc = float(input("Median Income: "))
HouseAge = float(input("House Age: "))
AveRooms = float(input("Average Rooms: "))
AveBedrms = float(input("Average Bedrooms: "))
Population = float(input("Population: "))
AveOccup = float(input("Average Occupancy: "))
Latitude = float(input("Latitude: "))
Longitude = float(input("Longitude: "))

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

print("\n🏠 Predicted House Value:", round(prediction[0], 3))