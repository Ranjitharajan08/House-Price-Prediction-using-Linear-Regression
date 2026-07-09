import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from pathlib import Path
# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(df.head())
# Features (Input)
X = df.drop("MedHouseVal", axis=1)

# Target (Output)
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)
print("Training Data:", X_train.shape)
print("Testing Data :", X_test.shape)
# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("✅ Model trained successfully!")
# Predict on test data
y_pred = model.predict(X_test)

print("\nFirst 5 Predictions:")
print(y_pred[:5])
# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("-" * 35)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
# Create models directory if it doesn't exist
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# Save the trained model
joblib.dump(model, models_dir / "linear_regression_model.pkl")

print("\n✅ Model saved successfully in models/linear_regression_model.pkl")
# Plot Actual vs Predicted values
plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)
plt.plot([0, 5], [0, 5], 'r--', linewidth=2, label="Perfect Prediction")
plt.legend()
plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")

plt.title("Actual vs Predicted House Prices")

plt.savefig("images/actual_vs_predicted.png")

plt.show()