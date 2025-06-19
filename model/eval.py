# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load data
iris_data = pd.read_csv("dataset/Iris.csv")

# Features and target
X = iris_data.drop(columns=["Id", "Species"])
Y = iris_data["Species"]

# Split data
_, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Load model
model = joblib.load("model/iris_model.joblib")

# Prediction
Y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model accuracy on test data: {accuracy:.2%}")
