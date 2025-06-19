# Imports
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the csv file
iris_data = pd.read_csv("dataset/Iris.csv")

# Features and target
X = iris_data.drop(columns=["Id", "Species"])
Y = iris_data["Species"]

# Split data
X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

# Save model
joblib.dump(model, "model/iris_model.joblib")
