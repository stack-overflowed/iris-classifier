# Imports
from flask import Flask, render_template, request
import joblib
import numpy as np

# Make an app
app = Flask(__name__)

# Load the model
model = joblib.load("model/iris_model.joblib")

# Get the inputs and display prediction
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Get form values
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])

            # Create input array for prediction
            features = np.array(
                [[sepal_length, sepal_width, petal_length, petal_width]])

            # Make prediction
            prediction = model.predict(features)[0]
            return render_template("index.html", prediction=prediction)
        except Exception as e:
            return f"Error: {str(e)}"
    return render_template("index.html", prediction=None)


if __name__ == "__main__":
    app.run(debug=True)
