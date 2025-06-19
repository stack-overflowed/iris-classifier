# Iris Species Classifier

A simple machine learning project that predicts the species of an Iris flower using a **Decision Tree Classifier**.


## About the Dataset

he dataset used is the classic [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set), which includes:
- 150 samples
- 3 species: *Iris-setosa*, *Iris-versicolor*, *Iris-virginica*
- 4 features: sepal length, sepal width, petal length, petal width

## How to Run
Make sure to have `Python 3.x` installed then <br>

Install dependencies <br>
`pip install -r requirements.txt`

Run the app
`python app.py`

Open browser at `http://127.0.0.1:5000` to see the app.

## Note
The model achieves 100% accuracy on the test set.
This is not unusual for the Iris dataset, which is small, clean, and well-separated.
Additionally, Decision Tree classifiers are capable of perfectly splitting such structured data.
However, this level of accuracy would likely not generalize to more complex, real-world datasets.