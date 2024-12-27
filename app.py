import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load the trained model
model = joblib.load("iris_model_D.pkl")

# Load Iris dataset details
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names

# Streamlit UI
st.title("Task - 1 ")
st.title("Iris Flower Prediction Application 🌸")
st.write("Predict the species of an Iris flower using its features!")

# Input sliders for features
sepal_length = st.slider("Sepal Length (cm)", float(iris.data[:, 0].min()), float(iris.data[:, 0].max()), 5.0)
sepal_width = st.slider("Sepal Width (cm)", float(iris.data[:, 1].min()), float(iris.data[:, 1].max()), 3.0)
petal_length = st.slider("Petal Length (cm)", float(iris.data[:, 2].min()), float(iris.data[:, 2].max()), 1.5)
petal_width = st.slider("Petal Width (cm)", float(iris.data[:, 3].min()), float(iris.data[:, 3].max()), 0.2)

# Predict button
if st.button("Predict"):
    # Prepare input features
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_features)
    predicted_class = target_names[prediction[0]]
    
    st.write(f"### Prediction: {predicted_class}")
    st.write("Model thinks this flower belongs to the -- **{}** -- species.".format(predicted_class))
