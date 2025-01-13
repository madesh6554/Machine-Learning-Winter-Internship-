import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle 
# Load the trained model
model = joblib.load("house_price_model.pkl")

 # For loading the trained model

# Title and description
st.title("House Price Prediction")
st.write("Enter the house features below to predict its price.")

# Input fields
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1.0, max_value=10.0, step=0.5, value=2.5)
living_area = st.number_input("Living Area (sqft)", min_value=500, max_value=10000, value=2920)
lot_area = st.number_input("Lot Area (sqft)", min_value=500, max_value=50000, value=4000)
floors = st.number_input("Number of Floors", min_value=1.0, max_value=3.0, step=0.5, value=1.5)
waterfront = st.selectbox("Waterfront Present?", [0, 1])
views = st.slider("Number of Views", min_value=0, max_value=10, value=0)
condition = st.slider("Condition of the House (1-5)", min_value=1, max_value=5, value=3)
grade = st.slider("Grade of the House (1-13)", min_value=1, max_value=13, value=8)
house_area = st.number_input("Area of the House (excluding basement, sqft)", min_value=500, max_value=10000, value=1910)
basement_area = st.number_input("Area of the Basement (sqft)", min_value=0, max_value=5000, value=1010)
built_year = st.number_input("Year Built", min_value=1800, max_value=2025, value=1909)
renov_year = st.number_input("Year Renovated (0 if never)", min_value=0, max_value=2025, value=0)
postal_code = st.number_input("Postal Code", min_value=10000, max_value=999999, value=122004)
latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=52.8878)
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-114.470)
living_area_renov = st.number_input("Living Area Renovated (sqft)", min_value=500, max_value=10000, value=2470)
lot_area_renov = st.number_input("Lot Area Renovated (sqft)", min_value=500, max_value=50000, value=4000)
schools_nearby = st.slider("Number of Schools Nearby", min_value=0, max_value=20, value=2)
airport_distance = st.number_input("Distance from the Airport (miles)", min_value=1, max_value=100, value=51)

# Predict button
if st.button("Predict Price"):
    if 'model' in locals():
        # Prepare input data
        input_data = np.array([[
            bedrooms, bathrooms, living_area, lot_area, floors, waterfront, views,
            condition, grade, house_area, basement_area, built_year, renov_year,
            postal_code, latitude, longitude, living_area_renov, lot_area_renov,
            schools_nearby, airport_distance
        ]])

        # Make prediction
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted House Price: ${prediction:,.2f}")
    else:
        st.error("No model is loaded for prediction.")

