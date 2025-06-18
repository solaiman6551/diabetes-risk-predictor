import streamlit as st
import pickle
import numpy as np

# Load trained model
with open('notebooks/diabetes_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Diabetes Risk Predictor")

# Input fields — customize based on dataset features
pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=100)
blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=150, value=70)
skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
insulin = st.number_input('Insulin Level', min_value=0, max_value=900, value=80)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input('Age', min_value=1, max_value=120, value=30)

# When user clicks the Predict button
if st.button('Predict'):
    # Prepare input data as numpy array
    input_data = np.array(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

    # Predict
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0][1]  # Probability of diabetes class

    if prediction[0] == 1:
        st.error(f"High risk of diabetes. Probability: {proba:.2f}")
    else:
        st.success(f"Low risk of diabetes. Probability: {proba:.2f}")
