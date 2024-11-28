import streamlit as st
import joblib
import numpy as np
import os

# Dynamically construct the model path using os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
MODEL_PATH = os.path.join(BASE_DIR, 'src', 'model', 'gradient_boosting_model.joblib')

# Load the trained model
try:
    clf = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file not found at: {MODEL_PATH}. Please check the path and try again.")
    st.stop()

# Define the primary feature list for user input
primary_features = [
    'Product Code',
    'Quality',
    'Ambient T (C)',
    'Process T (C)',
    'Rotation Speed (rpm)',
    'Torque (Nm)',
    'Operation Time (min)',
]

# Title and description
st.title("Machine Status Prediction App")
st.write(
    "Input the parameters below to predict whether the machine is working or broken."
)

# Input fields for each feature
input_data = []
quality_options = ['L', 'M', 'H']  # Assuming Quality is categorical with 'L', 'M', 'H' levels
product_code_example = "Enter a string code like 'A123' (optional)."

for feature in primary_features:
    if feature == 'Quality':
        value = st.selectbox(f"Select {feature}:", options=quality_options)
        # Convert categorical Quality to numerical (L=0, M=1, H=2) for model input
        value = quality_options.index(value)
    elif feature == 'Product Code':
        value = st.text_input(f"Enter {feature}: ({product_code_example})")
        if not value:
            value = 0  # Assign a default value if empty
        else:
            # Example encoding: sum of ASCII values
            value = sum([ord(char) for char in value])
    else:
        value = st.number_input(f"Enter {feature}:", value=0.0)
    input_data.append(value)

# Extract relevant inputs for derived features
try:
    process_temp = input_data[primary_features.index('Process T (C)')]
    ambient_temp = input_data[primary_features.index('Ambient T (C)')]
    operation_time = input_data[primary_features.index('Operation Time (min)')]
    rotation_speed = input_data[primary_features.index('Rotation Speed (rpm)')]
    torque = input_data[primary_features.index('Torque (Nm)')]

    # Calculate derived features
    t_difference_squared = (process_temp - ambient_temp) ** 2
    operation_time_per_temp_increase = (
        operation_time / t_difference_squared if t_difference_squared != 0 else 0
    )
    horsepower = (torque * rotation_speed) / 5252

    # Append derived features to the input data
    input_data.extend([t_difference_squared, operation_time_per_temp_increase, horsepower])

    # Convert input data to a numpy array for prediction
    input_data = np.array(input_data).reshape(1, -1)

except Exception as e:
    st.error(f"Error while calculating derived features: {str(e)}")
    st.stop()

# Predict button
if st.button("Predict Machine Status"):
    try:
        prediction = clf.predict(input_data)[0]
        prediction_proba = clf.predict_proba(input_data)[0]

        # Display prediction and probability
        if prediction == 1:
            st.error("Prediction: The machine is Broken")
        else:
            st.success("Prediction: The machine is Working")

        st.write(f"Probability of Working: {prediction_proba[0]:.2f}")
        st.write(f"Probability of Broken: {prediction_proba[1]:.2f}")

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
