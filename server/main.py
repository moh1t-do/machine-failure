import streamlit as st
import joblib
import numpy as np

# Load the trained model
clf = joblib.load('src/model/gradient_boosting_model.joblib')

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
    "Input the parameters below to predict whether the machine is working or broken based on the Gradient Boosting model."
)

# Input fields for each feature
input_data = []
quality_options = ['L', 'M', 'H']  # Assuming Quality is categorical with 'L', 'M', 'H' levels

for feature in primary_features:
    if feature == 'Quality':
        value = st.selectbox(f"Select {feature}:", options=quality_options)
        # Convert categorical Quality to numerical (L=0, M=1, H=2) for model input
        value = quality_options.index(value)
    elif feature == 'Product Code':
        value = st.text_input(f"Enter {feature}:")
        # You can use an encoding for Product Code if needed for the model
        value = sum([ord(char) for char in value])  # Example encoding: sum of ASCII values
    else:
        value = st.number_input(f"Enter {feature}:", value=0.0)
    input_data.append(value)

# Extract relevant inputs for derived features
process_temp = input_data[primary_features.index('Process T (C)')]
ambient_temp = input_data[primary_features.index('Ambient T (C)')]
operation_time = input_data[primary_features.index('Operation Time (min)')]  # Changed variable to Operation Time
rotation_speed = input_data[primary_features.index('Rotation Speed (rpm)')]
torque = input_data[primary_features.index('Torque (Nm)')]

# Calculate derived features
t_difference_squared = (process_temp - ambient_temp) ** 2
operation_time_per_temp_increase = operation_time / t_difference_squared if t_difference_squared != 0 else 0  # Changed variable name
horsepower = (torque * rotation_speed) / 5252

# Append derived features to the input data
input_data.extend([t_difference_squared, operation_time_per_temp_increase, horsepower])

# Convert input data to a numpy array for prediction
input_data = np.array(input_data).reshape(1, -1)

# Predict button
if st.button("Predict Machine Status"):
    prediction = clf.predict(input_data)[0]
    prediction_proba = clf.predict_proba(input_data)[0]
    
    # Display prediction and probability
    if prediction == 1:
        st.error("Prediction: The machine is Broken")
    else:
        st.success("Prediction: The machine is Working")
        
    st.write(f"Probability of Working: {prediction_proba[0]:.2f}")
    st.write(f"Probability of Broken: {prediction_proba[1]:.2f}")
