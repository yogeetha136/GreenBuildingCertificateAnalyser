import streamlit as st
import xgboost as xgb
import joblib
import pandas as pd
import numpy as np

# Load the saved model and encoders
model_path = "xgboost_green_certified_full_model.pkl"
encoders_path = "label_encoders.pkl"

bst = joblib.load(model_path)
label_encoders = joblib.load(encoders_path)

# Streamlit App
st.title("Building Green Certification Prediction")

# Define default input values
default_inputs = {
    "Area": "Banashankari",
    "Building_Type": "Institutional",
    "Construction_Year": 1999,
    "Number_of_Floors": 9,
    "Energy_Consumption_Per_SqM": 50.0,
    "Water_Usage_Per_Building": 412.8938923,
    "Waste_Recycled_Percentage": 27.1606493,
    "Occupancy_Rate": 75.17365813,
    "Indoor_Air_Quality": 60.44461195,
    "Smart_Devices_Count": 3,
    "Maintenance_Resolution_Time": 21.40687977,
    "Maintenance_Priority": "High",
    "Energy_Per_SqM": 107.5077815,
    "Number_of_Residents": 430,
    "Electricity_Bill": 18353.59161,
    "Last_Inspection_Date_Timestamp": 1648425600,
}

# Create form for user input
with st.form("input_form"):
    st.subheader("Input Parameters")
    inputs = {}
    
    for key, default_value in default_inputs.items():
        if key in label_encoders:  # Handle categorical inputs
            options = list(label_encoders[key].classes_)
            if default_value in options:
                inputs[key] = st.selectbox(key, options, index=options.index(default_value))
            else:
                inputs[key] = st.selectbox(key, options)  # Default to the first available option
        elif key == "Last_Inspection_Date_Timestamp":
            last_inspection_date = st.date_input("Last Inspection Date", value=pd.Timestamp(default_value, unit='s'))
            inputs[key] = int(pd.Timestamp(last_inspection_date).timestamp())
        else:  # Handle numeric inputs
            inputs[key] = st.number_input(key, value=default_value, format="%.4f" if isinstance(default_value, float) else "%d")
    
    submit_button = st.form_submit_button("Predict")

# Prediction logic
if submit_button:
    try:
        # Prepare the input data
        input_data = {k: [v] for k, v in inputs.items()}
        input_df = pd.DataFrame(input_data)

        # Encode categorical values
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = encoder.transform(input_df[col])
                except ValueError:  # Handle unseen labels silently
                    input_df[col] = encoder.transform([encoder.classes_[0]])[0]  # Default to the first class

        # Prepare the DMatrix
        dmatrix = xgb.DMatrix(input_df)

        # Make prediction
        prediction = (bst.predict(dmatrix) > 0.5).astype(int)
        certification = "Certified! Congratulations you can move forward to apply for\n GRIHA certification" if prediction[0] == 1 else "Not Certified \n Please improve the sustainability features of your building with respect to \nenergy, water, waste, air managements to apply for GRIHA certification"

        # Display the result
        st.subheader("Prediction Result")
        st.write(f"Prediction: *{certification}*")

    except Exception:
        pass  # Ignore all exceptions silently