import streamlit as st
import pandas as pd
from joblib import load

# Load model, encoders, and scaler
try:
    model = load('naive_bayes_model.pkl')
    encoder = load('label_encoders.pkl')
    scaler = load('scaler.pkl')
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
except Exception as e:
    st.error(f"An error occurred while loading files: {e}")

# Input form for user to provide data
st.title("Insurance Claim Prediction")
st.write("Provide the required information to predict whether a claim will occur.")

# Define expected features
categorical_features = [
    'Blind_Make', 'Blind_Model', 'Blind_Submodel', 
    'Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5', 'Cat6', 'Cat7', 'Cat8', 'Cat9', 
    'Cat10', 'Cat11', 'Cat12', 'OrdCat', 'NVCat'
]

continuous_features = [
    'Row_ID', 'Household_ID', 'Vehicle', 'Calendar_Year', 'Model_Year', 
    'Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Var6', 'Var7', 'Var8', 
    'NVVar1', 'NVVar2', 'NVVar3', 'NVVar4'
]

# Function to collect user input dynamically
def get_user_input():
    categorical_input = {}
    for col in categorical_features:
        if col in encoder:
            options = list(encoder[col].classes_)
            categorical_input[col] = st.selectbox(f"Select {col}", options)
        else:
            st.error(f"No encoder found for categorical feature: {col}")
            st.stop()
    
    continuous_input = {}
    for col in continuous_features:
        continuous_input[col] = st.number_input(f"Enter value for {col}", value=0.00)
    
    return categorical_input, continuous_input

categorical_input, continuous_input = get_user_input()

# Preprocess input
if st.button("Predict"):
    # Encode categorical inputs
    for col, value in categorical_input.items():
        categorical_input[col] = encoder[col].transform([value])[0]
    
    # Combine inputs into a single dataframe
    input_data = pd.DataFrame({**categorical_input, **continuous_input}, index=[0])
    
    # Align input data with scaler's expected features
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # Scale continuous features
    input_data[scaler.feature_names_in_] = scaler.transform(input_data[scaler.feature_names_in_])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display results
    if prediction == 1:
        st.success(f"Prediction: Claim Likely ")
    else:
        st.info(f"Prediction: Claim Unlikely ")
