import streamlit as st
import pandas as pd
from joblib import load

# Custom styles for the app
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #0056b3;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and description
st.title("🚗 Insurance Claim Prediction")
st.write("### Predict the likelihood of an insurance claim using vehicle and policyholder data.")
st.write("Provide the required information below to get started.")

# Load model, encoders, and scaler
try:
    model = load('naive_bayes_model.pkl')
    encoder = load('label_encoders.pkl')
    scaler = load('scaler.pkl')
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading files: {e}")
    st.stop()

# Define expected features
CATEGORICAL_FEATURES = [
    'Blind_Make', 'Blind_Model', 'Blind_Submodel',
    'Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5', 'Cat6', 'Cat7', 'Cat8', 'Cat9',
    'Cat10', 'Cat11', 'Cat12', 'OrdCat', 'NVCat'
]

CONTINUOUS_FEATURES = [
    'Row_ID', 'Household_ID', 'Vehicle', 'Calendar_Year', 'Model_Year',
    'Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Var6', 'Var7', 'Var8',
    'NVVar1', 'NVVar2', 'NVVar3', 'NVVar4'
]

# Function to collect user input dynamically
def get_user_input():
    """Collects categorical and continuous inputs from the user."""
    st.sidebar.header("Input User's Vehicle Features")

    # Collect categorical inputs
    categorical_input = {}
    with st.sidebar.expander("Categorical Features", expanded=True):
        for col in CATEGORICAL_FEATURES:
            if col in encoder:
                options = list(encoder[col].classes_)
                categorical_input[col] = st.selectbox(f"Select {col}", options)
            else:
                st.error(f"No encoder found for categorical feature: {col}")
                st.stop()

    # Collect continuous inputs
    continuous_input = {}
    with st.sidebar.expander("Continuous Features", expanded=True):
        for col in CONTINUOUS_FEATURES:
            continuous_input[col] = st.number_input(f"Enter value for {col}", value=0.00)

    return categorical_input, continuous_input

# Preprocess user input and make prediction
def preprocess_and_predict(categorical_input, continuous_input):
    """Preprocesses the inputs and makes a prediction."""
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
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"Prediction: 🚨 Claim Likely with a probability of {probability:.2f}")
    else:
        st.info(f"Prediction: ✅ Claim Unlikely with a probability of {1 - probability:.2f}")

# Collect user input
categorical_input, continuous_input = get_user_input()

# Make prediction on button click
if st.button("Predict 🚀"):
    preprocess_and_predict(categorical_input, continuous_input)
