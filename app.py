import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-container {
        background-color: #0E1117;
        border: 2px solid #1E88E5;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 25px;
    }
    .result-label {
        font-size: 18px;
        color: #E0E0E0;
    }
    .result-value {
        font-size: 32px;
        font-weight: bold;
        color: #00E676;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🚗 Car Price Prediction System</div>", unsafe_allow_html=True)
st.write("Enter the vehicle specifications below to estimate its market price.")

st.divider()

# Load Model Package
@st.cache_resource
def load_model_package():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'model.pkl')
    return joblib.load(model_path)

try:
    data_pkg = load_model_package()
    model = data_pkg.get('model')
    scaler = data_pkg.get('scaler')
    feature_columns = data_pkg.get('columns')
except Exception:
    data_pkg = None

st.subheader("Technical Specifications")

col1, col2 = st.columns(2)

with col1:
    horsepower = st.number_input("Horsepower", min_value=1, value=110)
    enginesize = st.number_input("Engine Size", min_value=1, value=130)
    curbweight = st.number_input("Curb Weight", min_value=1, value=2500)

with col2:
    highwaympg = st.number_input("Highway MPG", min_value=1, value=30)
    fueltype = st.selectbox("Fuel Type", options=["Gas", "Diesel"])
    carbody = st.selectbox("Car Body", options=["Sedan", "Hatchback", "Convertible", "Wagon"])

st.divider()

if st.button("Estimate Price", type="primary", use_container_width=True):
    if data_pkg is not None and feature_columns is not None:
        try:
            # 1. Start with average scaled inputs (using mean values like in Notebook)
            # If scaler has mean_, use it to populate baseline defaults
            if hasattr(scaler, 'mean_'):
                input_df = pd.DataFrame([scaler.mean_], columns=feature_columns)
            else:
                input_df = pd.DataFrame(0.0, index=[0], columns=feature_columns)

            # 2. Assign exact user values to numerical columns
            for col in input_df.columns:
                c_low = col.lower()
                if 'horsepower' in c_low:
                    input_df.loc[0, col] = float(horsepower)
                elif 'enginesize' in c_low or 'engine_size' in c_low:
                    input_df.loc[0, col] = float(enginesize)
                elif 'curbweight' in c_low or 'curb_weight' in c_low:
                    input_df.loc[0, col] = float(curbweight)
                elif 'highwaympg' in c_low or 'highway_mpg' in c_low:
                    input_df.loc[0, col] = float(highwaympg)

            # 3. Assign categorical values (resetting categorical dummies for fuel/body)
            for col in input_df.columns:
                c_low = col.lower()
                if 'fuel' in c_low:
                    input_df.loc[0, col] = 1.0 if fueltype.lower() in c_low else 0.0
                if 'body' in c_low:
                    input_df.loc[0, col] = 1.0 if carbody.lower() in c_low else 0.0

            # 4. Scale inputs and predict
            scaled_data = scaler.transform(input_df)
            predicted_price = model.predict(scaled_data)[0]

            # Display prediction result
            st.markdown(f"""
                <div class="result-container">
                    <div class="result-label">Estimated Price</div>
                    <div class="result-value">${predicted_price:,.2f}</div>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
    else:
        st.error("Model package ('model.pkl') not loaded properly.")