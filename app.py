import streamlit as st
import pandas as pd
import joblib
import time

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
model = joblib.load("fraud_model.pkl")

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("💳 Credit Card Fraud Detection System")
st.write("Enter transaction details below")

# Inputs matching training features
row_id = st.number_input("Transaction ID (Unnamed: 0)", min_value=0, value=0)
cc_num = st.text_input("Credit Card Number")
amt = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")

zip_code = st.number_input("ZIP Code", min_value=0)
lat = st.number_input("Customer Latitude", format="%.6f")
long = st.number_input("Customer Longitude", format="%.6f")

city_pop = st.number_input("City Population", min_value=0)
unix_time = st.number_input(
    "Unix Time (leave default for now)",
    value=int(time.time())
)

merch_lat = st.number_input("Merchant Latitude", format="%.6f")
merch_long = st.number_input("Merchant Longitude", format="%.6f")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Check For Fraud"):

    if cc_num:

        # Hash credit card number (same idea as training)
        cc_num_hashed = hash(cc_num) % (10 ** 6)

        # Create input DataFrame EXACTLY like training
        input_data = pd.DataFrame([[
            row_id,
            cc_num_hashed,
            amt,
            zip_code,
            lat,
            long,
            city_pop,
            unix_time,
            merch_lat,
            merch_long
        ]], columns=[
            "Unnamed: 0",
            "cc_num",
            "amt",
            "zip",
            "lat",
            "long",
            "city_pop",
            "unix_time",
            "merch_lat",
            "merch_long"
        ])

        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"🚨 Fraudulent Transaction (Risk: {probability:.2%})")
        else:
            st.success(f"✅ Legitimate Transaction (Risk: {probability:.2%})")

    else:
        st.warning("⚠️ Please enter Credit Card Number")
