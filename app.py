import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("blood_donation_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Blood Donation Prediction App")

recency = st.number_input("Months since Last Donation")
frequency = st.number_input("Number of Donations")
monetary = st.number_input("Total Volume Donated")
time = st.number_input("Months since First Donation")

if st.button("Predict"):
    data = np.array([[recency, frequency, monetary, time]])
    data = scaler.transform(data)
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Donor will donate again")
    else:
        st.error("Donor will NOT donate again")