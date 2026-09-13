import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Student Risk Prediction", page_icon="🎓")

st.title("🎓 Student Risk Prediction")
st.write("Early identification of students who may be at academic risk.")

# Load model and dataset
model = load_model("student_risk_model.keras")
df = pd.read_csv("dataset.csv", sep=";")

features = df.drop("Target", axis=1).columns

st.subheader("Enter Student Details")

values = []

for feature in features:
    value = st.number_input(
        feature,
        value=float(df[feature].median())
    )
    values.append(value)

if st.button("Predict Risk"):
    scaler = StandardScaler()
    scaler.fit(df[features])

    input_data = np.array(values).reshape(1, -1)
    input_data = scaler.transform(input_data)
    input_data = input_data.reshape(1, 1, len(features))

    probability = model.predict(input_data, verbose=0)[0][0]

    if probability >= 0.5:
        st.error(f"⚠️ At-Risk Student — Probability: {probability:.2%}")
    else:
        st.success(f"✅ Safe Student — Probability: {(1-probability):.2%}")