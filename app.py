
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

st.set_page_config(
    page_title="Student Risk Prediction",
    page_icon="🎓",
    layout="centered"
)

# Load dataset, model and scaler
df = pd.read_csv("dataset.csv", sep=";")
model = tf.keras.models.load_model("student_risk_model.keras")
scaler = joblib.load("scaler.pkl")

features = df.drop("Target", axis=1).columns.tolist()

st.title("🎓 Student Risk Prediction System")

st.write(
    "Predict whether a student is Safe or At-Risk "
    "based on academic information."
)

# Student Details
st.header("👤 Student Details")

name = st.text_input("Student Name")
register_no = st.text_input("Register Number")

department = st.selectbox(
    "Department",
    [
        "B.Sc Information Technology",
        "B.Sc Computer Science",
        "BCA",
        "B.Sc Data Science"
    ]
)

year = st.selectbox(
    "Year",
    ["I Year", "II Year", "III Year"]
)

semester = st.selectbox(
    "Semester",
    [
        "I Semester",
        "II Semester",
        "III Semester",
        "IV Semester",
        "V Semester",
        "VI Semester"
    ]
)

# Academic Information
st.header("📚 Academic Information")

selected_features = [
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (grade)"
]

values = {}

for feature in selected_features:
    if feature in df.columns:
        values[feature] = st.number_input(
            feature,
            min_value=0.0,
            value=float(df[feature].median())
        )

# Additional Information
st.header("📋 Additional Information")

additional_features = [
    "Tuition fees up to date",
    "Scholarship holder",
    "Debtor",
    "Displaced",
    "Educational special needs",
    "Age at enrollment"
]

for feature in additional_features:

    if feature in df.columns:

        if feature == "Age at enrollment":

            values[feature] = st.number_input(
                feature,
                min_value=15.0,
                max_value=100.0,
                value=float(df[feature].median())
            )

        else:

            answer = st.selectbox(
                feature,
                ["Yes", "No"]
            )

            values[feature] = 1 if answer == "Yes" else 0


# Prediction
if st.button("🔍 Predict Risk"):

    if name.strip() == "" or register_no.strip() == "":

        st.warning(
            "Please enter Student Name and Register Number."
        )

    else:

        # Create complete model input
        input_data = pd.DataFrame(
            [df[features].median().values],
            columns=features
        )

        # Replace selected values
        for feature, value in values.items():

            if feature in input_data.columns:
                input_data.loc[0, feature] = value

        # Scale input
        input_scaled = scaler.transform(input_data)

        # LSTM input shape
        input_lstm = input_scaled.reshape(
            1, 1, len(features)
        )

        # Prediction
        risk_probability = float(
            model.predict(
                input_lstm,
                verbose=0
            )[0][0]
        )

        safe_probability = 1 - risk_probability

        # Prediction Result
        st.header("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Safe Probability",
                f"{safe_probability:.2%}"
            )

        with col2:
            st.metric(
                "At-Risk Probability",
                f"{risk_probability:.2%}"
            )

        if risk_probability >= 0.5:

            st.error(
                "⚠️ Prediction: AT-RISK STUDENT"
            )

        else:

            st.success(
                "✅ Prediction: SAFE STUDENT"
            )


        # Academic Performance Graph
        st.subheader("📈 Academic Performance Graph")

        graph_data = pd.DataFrame({

            "Value": [
                values.get(
                    "Curricular units 1st sem (approved)", 0
                ),
                values.get(
                    "Curricular units 1st sem (enrolled)", 0
                ),
                values.get(
                    "Curricular units 1st sem (evaluations)", 0
                ),
                values.get(
                    "Curricular units 1st sem (grade)", 0
                ),
                values.get(
                    "Curricular units 2nd sem (approved)", 0
                ),
                values.get(
                    "Curricular units 2nd sem (enrolled)", 0
                ),
                values.get(
                    "Curricular units 2nd sem (evaluations)", 0
                ),
                values.get(
                    "Curricular units 2nd sem (grade)", 0
                )
            ]

        }, index=[

            "1st Approved",
            "1st Enrolled",
            "1st Evaluations",
            "1st Grade",
            "2nd Approved",
            "2nd Enrolled",
            "2nd Evaluations",
            "2nd Grade"

        ])

        st.line_chart(graph_data)


        # Student Information
        st.subheader("👤 Student Information")

        st.write(f"**Name:** {name}")
        st.write(f"**Register Number:** {register_no}")
        st.write(f"**Department:** {department}")
        st.write(f"**Year:** {year}")
        st.write(f"**Semester:** {semester}")


# Model Information
st.divider()

st.subheader("🤖 Model Information")

st.write(
    "**Model:** LSTM (Long Short-Term Memory)"
)

st.write(
    "**Test Accuracy:** 87.91%"
)

st.write(
    "**Dataset:** UCI Student Academic Success Dataset"
)

st.caption(
    "The system uses academic and student-related information "
    "to estimate the student's academic risk."
)
