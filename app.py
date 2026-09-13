import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Student Risk Prediction",
    page_icon="🎓"
)

st.title("🎓 Student Risk Prediction")
st.write("Early identification of students who may be at academic risk.")

# Load dataset
df = pd.read_csv("dataset.csv", sep=";")

# Prepare data
X = df.drop("Target", axis=1)

y = df["Target"].map({
    "Dropout": 1,
    "Graduate": 0,
    "Enrolled": 0
})

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Student details
st.subheader("Enter Student Details")

values = []

for feature in X.columns:
    value = st.number_input(
        feature,
        value=float(df[feature].median())
    )
    values.append(value)

# Prediction
if st.button("Predict Risk"):

    input_data = scaler.transform([values])

    probability = model.predict_proba(input_data)[0][1]

    safe_probability = 1 - probability
    at_risk_probability = probability

    st.subheader("📊 Prediction Result")

    st.success(
        f"✅ Safe Student — Probability: {safe_probability:.2%}"
    )

    st.error(
        f"⚠️ At-Risk Student — Probability: {at_risk_probability:.2%}"
    )

    if at_risk_probability >= 0.5:
        st.warning("⚠️ Overall Prediction: At-Risk Student")
    else:
        st.success("✅ Overall Prediction: Safe Student")
