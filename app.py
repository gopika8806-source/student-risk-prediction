import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Student Risk Prediction", page_icon="🎓")

st.title("🎓 Student Risk Prediction")
st.write("Early identification of students who may be at academic risk.")

df = pd.read_csv("dataset.csv", sep=";")

X = df.drop("Target", axis=1)
y = df["Target"].map({"Dropout": 1, "Graduate": 0, "Enrolled": 0})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

st.subheader("Enter Student Details")

values = []
for feature in X.columns:
    values.append(st.number_input(feature, value=float(df[feature].median())))

if st.button("Predict Risk"):
    input_data = scaler.transform([values])
    probability = model.predict_proba(input_data)[0][1]

    if probability >= 0.5:
        st.error(f"⚠️ At-Risk Student — Probability: {probability:.2%}")
    else:
        st.success(f"✅ Safe Student — Probability: {(1-probability):.2%}")
