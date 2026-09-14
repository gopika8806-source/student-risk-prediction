
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Page setup
st.set_page_config(
    page_title="Student Risk Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Risk Prediction System")
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

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Model performance
test_prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, test_prediction)
f1 = f1_score(y_test, test_prediction)

# ---------------- STUDENT DETAILS ----------------

st.header("👤 Student Details")

name = st.text_input("Student Name")
reg_no = st.text_input("Register Number")

department = st.selectbox(
    "Department",
    [
        "B.Sc Information Technology",
        "BCA",
        "B.Sc Computer Science",
        "Other"
    ]
)

year = st.selectbox(
    "Year",
    ["I Year", "II Year", "III Year"]
)

semester = st.selectbox(
    "Semester",
    [
        "Semester 1",
        "Semester 2",
        "Semester 3",
        "Semester 4",
        "Semester 5",
        "Semester 6"
    ]
)

# ---------------- ACADEMIC INFORMATION ----------------

st.header("📚 Academic Information")

academic_fields = [
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (grade)"
]

# Store user values
user_values = {}

for field in academic_fields:
    if field in X.columns:
        user_values[field] = st.number_input(
            field,
            value=float(df[field].median())
        )

# ---------------- ADDITIONAL INFORMATION ----------------

st.header("📊 Additional Information")

additional_fields = [
    "Tuition fees up to date",
    "Scholarship holder",
    "Debtor",
    "Displaced",
    "Educational special needs",
    "Age at enrollment"
]

for field in additional_fields:
    if field in X.columns:
        user_values[field] = st.number_input(
            field,
            value=float(df[field].median())
        )

# ---------------- PREDICTION ----------------

st.header("🎯 Prediction")

if st.button("Predict Risk", type="primary"):

    if name.strip() == "" or reg_no.strip() == "":
        st.warning("⚠️ Please enter Student Name and Register Number.")

    else:
        # Create input using dataset median values
        input_values = []

        for column in X.columns:

            if column in user_values:
                value = user_values[column]

            else:
                value = float(df[column].median())

            input_values.append(value)

        # Scale input
        input_data = scaler.transform([input_values])

        # Prediction probability
        probability = model.predict_proba(input_data)[0][1]

        safe_probability = 1 - probability
        at_risk_probability = probability

        # Result
        st.subheader("📌 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.success(
                f"✅ Safe Probability\n\n"
                f"{safe_probability:.2%}"
            )

        with col2:
            st.error(
                f"⚠️ At-Risk Probability\n\n"
                f"{at_risk_probability:.2%}"
            )

        if probability >= 0.5:
            st.error("⚠️ Overall Prediction: At-Risk Student")
        else:
            st.success("✅ Overall Prediction: Safe Student")

        # Student information
        st.info(
            f"👤 Student: {name}\n\n"
            f"🆔 Register Number: {reg_no}\n\n"
            f"🏫 Department: {department}\n\n"
            f"📅 Year: {year}\n\n"
            f"📖 Semester: {semester}"
        )

# ---------------- MODEL INFORMATION ----------------

st.divider()

st.subheader("🤖 Model Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Model", "Logistic Regression")

with col2:
    st.metric("Accuracy", f"{accuracy:.2%}")

st.caption(
    "The system uses academic and student-related information "
    "to estimate the student's academic risk."
)
