import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
df = pd.read_csv("dataset.csv", sep=";")

# Convert target to binary
df["Target"] = df["Target"].apply(
    lambda x: 1 if x == "Dropout" else 0
)

# Separate input and target
X = df.drop("Target", axis=1)
y = df["Target"]

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape for LSTM
X = X.reshape(X.shape[0], 1, X.shape[1])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build lightweight LSTM
model = Sequential([
    LSTM(32, input_shape=(1, X.shape[2])),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)
model.save("student_risk_model.keras")