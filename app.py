import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("dataset.csv")

# Preprocessing
df = df.drop('favorite_AI_tool', axis=1)
df = pd.get_dummies(df, drop_first=True)

X = df.drop("GPA", axis=1)
y = df["GPA"]

# Train model
model = LinearRegression()
model.fit(X, y)

# UI
st.title("🎓 Student GPA Predictor")

study_hours = st.slider("Study Hours per Day", 0, 12, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
sleep = st.slider("Sleep Hours", 0, 12, 7)

# Create input
input_dict = {col: 0 for col in X.columns}

if 'study_hours_per_day' in input_dict:
    input_dict['study_hours_per_day'] = study_hours

if 'class_attendance_percent' in input_dict:
    input_dict['class_attendance_percent'] = attendance

if 'sleep_hours' in input_dict:
    input_dict['sleep_hours'] = sleep

input_df = pd.DataFrame([input_dict])

# Predict
if st.button("Predict GPA"):
    prediction = model.predict(input_df)
    st.success(f"🎯 Predicted GPA: {prediction[0]:.2f}")