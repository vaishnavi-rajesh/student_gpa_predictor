import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("🎓 Student GPA Predictor")

st.write("Enter student details below:")

# User Inputs
study_hours = st.slider("Study Hours per Day", 0, 12, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
sleep = st.slider("Sleep Hours", 0, 12, 7)
social = st.slider("Social Media Hours", 0, 10, 2)
screen = st.slider("Screen Time (Hours)", 0, 12, 4)
gaming = st.slider("Gaming Hours", 0, 10, 1)

input_dict = {}

for col in model.feature_names_in_:
    input_dict[col] = 0

if 'study_hours_per_day' in input_dict:
    input_dict['study_hours_per_day'] = study_hours

if 'class_attendance_percent' in input_dict:
    input_dict['class_attendance_percent'] = attendance

if 'sleep_hours' in input_dict:
    input_dict['sleep_hours'] = sleep

if 'social_media_hours' in input_dict:
    input_dict['social_media_hours'] = social

if 'screen_time_hours' in input_dict:
    input_dict['screen_time_hours'] = screen

if 'gaming_hours' in input_dict:
    input_dict['gaming_hours'] = gaming

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Prediction
if st.button("Predict GPA"):
    prediction = model.predict(input_df)
    st.success(f"🎯 Predicted GPA: {prediction[0]:.2f}")