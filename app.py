import streamlit as st
import pandas as pd
import pickle

# Load model (FAST)
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

# Load dataset ONLY for columns
df = pd.read_csv("dataset.csv")

if 'favorite_AI_tool' in df.columns:
    df = df.drop('favorite_AI_tool', axis=1)

df = pd.get_dummies(df, drop_first=True)
columns = df.drop("GPA", axis=1).columns

# UI
st.title("🎓 Student GPA Predictor")

study_hours = st.slider("Study Hours per Day", 0, 12, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
sleep = st.slider("Sleep Hours", 0, 12, 7)

# Input
input_dict = {col: 0 for col in columns}

if 'study_hours_per_day' in input_dict:
    input_dict['study_hours_per_day'] = study_hours

if 'class_attendance_percent' in input_dict:
    input_dict['class_attendance_percent'] = attendance

if 'sleep_hours' in input_dict:
    input_dict['sleep_hours'] = sleep

input_df = pd.DataFrame([input_dict])

# Prediction
if st.button("Predict GPA"):
    prediction = model.predict(input_df)
    st.success(f"🎯 Predicted GPA: {prediction[0]:.2f}")