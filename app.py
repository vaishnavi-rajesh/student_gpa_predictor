import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# -------------------------------
# CACHE MODEL (VERY IMPORTANT)
# -------------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("dataset.csv")

    # Drop unnecessary column
    if 'favorite_AI_tool' in df.columns:
        df = df.drop('favorite_AI_tool', axis=1)

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Features & target
    X = df.drop("GPA", axis=1)
    y = df["GPA"]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    return model, X.columns

# Load model + columns
model, columns = train_model()

# -------------------------------
# UI
# -------------------------------
st.title("🎓 Student GPA Predictor")

st.write("Adjust the values and click predict:")

study_hours = st.slider("Study Hours per Day", 0, 12, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
sleep = st.slider("Sleep Hours", 0, 12, 7)

# -------------------------------
# INPUT PREPARATION
# -------------------------------
input_dict = {col: 0 for col in columns}

if 'study_hours_per_day' in input_dict:
    input_dict['study_hours_per_day'] = study_hours

if 'class_attendance_percent' in input_dict:
    input_dict['class_attendance_percent'] = attendance

if 'sleep_hours' in input_dict:
    input_dict['sleep_hours'] = sleep

input_df = pd.DataFrame([input_dict])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict GPA"):
    try:
        prediction = model.predict(input_df)
        st.success(f"🎯 Predicted GPA: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")