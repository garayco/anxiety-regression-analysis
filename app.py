import streamlit as st
import pandas as pd
import pickle
import streamlit as st

@st.cache_resource
def load_model(path="outputs/models/ols_anxiety_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("Anxiety Level Predictor")

# Inputs
stress_level = st.slider("Stress Level", 1, 10, 5)
sleep_hours = st.number_input("Sleep Hours per Night", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
therapy_sessions = st.number_input("Therapy Sessions per Month", min_value=0, max_value=12, value=2)
caffeine_intake = st.number_input("Caffeine Intake (mg/day)", min_value=0, max_value=600, value=200)
physical_activity = st.number_input("Physical Activity (hrs/week)", min_value=0.0, max_value=10.0, value=3.0, step=0.5)

X_new = pd.DataFrame([{
    "Stress Level (1-10)": stress_level,
    "Sleep Hours": sleep_hours,
    "Therapy Sessions (per month)": therapy_sessions,
    "Caffeine Intake (mg/day)": caffeine_intake,
    "Physical Activity (hrs/week)": physical_activity
}])

pred = model.predict(X_new)[0]
st.metric("Anxiety Level Prediction", f"{pred*100:.3f}%")

# python -m streamlit run app.py
