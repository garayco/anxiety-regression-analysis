import streamlit as st
import pandas as pd
import pickle
import sys
import streamlit as st

st.title("Predictor de Nivel de Ansiedad (Escala 0 a 1)")


@st.cache_resource
def load_model(path="modelo_ansiedad.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model()

# Entradas del usuario
age = st.slider("Edad", 18, 100, 30)
sleep_hours = st.slider("Horas de sueño por día", 0.0, 12.0, 7.0)
activity = st.slider("Horas de actividad física por semana", 0.0, 20.0, 3.0)
social_support = st.slider("Puntaje de apoyo social (0-100)", 0, 100, 50)
financial_stress = st.slider("Estrés financiero (0-100)", 0, 100, 50)
work_stress = st.slider("Estrés laboral (0-100)", 0, 100, 50)
self_esteem = st.slider("Autoestima (0-100)", 0, 100, 50)
life_satisfaction = st.slider("Satisfacción con la vida (0-100)", 0, 100, 50)
loneliness = st.slider("Puntaje de soledad (0-100)", 0, 100, 50)

# DataFrame de entrada
X_new = pd.DataFrame([{
    "Age": age,
    "Sleep_Hours": sleep_hours,
    "Physical_Activity_Hrs": activity,
    "Social_Support_Score": social_support,
    "Financial_Stress": financial_stress,
    "Work_Stress": work_stress,
    "Self_Esteem_Score": self_esteem,
    "Life_Satisfaction_Score": life_satisfaction,
    "Loneliness_Score": loneliness
}])

# Predicción
pred = model.predict(X_new)[0]
st.metric("Predicción de nivel de ansiedad (0 a 1)", f"{pred:.2f}")

# python -m streamlit run app.py
