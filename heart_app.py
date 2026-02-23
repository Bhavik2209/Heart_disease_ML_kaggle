import streamlit as st
import dagshub
import mlflow
import pandas as pd
import sklearn

dagshub.init(
    repo_owner="Bhavik2209",
    repo_name="Heart_disease_ML_kaggle",
    mlflow=True
)

@st.cache_resource
def load_model():
    return mlflow.sklearn.load_model(
        "models:/Heart_Disease_XGB_Calibrated/1"
    )

model = load_model()

st.title("🫀 Heart Disease Prediction App")

st.write("Enter patient details below:")

# ------------------------------
# User Inputs
# ------------------------------

age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", [0, 1])  # 0 = Female, 1 = Male
chest_pain = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
bp = st.number_input("Resting Blood Pressure", value=130)
chol = st.number_input("Cholesterol", value=240)
fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
ekg = st.selectbox("EKG Results", [0, 1, 2])
max_hr = st.number_input("Maximum Heart Rate", value=150)
ex_angina = st.selectbox("Exercise Induced Angina", [0, 1])
st_dep = st.number_input("ST Depression", value=1.0)
slope = st.selectbox("Slope of ST", [1, 2, 3])
vessels = st.selectbox("Number of Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thallium", [3, 6, 7])

# ------------------------------
# Prediction Button
# ------------------------------

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "Chest pain type": chest_pain,
        "BP": bp,
        "Cholesterol": chol,
        "FBS over 120": fbs,
        "EKG results": ekg,
        "Max HR": max_hr,
        "Exercise angina": ex_angina,
        "ST depression": st_dep,
        "Slope of ST": slope,
        "Number of vessels fluro": vessels,
        "Thallium": thal
    }])

    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    st.write(f"Heart Disease Probability: **{prob}**")


    if prob < 0.4300:
        st.success("Low Risk")
    elif prob < 0.6500:
        st.warning("Medium Risk")


