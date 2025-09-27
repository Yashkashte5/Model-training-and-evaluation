import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline (includes preprocessing + model)
pipeline = joblib.load("titanic_pipeline.pkl")

st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger details to predict survival:")

# --- User Inputs ---
pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", 0, 100, 30)
sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Parents/Children aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Prepare raw input DataFrame (same columns as training data)
input_df = pd.DataFrame([{
    "pclass": pclass,
    "sex": sex,
    "age": age,
    "sibsp": sibsp,
    "parch": parch,
    "fare": fare,
    "embarked": embarked
}])

# --- Prediction ---
if st.button("Predict Survival"):
    try:
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success(f"Prediction: Survived ✅ (Probability: {probability:.2f})")
        else:
            st.error(f"Prediction: Did not survive ❌ (Probability: {1 - probability:.2f})")
    except Exception as e:
        st.error(f"Error: {e}")
