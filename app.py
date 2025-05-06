import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load("best_model.pkl")

# Define the mapping for obesity levels
obesity_map = {
    0: 'Insufficient Weight',
    1: 'Normal Weight',
    2: 'Overweight Level I',
    3: 'Overweight Level II',
    4: 'Obesity Type I',
    5: 'Obesity Type II',
    6: 'Obesity Type III'
}

yes_no_options = {"No": "no", "Yes": "yes"}
snack_options = {
    "Never": "no",
    "Sometimes": "Sometimes",
    "Frequently": "Frequently",
    "Always": "Always"
}

transport_options = {
    "Public Transportation": "Public_Transportation",
    "Automobile": "Automobile",
    "Walking": "Walking",
    "Motorbike": "Motorbike",
    "Bike": "Bike"
}


# Streamlit interface
st.set_page_config(page_title="Obesity Level Prediction", page_icon=":guardsman:", layout="wide")
st.title("Obesity Level Prediction Application")

st.markdown("""
### 🧠 About the Model

This application predicts the level of obesity based on the data provided by the user. The model was trained using a dataset that includes various health and lifestyle-related features.

- ✅ **Model Type:** CatBoost Classifier
- 🔁 **Preprocessing:** Scikit-learn Pipeline with:
  - Ordinal Encoding (for categorical and binary features)
  - One-Hot Encoding (for multi-class categories)
  - Standard Scaling (for numerical inputs)
- 📊 **Features Used:**
  - Demographics: Gender, Age, Height, Weight
  - Lifestyle: Smoking, Alcohol, Physical Activity, Water Intake, Meal Habits
  - Family History and Transportation Mode
- 🎯 **Model Accuracy:** 97.70%
- 🧪 **Validation:** Stratified train-test split with 25% test size

> 💡 *Disclaimer: This prediction is for informational purposes only and not a substitute for professional medical advice.*

""", unsafe_allow_html=True)


st.markdown("Please fill in the following fields to get a prediction.")

# User inputs
st.subheader("User Information")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=14, max_value=61, value=25)
height = st.number_input("Height (in meters)", min_value=1.45, max_value=1.98, value=1.65, step=0.01)
weight = st.number_input("Weight (in kg)", min_value=39.0, max_value=173.0, value=70.0)
family_history_with_overweight = yes_no_options[st.selectbox("Is there a family history of overweight?", list(yes_no_options.keys()))]
FAVC = yes_no_options[st.selectbox("Do you often eat high-calorie foods?", list(yes_no_options.keys()))]
FCVC = st.slider("How often you consume vegetables? (1-3)", min_value=1.0, max_value=3.0, step=0.1)
NCP = st.slider("How many main meals do you eat per day?", min_value=1.0, max_value=4.0, step=0.1)
CAEC = snack_options[st.selectbox("How often do you snack between meals?", list(snack_options.keys()))]
SMOKE = yes_no_options[st.selectbox("Do you smoke?", list(yes_no_options.keys()))]
CH2O = st.slider("Daily water consumption: (1-3)", min_value=1.0, max_value=3.0, step=0.1)
SCC = yes_no_options[st.selectbox("Do you monitor your calorie intake?", list(yes_no_options.keys()))]
FAF = st.slider("Frequency of physical activity (0-3)", min_value=0.0, max_value=3.0, step=0.1)
TUE = st.slider("Daily use of technology (0-3 hours)", min_value=0.0, max_value=3.0, step=0.1)
CALC = snack_options[st.selectbox("Frequency of alcohol consumption", list(snack_options.keys()))]
MTRANS = transport_options[st.selectbox("Main mode of transportation:", list(transport_options.keys()))]


# Prediction button
st.subheader("Prediction")
if st.button("Predict Obesity Level"):
    # Transformation the user input into a DataFrame
    input_df = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "family_history_with_overweight": [family_history_with_overweight],
        "FAVC": [FAVC],
        "FCVC": [FCVC],
        "NCP": [NCP],
        "CAEC": [CAEC],
        "SMOKE": [SMOKE],
        "CH2O": [CH2O],
        "SCC": [SCC],
        "FAF": [FAF],
        "TUE": [TUE],
        "CALC": [CALC],
        "MTRANS": [MTRANS]
})


    required_columns = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']

    input_df = input_df[required_columns]


    # Model prediction
    prediction = model.predict(input_df)

    # Prediction will return an ndarray, so we take the first value
    prediction = prediction[0]

    # Map the prediction to the corresponding obesity level
    predicted_label = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction



    # Display the prediction result
    st.subheader("Prediction Result")
    st.success(f"Estimated level of obesity: **{predicted_label.replace('_', ' ')}**")

    st.warning(
    "⚠️  This application makes estimates based on the data entered. The results are for reference only and are not intended to make a definitive medical diagnosis. It is recommended that you consult a doctor to get precise information about your health problems."
    )
