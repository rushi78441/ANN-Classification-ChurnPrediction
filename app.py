import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras import models

# Load trained model and preprocessor
model = models.load_model('churn_model.keras' , compile=False)
preproc = pickle.load(open('preprocessor.pkl', 'rb'))

st.title("Customer Churn Prediction")

def predict_churn(prob):
    return "The customer is likely to churn." if prob > 0.5 else "The customer is likely to stay."

st.header("Input Customer Data")

# Collect user input
input_data = {
    'CreditScore': st.number_input('Credit Score', min_value=300, max_value=850),
    'Geography': st.selectbox('Geography', ['France', 'Spain', 'Germany']),
    'Gender': st.selectbox('Gender', ['Male','Female']),
    'Age': st.number_input('Age', min_value=18, max_value=100),
    'Tenure': st.number_input('Tenure', min_value=0, max_value=10),
    'Balance': st.number_input('Balance', min_value=0),
    'NumOfProducts': st.number_input('Number of Products', min_value=1, max_value=4),
    'HasCrCard': 1 if st.selectbox('HasCrCard', ['No', 'Yes']) == 'Yes' else 0,
    'IsActiveMember': 1 if st.selectbox('Is Active Member', ['No', 'Yes']) == 'Yes' else 0,
    'EstimatedSalary': st.number_input('Estimated Salary', min_value=0)
}

if st.button("Predict Churn"):
    input_df = pd.DataFrame([input_data])
    processed = preproc.transform(input_df)
    prediction = model.predict(processed)
    prob = float(prediction[0][0])

    st.subheader("Prediction Result")
    st.write(predict_churn(prob))
    st.write(f"Churn Probability: {prob:.2f}")
