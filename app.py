import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd


model=load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('label_encoder_geo.pkl','rb') as file:
    label_encoder_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})
label_encoded_geo=label_encoder_geo.transform([input_data['Geography']])
geo_encoded_df=pd.DataFrame(label_encoded_geo,columns=['Geography_France', 'Geography_Germany', 'Geography_Spain'])

input_df=pd.concat([input_data.drop('Geography',axis=1),geo_encoded_df],axis=1)

input_df=scaler.transform(input_df)

prediction=model.predict(input_df)[0][0]

st.write(f"Churn Probabilty is {prediction}")
if prediction>=0.5:
    st.write('The Customer is likely to churn')
else:
    st.write("The Customer is not gonna churn")
