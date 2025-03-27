#Importing the necessary libraries
import streamlit as st
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE #SMOTE for oversampling unbalanced data 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings 
warnings.filterwarnings('ignore')
import joblib

st.set_page_config(page_title='Diabetes Predictor', layout='centered', initial_sidebar_state='auto')
st.markdown("<div style='background-color:#219C90; border-radius:50px; align-items:center; justify-content: center;'><h1 style='text-align:center; color:white;'>Diabetes Predictor</h1></div>",unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:black;'>Find out if you are diabetic based on your health features </h4>",unsafe_allow_html=True)

#Load the model and scaler 
best_model = joblib.load('ada_model.joblib')
scaler = joblib.load('scaler.joblib')



#Styling Streamlit Web App

col1, col2 = st.columns(2)

with col1:
  st.write("  ")
  st.write("  ")
  st.write("  ")
  st.write("  ")
  st.image("360_F_276205639_zXwXmtHSonG36a9pXiF2mYI6pBTIIMc8.jpg", use_column_width = True)

with col2:
  pregnancies = st.number_input(label = 'Amount of times user was pregnant',placeholder="Enter number of pregnancies",value=None,min_value=0,max_value=15,step=1)

  glucose = st.number_input(label = 'Enter your glucose level', placeholder="Enter your glucose level",value=None,min_value=0,max_value=200,step=1)

  

  col3, col4 = st.columns(2)
  with col3:
    blood_pressure = st.number_input(label = 'Enter your blood pressure',placeholder="Enter your blood pressure",value=None,min_value=0,max_value=125, step=1)
  with col4:
    skin_thickness = st.number_input(label = 'Enter your skin thickness',placeholder="Enter your skin thickness",value=None,min_value=0,max_value=100,step=1)
  col5, col6 = st.columns(2)
  with col5:
    insulin = st.number_input(label = 'Enter your insulin level',placeholder="Enter your insulin level",value=None,min_value=0,max_value=150,step=1)
  with col6:
    bmi = st.number_input(label = 'Enter your BMI(Body Mass Index)',placeholder="Enter your BMI(Body Mass Index)",value=None,min_value=0.0,max_value=99.0,step=0.1)
  col7, col8 = st.columns(2)
  with col7:
    diabetes_pedigree_function = st.number_input(label = 'Enter your diabetes pedigree function',placeholder="Enter your diabetes pedigree function",value=None,min_value=0.0,max_value=2.4,step=0.1)
  with col8:
    age = st.number_input(label = 'Enter your age',placeholder="Enter your age",value=None,min_value=0,max_value=100,step=1)
    

diagnosis = ['No Diabetes', 'Diabetes']
pred = st.button("Predict", use_container_width = True)
input_array = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]).reshape(1,-1)
scaled_input = scaler.transform(input_array)
if pred:
  prediction = best_model.predict(scaled_input)
  st.write(f"Based on the features you provided, you are diagnosed with: {diagnosis[prediction[0]]}")