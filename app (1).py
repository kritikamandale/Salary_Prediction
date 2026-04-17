
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title='Salary Predictor App', layout='centered')

st.title('Predict Your Salary')

# Load the trained model
@st.cache_resource
def load_model():
    with open('random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# --- Load original data to fit LabelEncoders ---
# This is crucial for consistent encoding of categorical features
@st.cache_data
def load_original_data_for_encoders():
    try:
        raw_df = pd.read_csv('Salary_Data.csv') # Assuming Salary_Data.csv is available in the same directory
    except FileNotFoundError:
        st.error("Error: 'Salary_Data.csv' not found. Please ensure it's in the same directory as your app.")
        st.stop()
    
    # Impute missing values for categorical columns (mode) and numerical (mean) as done during training
    for col in raw_df.columns:
        if raw_df[col].dtype == 'object':
            raw_df[col] = raw_df[col].fillna(raw_df[col].mode()[0])
        else:
            raw_df[col] = raw_df[col].fillna(raw_df[col].mean())
    return raw_df

original_data = load_original_data_for_encoders()

# Create and fit LabelEncoders for each categorical column
# We fit them on the *imputed* original data to get all possible categories seen during training
def get_fitted_label_encoder(column_data):
    le = LabelEncoder()
    le.fit(column_data)
    return le

le_gender = get_fitted_label_encoder(original_data['Gender'])
le_education = get_fitted_label_encoder(original_data['Education Level'])
le_job_title = get_fitted_label_encoder(original_data['Job Title'])

# Input features from the user
st.sidebar.header('Input Your Details')

# Age
age = st.sidebar.slider('Age', min_value=18, max_value=65, value=30)

# Gender
gender_options = list(le_gender.classes_)
gender_input = st.sidebar.selectbox('Gender', gender_options)

# Education Level
education_options = list(le_education.classes_)
education_input = st.sidebar.selectbox('Education Level', education_options)

# Job Title
job_title_options = list(le_job_title.classes_)
job_title_input = st.sidebar.selectbox('Job Title', job_title_options)

# Years of Experience
years_of_experience = st.sidebar.slider('Years of Experience', min_value=0.0, max_value=40.0, value=5.0, step=0.5)


# Preprocess user input
def preprocess_input(age, gender, education, job_title, years_of_experience):
    # Create a DataFrame for the single input
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Education Level': education,
        'Job Title': job_title,
        'Years of Experience': years_of_experience
    }])

    # Apply Label Encoding using the fitted encoders
    input_data['Gender'] = le_gender.transform(input_data['Gender'])
    input_data['Education Level'] = le_education.transform(input_data['Education Level'])
    input_data['Job Title'] = le_job_title.transform(input_data['Job Title'])

    return input_data


# Prediction button
if st.button('Predict Salary'):
    processed_input = preprocess_input(age, gender_input, education_input, job_title_input, years_of_experience)
    prediction = model.predict(processed_input)
    st.success(f'Predicted Salary: ${prediction[0]:,.2f}')
