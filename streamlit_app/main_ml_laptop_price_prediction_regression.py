import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from joblib import load

# pipline = load("artifact/pipeline_rdf.joblib")
pipline = load("artifact/pipeline.joblib")
df = load("artifact/dataframe.joblib")

# Filter unique values for the dropdown menus
companies = df['company'].unique()
cpu_names = df['cpu_name'].unique()
gpu_names = df['gpu_name'].unique()
rams = df['ram'].unique()
new_types = df['new_type_name'].unique()  # 'new_type_name' will be filtered based on company
ppis = df['ppi'].unique()
ipspanels = df['ipspanel'].unique()
primarystoragetypes = df['primarystoragetype'].unique()
secondarystorage = df['secondarystorage'].unique()

# Define the pipeline
# numerical_features = ['ram', 'secondarystorage', 'ppi']
# categorical_features = ['company', 'ipspanel', 'primarystoragetype', 'new_type_name', 'gpu_name', 'cpu_name']

# Streamlit UI
st.title("Laptop Price Prediction")

# User inputs
with st.form(key='input_form'):
    # Columns layout for selecting features
    col1, col2, col3 = st.columns(3)

    with col1:
        company = st.selectbox('Company', companies)

    with col2:
        cpu_name = st.selectbox('CPU Name', cpu_names)

    with col3:
        gpu_name = st.selectbox('GPU Name', gpu_names)

    with col1:
        ram = st.selectbox('RAM (GB)', rams)

    # Filter 'new_type_name' based on company selection
    # filtered_new_type = df[df['company'] == company]['new_type_name'].unique()
    with col2:
        new_type = st.selectbox('Computer Type', new_types)

    with col3:
        ppi = st.selectbox('PPI', ppis)

    with col1:
        ipspanel = st.selectbox('IPS Panel Type', ipspanels)

    with col2:
        primarystoragetype = st.selectbox('Primary Storage Type', primarystoragetypes)

    with col3:
        secondarystorage_value = st.selectbox('Secondary Storage (GB)', secondarystorage)

    # Submit button
    submit_button = st.form_submit_button(label='Predict Price')

# If the form is submitted
list_columns = ['company', 'ram', 'price_euros', 'ipspanel', 'secondarystorage',
       'primarystoragetype', 'new_type_name', 'ppi', 'gpu_name', 'cpu_name']
if submit_button:
    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'company': [company],
        'cpu_name': [cpu_name],
        'gpu_name': [gpu_name],
        'ram': [ram],
        'new_type_name': [new_type],
        'ppi': [ppi],
        'ipspanel': [ipspanel],
        'primarystoragetype': [primarystoragetype],
        'secondarystorage': [secondarystorage_value]
    })
    # st.write(input_data)
    # Predict the price
    prediction = pipline.predict(input_data)

    # Display the result
    st.success(f"The predicted laptop price is: ${prediction[0]:.2f}")
