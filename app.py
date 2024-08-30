##sckit LR trained tokenized TFIDF model

import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline model
pipeline = joblib.load('sckit_regression_model_token.joblib')

# Load the DataFrame (modify with the actual method to load your DataFrame)
df = pd.read_csv('merged.csv')  # Replace with the actual path to your data


# Define the Streamlit application
def main():
    st.title("University Major Salary Prediction App")

    # Create dropdowns for unique values in each column
    instnm_x_values = df['INSTNM_x'].unique()
    cipdesc_values = df['CIPDESC'].unique()
    
    instnm_x = st.selectbox("Select the University Name:", instnm_x_values)
    cipdesc = st.selectbox("Select the Major Description:", cipdesc_values)
    
    if st.button("Predict"):
        data_for_prediction = {'INSTNM_x': [instnm_x], 'CIPDESC': [cipdesc]}
        input_df = pd.DataFrame(data_for_prediction)
        
        predicted_value = pipeline.predict(input_df)
        
        st.write(f"Predicted YR 1 AVG: {predicted_value[0]}")
        
    st.write("Model Metrics")
    st.write("Root Mean Squared Error: 8341.01")  # replace with the actual value
    st.write("R-squared: 0.76")  # replace with the actual value

if __name__ == "__main__":
    main()