import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('/Users/obinwosu/Downloads/best_rf_model1.pkl', 'rb'))

# Define the Streamlit app
st.title('Predictive Model for Stapes Surgery Outcome')
st.write("a machine learning app for predicting postoperative airbone gap using only presurgical variables")

# Input features
st.sidebar.header("Enter your features")

feature1 = st.sidebar.number_input('Surgeon Experience (years)', min_value=0, max_value=50, value=0, step=1)
feature2 = st.sidebar.number_input('Preoperative Air Pure Tone Average', min_value=0, max_value=200, value=0, step=1)
feature3 = st.sidebar.number_input('Preoperative Bone Pure Tone Average', min_value=0, max_value=200, value=0, step=1)

# Calculate ABG_pre
abg_pre = feature2 - feature3

# Display the calculated ABG_pre
#st.sidebar.write(f'Calculated ABG_pre: {abg_pre}')

feature4 = st.sidebar.number_input('Preoperative Airbone Gap', min_value=0, max_value=100, value=abg_pre, step=1)
feature5 = st.sidebar.number_input('Age', min_value=18, max_value=100, value=18, step=1)

# Select 'Left' or 'Right' for laterality
selected_laterality = st.sidebar.radio('Select Laterality', ['Left', 'Right'])
laterality_left = np.uint8(1 if selected_laterality == 'Left' else 0)
laterality_right = np.uint8(1 if selected_laterality == 'Right' else 0)

# Create a radio button for race selection
selected_race = st.sidebar.radio('Select Race', ['White', 'Black', 'NativeAmerican', 'Asian', 'Hawaiian', 'Hispanic', 'Other'])
race_options = ['White', 'Black', 'NativeAmerican', 'Asian', 'Hawaiian', 'Hispanic', 'Other']
race_selection = [int(1 if selected_race == race else 0) for race in race_options]

# Create a radio button for gender selection
selected_gender = st.sidebar.radio('Select Gender', ['Male', 'Female'])
gender_selection = [int(1 if selected_gender == 'Male' else 0), int(1 if selected_gender == 'Female' else 0)]

# Combine all feature selections
input_data = np.array([int(feature1), int(feature2), int(feature3), int(feature4), int(feature5),
                       laterality_left, laterality_right] + race_selection + gender_selection, dtype=np.int64)

# Make predictions
if st.button('Predict'):
    try:
        prediction = model.predict(input_data.reshape(1, -1))

        if prediction[0] <= 10.9:
            st.write(f'Your predicted postoperative ABG is {prediction[0]:.0f}. This is considered an excellent result.')

        elif prediction[0] <= 20:
            st.write(f'Your predicted postoperative ABG is {prediction[0]:.0f}. This is considered a good result.')

        else:
            st.write(f'Your predicted postoperative ABG is {prediction[0]:.0f}.')

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")