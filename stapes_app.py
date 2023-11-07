import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('best_rf_model1.pkl', 'rb'))

# Define the Streamlit app
st.title('Predictive Model for Stapes Surgery Outcomes')
st.write("a machine learning app for predicting postoperative airbone gap using only presurgical variables")
st.markdown("<p style='font-size: 12px;'>This app is currently for research purposes only.</p>", unsafe_allow_html=True)

# Input features
st.header("Enter Preoperative Features")

feature1 = st.number_input('Surgeon Experience (years)', min_value=0, max_value=50, value=10, step=1)
feature2 = st.number_input('Preoperative Air Pure Tone Average', min_value=0, max_value=200, value=50, step=1)
feature3 = st.number_input('Preoperative Bone Pure Tone Average', min_value=0, max_value=200, value=30, step=1)

# Calculate ABG_pre
abg_pre = feature2 - feature3

# Display the calculated ABG_pre
#st.write(f'Calculated ABG_pre: {abg_pre}')

feature4 = st.number_input('Preoperative Airbone Gap', min_value=0, max_value=100, value=abg_pre, step=1)
feature5 = st.number_input('Age', min_value=18, max_value=100, value=18, step=1)

# Check if feature4 matches the calculated ABG_pre
if feature4 != abg_pre:
    st.error("Error: The value of 'Preoperative Airbone Gap' does not match the calculated value. It should be {abg_pre}. Please correct it to continue.")

# Select 'Left' or 'Right' for laterality
selected_laterality = st.radio('Select Laterality', ['Left', 'Right'])
laterality_left = np.uint8(1 if selected_laterality == 'Left' else 0)
laterality_right = np.uint8(1 if selected_laterality == 'Right' else 0)

# Create a radio button for race selection
selected_race = st.radio('Select Race', ['White', 'Black', 'NativeAmerican', 'Asian', 'Hawaiian', 'Hispanic', 'Other'])
race_options = ['White', 'Black', 'NativeAmerican', 'Asian', 'Hawaiian', 'Hispanic', 'Other']
race_selection = [int(1 if selected_race == race else 0) for race in race_options]

# Create a radio button for gender selection
selected_gender = st.radio('Select Gender', ['Male', 'Female'])
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
