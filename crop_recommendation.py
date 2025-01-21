import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
import requests
import os

# Sidebar
# Sidebar
st.sidebar.markdown("""
    <h1 style='font-weight: bold; color: transparent; background: linear-gradient(270deg, #CDDC39, #4FC3F7); 
    -webkit-background-clip: text; text-align: center;'>KrishiGyaan</h1>
""", unsafe_allow_html=True)

app_mode = st.sidebar.radio("Select Page", ["HOME", "ABOUT", "CROP PREDICTION", "FEEDBACK"], index=0)

# Display Image
img = Image.open("crops.png")
st.image(img)

# Load Dataset and Train Model
df = pd.read_csv('Crop_recommendation.csv')
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Split the data into training and testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)
RF = RandomForestClassifier(n_estimators=20, random_state=5)
RF.fit(Xtrain, Ytrain)

# Dump the trained Random Forest classifier with Pickle
RF_pkl_filename = 'RF.pkl'
with open(RF_pkl_filename, 'wb') as file:
    pickle.dump(RF, file)

# Load the trained Random Forest model
RF_Model_pkl = pickle.load(open('RF.pkl', 'rb'))

# Function to make predictions
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    # Create a DataFrame with the same column names as the training data
    input_data = pd.DataFrame([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]], 
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    
    # Make prediction using the trained model
    prediction = RF_Model_pkl.predict(input_data)
    return prediction

# Function to load and display an image of the predicted crop
def show_crop_image(crop_name):
    # Construct the URL to the crop image on GitHub
    image_url = f"https://raw.githubusercontent.com/Yashrajgithub/Crop-Recommendation/main/crop_images/{crop_name.lower()}.jpeg"
    
    # Try to load the image directly from the URL
    try:
        st.image(image_url, caption=f"Recommended crop: {crop_name}", use_container_width=True)
    except Exception as e:
        # Fallback to a placeholder image if the URL fails
        placeholder_url = "https://raw.githubusercontent.com/Yashrajgithub/Crop-Recommendation/main/crop_images/placeholder.jpeg"
        st.image(placeholder_url, caption=f"Recommended crop: {crop_name}", use_container_width=True)


# Function to submit feedback
def submit_feedback(name, email, feedback):
    url = "https://api.web3forms.com/submit"
    payload = {
        "access_key": "4ffcbd0a-8334-41a7-af0a-d8552c02dd27",
        "name": name,
        "email": email,
        "message": feedback
    }
    response = requests.post(url, data=payload)
    return response

# Feedback form page
def feedback():
    st.markdown("""
    <h1 style='font-weight: bold; color: transparent; background: linear-gradient(270deg, #CDDC39, #4FC3F7); 
    -webkit-background-clip: text; text-align: center;'>Feedback / Suggestions</h1>
    """, unsafe_allow_html=True)
    st.markdown("We value your feedback! Please provide your suggestions or feedback below:")

    with st.form(key='feedback_form', clear_on_submit=True):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        feedback_text = st.text_area("Your Feedback/Suggestions", height=150)
        
        submit_feedback_btn = st.form_submit_button(label='Submit Feedback')
        
        if submit_feedback_btn:
            if name and email and feedback_text:
                response = submit_feedback(name, email, feedback_text)
                if response.status_code == 200:
                    st.success("Your feedback has been submitted successfully! Thank you.")
                else:
                    st.error("Failed to submit feedback. Please try again later.")
            else:
                st.warning("Please fill out all fields.")

# Main Page
if app_mode == "HOME":
    st.markdown("""
    <h1 style='font-weight: bold; color: transparent; background: linear-gradient(270deg, #CDDC39, #4FC3F7); 
    -webkit-background-clip: text; text-align: center;'>CROP SUGGESTION TOOL</h1>
    """, unsafe_allow_html=True)
    st.subheader("Welcome to our Crop Suggestion Tool!")
    st.write("This tool assists farmers in determining the ideal crop to cultivate, considering various environmental factors. Leveraging cutting-edge data analytics, our platform provides the best crop suggestions for your land based on parameters like soil nutrients, climate conditions, humidity, pH values, and rainfall. Its intuitive design allows users to easily enter their crop information and receive personalized suggestions with just a few clicks. Dive into our platform to explore how the Crop Suggestion Tool can support you and enhance your farming experience!")

    st.subheader("How It Functions:")
    st.write("1. Go to the 'CROP PREDICTION' section in the menu.")
    st.write("2. Provide information about your soil and surrounding environmental factors.")
    st.write("3. Hit the 'Predict' button to receive the suggested crop for your conditions.")

elif app_mode == "ABOUT":
    st.markdown("""
    <h1 style='font-weight: bold; color: transparent; background: linear-gradient(270deg, #CDDC39, #4FC3F7); 
    -webkit-background-clip: text; text-align: center;'>CROP SELECTION GUIDE</h1>
    """, unsafe_allow_html=True)    
    st.write("This Crop Selection Guide is created as part of a Data Analytics initiative. Our platform is designed to help farmers make well-informed choices regarding the best crops to grow, ultimately boosting agricultural productivity and promoting sustainability.")
    st.write("By harnessing data analytics, our system evaluates several factors such as soil nutrients (Nitrogen, Phosphorus, Potassium), environmental variables (temperature, humidity, rainfall), and soil pH levels to offer tailored crop suggestions. By factoring in these essential elements, we aim to enhance crop selection suited for particular land conditions, thus improving yields and profitability for farmers.")

    st.subheader("Main Features:")
    st.write("- Recommends the most appropriate crop based on soil and environmental parameters.")
    st.write("- Utilizes a Random Forest classifier trained on crop data to make predictions.")
    st.write("- Offers an intuitive and accessible interface for farmers to make smart choices.")

elif app_mode == "CROP PREDICTION":
    st.markdown("<h1 style='text-align: center;'>CROP PREDICTION</h1>", unsafe_allow_html=True)

    st.markdown("""
    <h1 style='text-align: center; background: linear-gradient(270deg, #3C8D40, #6A9B4D); color: white; 
    text-shadow: 2px 2px 5px rgba(0,0,0,0.7);'>ADVANCED CROP PREDICTION SOLUTIONS</h1>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Input Environmental Factors")
        nitrogen = st.number_input("Nitrogen", min_value=0.0, max_value=140.0, value=0.0, step=0.1)
        phosphorus = st.number_input("Phosphorus", min_value=0.0, max_value=145.0, value=0.0, step=0.1)
        potassium = st.number_input("Potassium", min_value=0.0, max_value=205.0, value=0.0, step=0.1)
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=51.0, value=0.0, step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=0.0, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)
    
    with col2:
        st.header("Predict Crop")
        if st.button("Predict"):
            inputs = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
            if not inputs.any() or np.isnan(inputs).any() or (inputs == 0).all():
                st.error("Please fill in all input fields with valid values before predicting.")
            else:
                prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
                st.success(f"The recommended crop is: {prediction[0]}")
                show_crop_image(prediction[0])

            st.markdown("<h3 style='text-align: center; font-weight: bold; color: #3C8D40;'>Your crops are ready to thrive. Start planting for success! 🌱</h3>", unsafe_allow_html=True)

elif app_mode == "FEEDBACK":
    feedback()
