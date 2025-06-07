# app.py - Your Streamlit Web App Script

import streamlit as st
import pandas as pd
import joblib

# --- Functions ---

# The @st.cache_resource decorator ensures the model is loaded only once, making the app faster.
@st.cache_resource
def load_model():
    """Loads the pre-trained model from the file."""
    model = joblib.load('malaria_risk_model.joblib')
    return model

# --- Main App ---

# 1. Load the pre-trained model
model = load_model()

# 2. Configure the page layout and title
st.set_page_config(page_title="Kenya Malaria Risk Predictor", page_icon="🦟", layout="wide")

# 3. Create the app's title and description
st.title("🦟 Kenya County Malaria Risk Predictor")
st.markdown("""
This web application predicts the estimated number of malaria cases per 100,000 people in a Kenyan county. 
It uses a **Gradient Boosting** machine learning model trained on public data from HDX, KNBS, and Open-Meteo to support **SDG 3: Good Health and Well-being**.
Enter the environmental and demographic factors for a county in the sidebar to get a prediction.
""")

# 4. Create the input widgets in the sidebar
st.sidebar.header("Enter County Data:")

# Use sliders for a better user experience
pop_density = st.sidebar.slider(
    'Population Density (people per Sq Km)', 
    min_value=0, max_value=7000, value=500, step=10
)
water_access = st.sidebar.slider(
    'Access to Piped Water (%)', 
    min_value=0.0, max_value=100.0, value=30.0, step=0.5
)
temp = st.sidebar.slider(
    'Average Temperature (°C)', 
    min_value=15.0, max_value=35.0, value=22.0, step=0.1
)
precip = st.sidebar.slider(
    'Average Daily Precipitation (mm)', 
    min_value=0.0, max_value=10.0, value=3.0, step=0.1
)

# 5. Create the prediction logic
# This part runs when the user clicks the button.
if st.sidebar.button("Predict Malaria Risk"):
    # Create a DataFrame from the user's inputs in the correct order
    input_data = {
        'Population_Density_per_Sq_Km': [pop_density],
        'Piped_Water_Access_Percent': [water_access],
        'Avg_Temperature': [temp],
        'Avg_Precipitation': [precip]
    }
    input_df = pd.DataFrame(input_data)
    
    # Use the loaded model to make a prediction
    prediction = model.predict(input_df)
    
    # Display the result
    st.subheader("Prediction Result")
    # Use st.metric for a nicer display
    st.metric(label="Estimated Malaria Cases", value=f"{prediction[0]:.2f} per 100,000 people")
    
    # Add a visual interpretation of the risk level
    if prediction[0] < 100:
        st.success("Risk Level: Low")
    elif prediction[0] < 1000:
        st.warning("Risk Level: Moderate")
    else:
        st.error("Risk Level: High")
        st.markdown("_High-risk areas require priority attention for resource allocation and preventative measures._")

# 6. Add a concluding disclaimer
st.write("---")
st.markdown("""
**Disclaimer:** This is a proof-of-concept tool. The model is based on historical data and should not be used for medical diagnosis or as the sole factor for definitive resource allocation. Always consult with local public health experts.
""")