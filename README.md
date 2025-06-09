🦟 Kenya Malaria Risk Predictor

This web application predicts the estimated number of malaria cases per 100,000 people in a Kenyan county using environmental and demographic factors. It leverages a Gradient Boosting Machine Learning model trained on publicly available data to support Sustainable Development Goal 3: Good Health and Well-being.
🌐 Live Demo

👉 Try the App Now - https://myfineapp.streamlit.app/
(Hosted on Streamlit Cloud)
📂 Project Structure

.
├── app.py                         # Streamlit application script
├── malaria_risk_model.joblib  # Pre-trained ML model
├── README.md                     # This file
` requirements.txt

🧪 Features

    ✅ Interactive sliders for user input

    ✅ Real-time prediction using a trained Gradient Boosting model

    ✅ Risk classification: Low, Moderate, or High

    ✅ Intuitive UI with Streamlit

    ✅ Promotes data-driven health insights aligned with SDG 3

🧠 Model Inputs
Feature	Description
Population Density	People per square kilometer
Access to Piped Water (%)	Percentage of the population with access
Average Temperature (°C)	Mean daily temperature
Daily Precipitation (mm)	Mean daily rainfall
🚀 Getting Started
Prerequisites

Make sure you have:

    Python 3.7 or higher

    streamlit, pandas, joblib

    (Optional) scikit-learn for model retraining

Installation

    Clone the repository:

git clone https://github.com/yourusername/kenya-malaria-risk-predictor.git
cd kenya-malaria-risk-predictor

Install the required packages:

pip install -r requirements.txt

Run the app locally:

    streamlit run app.py

    Access the app:
    Open your browser to http://localhost:8501

📊 Example Scenario

    A local health officer in Kisumu wants to assess how changing climate and water access could impact malaria risk. This app allows them to input new data and instantly receive a prediction, helping support planning and intervention efforts.

⚠️ Disclaimer

This is a proof-of-concept and should not be used for medical diagnosis or official health planning without validation and expert consultation. Results are based on historical, public data and a predictive model.
📄 License

MIT License
(Include LICENSE file if needed)

🙏 Acknowledgements

    Public data from Kenyan health and climate agencies

    Built using Streamlit

    Powered by Gradient Boosting models from scikit-learn

