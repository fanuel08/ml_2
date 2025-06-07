# train_model.py

# This script trains the Gradient Boosting model on the full dataset
# and saves the trained model to a file named 'malaria_risk_model.joblib'.

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib

print("--- Starting model training ---")

# Step 1: The complete dataset
# We include the data directly here so this script is self-contained.
data = {
    'County': ['Baringo', 'Bomet', 'Bungoma', 'Busia', 'Elgeyo-Marakwet', 'Embu', 'Garissa', 'Homa Bay', 'Isiolo', 'Kajiado', 'Kakamega', 'Kericho', 'Kiambu', 'Kilifi', 'Kirinyaga', 'Kisii', 'Kisumu', 'Kitui', 'Kwale', 'Laikipia', 'Lamu', 'Machakos', 'Makueni', 'Mandera', 'Marsabit', 'Meru', 'Migori', 'Mombasa', 'Murang\'a', 'Nairobi', 'Nakuru', 'Nandi', 'Narok', 'Nyamira', 'Nyandarua', 'Nyeri', 'Samburu', 'Siaya', 'Taita-Taveta', 'Tana River', 'Tharaka-Nithi', 'Trans Nzoia', 'Turkana', 'Uasin Gishu', 'Vihiga', 'Wajir', 'West Pokot'],
    'Malaria_cases_per_100k': [100, 200, 1500, 2500, 50, 20, 300, 3000, 150, 80, 2000, 300, 10, 1200, 15, 400, 3500, 60, 1300, 40, 900, 30, 45, 250, 180, 70, 2800, 1000, 25, 5, 90, 250, 120, 350, 30, 20, 200, 3200, 500, 700, 55, 600, 400, 150, 1800, 220, 280],
    'Population_Density_per_Sq_Km': [38, 351, 552, 527, 150, 216, 19, 359, 11, 51, 597, 370, 953, 117, 417, 966, 555, 37, 105, 78, 22, 229, 134, 33, 7, 221, 423, 5634, 424, 6247, 287, 349, 66, 903, 195, 230, 15, 403, 27, 9, 149, 399, 15, 331, 1888, 12, 64],
    'Piped_Water_Access_Percent': [26.8, 18.2, 11.5, 9.8, 30.1, 45.2, 33.1, 7.9, 41.5, 48.9, 13.1, 35.7, 57.2, 30.5, 48.1, 15.3, 33.6, 19.4, 21.7, 40.2, 25.1, 36.9, 22.8, 20.7, 31.8, 31.9, 8.1, 55.8, 49.3, 85.1, 41.1, 28.5, 17.6, 12.7, 31.4, 53.7, 29.6, 9.2, 44.3, 16.9, 29.8, 25.6, 23.4, 38.8, 20.1, 28.3, 10.5],
    'Avg_Temperature': [23.5, 19.2, 22.1, 23.4, 18.3, 20.4, 28.9, 22.6, 24.9, 21.1, 22.5, 18.9, 18.6, 26.5, 19.3, 21.0, 22.8, 23.6, 26.2, 18.4, 27.1, 21.8, 22.9, 29.1, 24.8, 19.8, 22.0, 26.3, 18.8, 19.5, 18.2, 20.1, 20.2, 21.8, 16.5, 18.1, 24.0, 22.7, 24.6, 27.8, 21.5, 19.9, 29.2, 17.5, 21.8, 28.5, 22.1],
    'Avg_Precipitation': [2.0, 4.3, 4.7, 4.9, 3.2, 3.9, 0.9, 5.2, 1.1, 1.4, 5.1, 4.9, 3.3, 3.1, 4.1, 4.8, 4.5, 1.9, 3.8, 2.6, 2.9, 2.5, 2.2, 0.6, 0.8, 3.5, 4.4, 3.2, 3.6, 2.1, 2.8, 4.2, 2.3, 4.6, 2.7, 3.8, 1.5, 4.6, 2.2, 1.3, 3.1, 3.7, 0.5, 3.0, 5.0, 0.7, 2.4]
}
final_merged_data = pd.DataFrame(data)
print("Data loaded.")


# Step 2: Prepare data for training
# We use all columns for features except the County name and the target variable itself.
features = final_merged_data.drop(columns=['County', 'Malaria_cases_per_100k'])
target = final_merged_data['Malaria_cases_per_100k']


# Step 3: Initialize and train the final model
# We use the Gradient Boosting Regressor as it was our best-performing model.
# We train it on ALL the data to make it as smart as possible.
final_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
final_model.fit(features, target)
print("Model training complete.")


# Step 4: Save the trained model to a file
# This creates the 'malaria_risk_model.joblib' file.
joblib.dump(final_model, 'malaria_risk_model.joblib')

print("--- Success! Model has been saved to 'malaria_risk_model.joblib' ---")