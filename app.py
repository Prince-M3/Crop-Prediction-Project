from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the dataset
crop_data = pd.read_csv('crop_data.csv')

# Extract unique soil types and climates
soil_types = sorted(crop_data['Soil_Type'].unique())
climate_types = sorted(crop_data['Climate'].unique())

# Preprocess the data
features = ['pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Rainfall']
X = crop_data[features]
y = crop_data['Crop_Type']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

@app.route('/')
def index():
    return render_template('index.html', soil_types=soil_types, climate_types=climate_types)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs
    ph = float(request.form['ph'])
    nitrogen = float(request.form['nitrogen'])
    phosphorus = float(request.form['phosphorus'])
    potassium = float(request.form['potassium'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    rainfall = float(request.form['rainfall'])
    soil_type = request.form['soil_type']
    climate = request.form['climate']

    # Prepare input data
    input_data = [[ph, nitrogen, phosphorus, potassium, temperature, humidity, rainfall]]
    
    # Predict the crop
    predicted_index = model.predict(input_data)[0]
    predicted_crop = label_encoder.inverse_transform([predicted_index])[0]

    # Get data for the predicted crop
    predicted_crop_data = crop_data[crop_data['Crop_Type'] == predicted_crop]
    
    # Handle the case where there is no data for the selected soil type for the predicted crop
    suitable_soil_data = predicted_crop_data[predicted_crop_data['Soil_Type'] == soil_type]

    # If no match for the soil type, default to the mean values for the entire predicted crop dataset
    if suitable_soil_data.empty:
        suitable_soil_data = predicted_crop_data.mean(numeric_only=True)

    # Calculate the yield percentage for each feature
    yield_percentage = {
        'pH': (ph / suitable_soil_data['pH']) * 100,
        'Nitrogen': (nitrogen / suitable_soil_data['Nitrogen']) * 100,
        'Phosphorus': (phosphorus / suitable_soil_data['Phosphorus']) * 100,
        'Potassium': (potassium / suitable_soil_data['Potassium']) * 100,
        'Temperature': (temperature / suitable_soil_data['Temperature']) * 100,
        'Humidity': (humidity / suitable_soil_data['Humidity']) * 100,
        'Rainfall': (rainfall / suitable_soil_data['Rainfall']) * 100
    }

    # Clamp the values to ensure they are between 0 and 100
    for key in yield_percentage:
        yield_percentage[key] = max(0, min(100, yield_percentage[key]))

    # Calculate total prediction yield percentage
    valid_yield_percentages = [value for value in yield_percentage.values() if not np.isnan(value)]
    total_yield_percentage = np.mean(valid_yield_percentages) if valid_yield_percentages else 0

    # Pass data to the template
    return render_template(
        'result.html',
        predicted_crop=predicted_crop,
        ph=ph,
        nitrogen=nitrogen,
        phosphorus=phosphorus,
        potassium=potassium,
        temperature=temperature,
        humidity=humidity,
        rainfall=rainfall,
        soil_type=soil_type,
        climate=climate,
        total_yield_percentage=total_yield_percentage,
        yield_percentage=yield_percentage,
        suitable_soil_data=suitable_soil_data.to_dict(),  # Pass suitable_soil_data for use in the template
        numeric_features=features,
        user_values=[ph, nitrogen, phosphorus, potassium, temperature, humidity, rainfall],
        crop_values=[suitable_soil_data[feature] if feature in suitable_soil_data else np.nan for feature in features]  # Predicted crop averages
    )

if __name__ == '__main__':
    app.run(debug=True)
