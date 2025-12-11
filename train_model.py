import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATA_URL = "https://raw.githubusercontent.com/florian-huber/weather_prediction_dataset/main/dataset/weather_prediction_dataset.csv"
CSV_FILE = "weather_prediction_dataset.csv"

if not os.path.exists(CSV_FILE):
    response = requests.get(DATA_URL)
    with open(CSV_FILE, 'wb') as f:
        f.write(response.content)

df = pd.read_csv(CSV_FILE)

def calculate_quality(row):
    temp = row['BASEL_temp_mean']
    humidity = row['BASEL_humidity']
    cloud_cover = row['BASEL_cloud_cover']
    
    cloud_factor = cloud_cover * 12.5
    
    score = 100 - abs(temp - 22) * 2 - abs(humidity - 50) * 0.5 - cloud_factor * 0.5
    return max(0, min(100, score))

df = df.dropna(subset=['BASEL_temp_mean', 'BASEL_humidity', 'BASEL_cloud_cover'])

df['quality'] = df.apply(calculate_quality, axis=1)

X = df[['BASEL_temp_mean', 'BASEL_humidity', 'BASEL_cloud_cover']].values
y = df['quality'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    layers.Input(shape=(3,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

loss, mae = model.evaluate(X_test, y_test)

model.save('weather_model.keras')

