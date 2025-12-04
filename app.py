from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import joblib
import datetime
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load model and scaler at startup
MODEL_PATH = 'weather_model.keras'
SCALER_PATH = 'scaler.pkl'
model = None
scaler = None

# API Config
WEATHER_API_URL = os.getenv('WEATHER_API_URL')
GEOCODING_API_URL = os.getenv('GEOCODING_API_URL')

def load_assets():
    global model, scaler
    try:
        print("Attempting to load model...")
        # Try loading local file first
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1000:
             print(f"Loading local model from {MODEL_PATH}...")
             model = keras.models.load_model(MODEL_PATH)
        else:
             print("Local model not found or too small (LFS pointer). Downloading from Hub...")
             from huggingface_hub import hf_hub_download
             # Download from the Space itself
             model_path = hf_hub_download(repo_id="ebubekirkurtt/weather-quality-app", filename="weather_model.keras", repo_type="space")
             model = keras.models.load_model(model_path)
             
        if os.path.exists(SCALER_PATH):
            print(f"Loading scaler from {SCALER_PATH}...")
            scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        print(f"Error loading assets: {e}")

load_assets()

import time
from functools import wraps

# Simple TTL Cache Decorator
def ttl_cache(maxsize=128, ttl=300):
    def decorator(func):
        cache = {}
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            now = time.time()
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl:
                    return result
            result = func(*args, **kwargs)
            if len(cache) >= maxsize:
                cache.pop(next(iter(cache))) # Remove oldest
            cache[key] = (result, now)
            return result
        return wrapper
    return decorator

def get_weather_icon(code):
    """Maps WMO weather code to FontAwesome icon class."""
    if code == 0: return "fa-sun"
    if code in [1, 2, 3]: return "fa-cloud-sun"
    if code in [45, 48]: return "fa-smog"
    if code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: return "fa-cloud-rain"
    if code in [71, 73, 75, 77, 85, 86]: return "fa-snowflake"
    if code in [95, 96, 99]: return "fa-bolt"
    return "fa-cloud"

def get_quality_category(score):
    if score >= 90: return "Excellent", "#00b894" # Green
    if score >= 70: return "Good", "#0984e3"      # Blue
    if score >= 50: return "Moderate", "#fdcb6e"  # Yellow
    if score >= 30: return "Poor", "#e17055"      # Orange
    return "Terrible", "#d63031"                  # Red

    if score >= 30: return "Poor", "#e17055"      # Orange
    return "Terrible", "#d63031"                  # Red

def generate_explanation(temp, humidity, cloud_cover, score):
    """Generates a short explanation for the quality score."""
    if score >= 80:
        if 18 <= temp <= 25: return "Ideal Temperature"
        if humidity < 50: return "Low Humidity"
        return "Great Conditions"
    if score < 50:
        if humidity > 80: return "High Humidity"
        if cloud_cover > 80: return "Heavy Cloud Cover"
        if temp < 5: return "Too Cold"
        if temp > 30: return "Too Hot"
        return "Poor Conditions"
    return "Moderate Conditions"

def get_activity_recommendations(forecasts):
    """Finds the best day for specific activities."""
    activities = {
        "Running": {"icon": "fa-person-running", "day": "N/A", "score": -1},
        "Picnic": {"icon": "fa-utensils", "day": "N/A", "score": -1},
        "Photography": {"icon": "fa-camera", "day": "N/A", "score": -1}
    }
    
    for day in forecasts:
        # Running: Moderate temp, no rain (we assume high quality = no rain/good conditions)
        run_score = day['quality']
        if 10 <= day['temp'] <= 25 and day['quality'] > 60:
            if run_score > activities["Running"]["score"]:
                activities["Running"]["score"] = run_score
                activities["Running"]["day"] = day['day']

        # Picnic: Warm, dry
        picnic_score = day['quality']
        if 18 <= day['temp'] <= 30 and day['quality'] > 70:
            if picnic_score > activities["Picnic"]["score"]:
                activities["Picnic"]["score"] = picnic_score
                activities["Picnic"]["day"] = day['day']
                
        # Photography: Clear skies (low cloud cover)
        photo_score = (100 - (day['cloud_cover'] * 12.5)) # Rough conversion back to 0-100 scale or just use quality
        if day['quality'] > 60 and day['cloud_cover'] < 3: # < 3 oktas
            if photo_score > activities["Photography"]["score"]:
                activities["Photography"]["score"] = photo_score
                activities["Photography"]["day"] = day['day']
                
    return activities

def get_historical_data(lat, lon, start_date, end_date):
    """Fetches historical weather data for the same period over the last 10 years."""
    try:
        # We want to see the trend of the "same week" across 10 years.
        # Strategy: Fetch daily mean temp for the start_date of the forecast, going back 10 years.
        
        target_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        years_back = 10
        historical_temps = []
        
        # We will fetch a single continuous chunk if possible, but Open-Meteo Archive is good for this.
        # Actually, fetching 10 separate years might be slow. 
        # Optimization: Fetch the entire last 10 years of daily data? No, too heavy.
        # Better: Fetch the specific week for each year. 
        # Even Better for "Climate Trend": Just fetch the annual average? No, we want "This Week in History".
        
        # Let's fetch the data for [start_date, end_date] shifted back by 1..10 years.
        # To avoid 10 API calls, we can try to use the `start_date` and `end_date` of the archive API smartly,
        # but the archive API requires a continuous range.
        # So we will make 5 calls (2 years per call? No, that's complex).
        # Let's stick to a simple loop for now, but limit to 5 years to be safe on latency, or 10 if fast.
        # Let's do 5 years for speed + 1 current year context.
        
        years_to_fetch = 5 
        trend_data = []

        for i in range(1, years_to_fetch + 1):
            past_start = target_date - datetime.timedelta(days=365 * i)
            past_end = past_start + datetime.timedelta(days=6) # 1 week window
            
            params = {
                'latitude': lat,
                'longitude': lon,
                'start_date': past_start.strftime('%Y-%m-%d'),
                'end_date': past_end.strftime('%Y-%m-%d'),
                'daily': 'temperature_2m_mean',
                'timezone': 'auto'
            }
            
            # We use a short timeout to not block too long
            try:
                res = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=2)
                if res.status_code == 200:
                    data = res.json()
                    if 'daily' in data and 'temperature_2m_mean' in data['daily']:
                        temps = data['daily']['temperature_2m_mean']
                        if temps:
                            avg_temp = sum(temps) / len(temps)
                            trend_data.append({
                                'year': past_start.year,
                                'temp': round(avg_temp, 1)
                            })
            except Exception as e:
                print(f"Error fetching history for year {i}: {e}")
                continue
                
        # Sort by year
        trend_data.sort(key=lambda x: x['year'])
        return trend_data
    except Exception as e:
        print(f"Global history error: {e}")
        return []

@ttl_cache(ttl=600) # Cache for 10 minutes
def get_city_forecast(city_name, include_history=True):
    """Fetches weather and predicts quality for a given city."""
    try:
        # 1. Geocoding
        geo_params = {'name': city_name, 'count': 1, 'language': 'en', 'format': 'json'}
        geo_res = requests.get(GEOCODING_API_URL, params=geo_params)
        geo_data = geo_res.json()
        
        if not geo_data.get('results'):
             return None, f"City '{city_name}' not found."
             
        lat = geo_data['results'][0]['latitude']
        lon = geo_data['results'][0]['longitude']
        
        # 2. Weather Forecast
        weather_params = {
            'latitude': lat,
            'longitude': lon,
            'daily': 'temperature_2m_mean,relative_humidity_2m_mean,cloud_cover_mean,weathercode',
            'timezone': 'auto',
            'forecast_days': 14,
            'past_days': 1
        }
        
        weather_res = requests.get(WEATHER_API_URL, params=weather_params)
        weather_data = weather_res.json()
        
        if 'daily' not in weather_data:
             return None, "Could not fetch weather data."
             
        daily = weather_data['daily']
        dates = daily['time']
        temps = daily['temperature_2m_mean']
        humidities = daily['relative_humidity_2m_mean']
        clouds = daily['cloud_cover_mean']
        codes = daily['weathercode']
        
        forecasts = []
        
        # 3. Predict for each day (up to 15 to include yesterday)
        for i in range(min(15, len(dates))):
            date_str = dates[i]
            date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            today = datetime.date.today()
            delta = (date_obj - today).days
            
            if delta == 0:
                day_name = "Today"
            elif delta == 1:
                day_name = "Tomorrow"
            elif delta == -1:
                day_name = "Yesterday"
            else:
                day_name = date_obj.strftime("%A")
            
            t = temps[i]
            h = humidities[i]
            c = clouds[i]
            code = codes[i]
            c_oktas = (c / 100.0) * 8.0
            
            input_data = np.array([[t, h, c_oktas]])
            input_scaled = scaler.transform(input_data)
            
            prediction = model.predict(input_scaled, verbose=0)
            score = float(prediction[0][0])
            score = round(max(0, min(100, score)), 1)
            
            cat_name, cat_color = get_quality_category(score)
            explanation = generate_explanation(t, h, c, score)
            
            forecasts.append({
                'day': day_name,
                'temp': round(t, 1),
                'humidity': round(h, 1),
                'cloud_cover': round(c_oktas, 1),
                'quality': score,
                'icon': get_weather_icon(code),
                'category': cat_name,
                'color': cat_color,
                'lat': lat,
                'lon': lon,
                'explanation': explanation
            })
            
        # 4. Get Extra Data (History & Activities)
        history_temps = []
        if include_history:
            history_temps = get_historical_data(lat, lon, dates[0], dates[-1])
        activities = get_activity_recommendations(forecasts)
        
        return {
            'forecasts': forecasts,
            'history': history_temps,
            'activities': activities
        }, None
    except Exception as e:
        return None, str(e)

DEFAULT_CITIES = [
    "New York", "Tokyo", "London", "Paris", "Berlin", "Sydney",
    "Dubai", "Singapore", "Los Angeles", "Toronto", "Mumbai", 
    "Cairo", "Rio de Janeiro", "Moscow", "Cape Town", "Bangkok"
]

@app.route('/', methods=['GET'])
def index():
    global model, scaler
    if model is None or scaler is None:
        load_assets()

    # Main City: Istanbul
    main_city = "Istanbul"
    data, error = get_city_forecast(main_city)
    
    main_forecast = data['forecasts'] if data else None
    history = data['history'] if data else []
    activities = data['activities'] if data else {}
    
    # Other Cities
    other_forecasts = {}
    for city in DEFAULT_CITIES:
        try:
            # Skip history for sidebar cities to speed up loading (avoids timeout)
            data_other, _ = get_city_forecast(city, include_history=False)
            if data_other and data_other.get('forecasts'):
                forecasts = data_other['forecasts']
                # Find "Today" or fallback to first available
                today_forecast = next((f for f in forecasts if f.get('day') == "Today"), None)
                if not today_forecast and forecasts:
                    today_forecast = forecasts[0]
                
                if today_forecast:
                    other_forecasts[city] = today_forecast
        except Exception as e:
            print(f"Error processing {city}: {e}")

    return render_template('index.html', 
                           main_city=main_city, 
                           main_forecast=main_forecast,
                           history=history,
                           activities=activities,
                           other_forecasts=other_forecasts,
                           error=error)

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler
    if model is None or scaler is None:
        load_assets()

    search_city = request.form.get('city', 'Istanbul')
    
    # Main City: Searched City
    data, error = get_city_forecast(search_city)
    
    main_forecast = data['forecasts'] if data else None
    history = data['history'] if data else []
    activities = data['activities'] if data else {}
    
    # Other Cities (Keep defaults)
    other_forecasts = {}
    for city in DEFAULT_CITIES:
        try:
            data_other, _ = get_city_forecast(city, include_history=False)
            if data_other and data_other.get('forecasts'):
                forecasts = data_other['forecasts']
                today_forecast = next((f for f in forecasts if f.get('day') == "Today"), None)
                if not today_forecast and forecasts:
                    today_forecast = forecasts[0]
                
                if today_forecast:
                    other_forecasts[city] = today_forecast
        except Exception as e:
            print(f"Error processing {city}: {e}")

    return render_template('index.html', 
                           main_city=search_city, 
                           main_forecast=main_forecast,
                           history=history,
                           activities=activities,
                           other_forecasts=other_forecasts,
                           error=error)

@app.route('/api/search-city', methods=['GET'])
def search_city_api():
    query = request.args.get('q', '')
    if len(query) < 3:
        return {'results': []}
        
    try:
        geo_params = {'name': query, 'count': 5, 'language': 'en', 'format': 'json'}
        geo_res = requests.get(GEOCODING_API_URL, params=geo_params)
        geo_data = geo_res.json()
        
        results = []
        if geo_data.get('results'):
            for item in geo_data['results']:
                # Build a display string: "City, Country" or "City, Admin1, Country"
                parts = [item.get('name')]
                if item.get('admin1'): parts.append(item.get('admin1'))
                if item.get('country'): parts.append(item.get('country'))
                
                results.append({
                    'name': item.get('name'),
                    'display': ", ".join(parts),
                    'lat': item.get('latitude'),
                    'lon': item.get('longitude')
                })
        return {'results': results}
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/api/forecast', methods=['GET'])
def forecast_api():
    city = request.args.get('city')
    if not city:
        return {'error': 'City parameter required'}, 400
        
    forecast, error = get_city_forecast(city)
    if error:
        return {'error': error}, 404
        
    return {'city': city, 'forecast': forecast}

@app.route('/api/forecast-by-coords', methods=['GET'])
def forecast_by_coords_api():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    
    if not lat or not lon:
        return {'error': 'Latitude and Longitude required'}, 400
        
    try:
        # Reverse Geocoding to get City Name
        # Open-Meteo Geocoding API doesn't support reverse geocoding directly in the free tier easily 
        # (actually it does via BigDataCloud or others, but let's check if we can use a simple hack or just use the coords).
        # Wait, Open-Meteo doesn't have reverse geocoding. 
        # We will use `nominatim.openstreetmap.org` (Free, requires User-Agent).
        
        headers = {'User-Agent': 'WeatherApp/1.0'}
        reverse_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
        
        res = requests.get(reverse_url, headers=headers, timeout=5)
        city_name = "Unknown Location"
        
        if res.status_code == 200:
            data = res.json()
            # Try to find a suitable name
            address = data.get('address', {})
            city_name = address.get('city') or address.get('town') or address.get('village') or address.get('county') or "Unknown Location"
            
        # Now get forecast for this city name (or just coords if we refactored get_city_forecast, 
        # but get_city_forecast does geocoding internally. 
        # To be efficient, we should refactor get_city_forecast to accept coords, 
        # but for now, let's just pass the name we found.
        
        return forecast_api_internal(city_name)
        
    except Exception as e:
        return {'error': str(e)}, 500

def forecast_api_internal(city):
    """Helper to reuse logic"""
    forecast, error = get_city_forecast(city)
    if error:
        return {'error': error}, 404
    return {'city': city, 'forecast': forecast}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5055, debug=True)
