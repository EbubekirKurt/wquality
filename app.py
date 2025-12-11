import datetime
import os

import joblib
import numpy as np
import requests
import tensorflow as tf
from dotenv import load_dotenv
from flask import Flask, render_template, request
from tensorflow import keras

load_dotenv()

app = Flask(__name__)

MODEL_PATH = 'weather_model.keras'
SCALER_PATH = 'scaler.pkl'
model = None
scaler = None

WEATHER_API_URL = os.getenv('WEATHER_API_URL')
GEOCODING_API_URL = os.getenv('GEOCODING_API_URL')


def load_assets():
    global model, scaler
    try:
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1000:
            model = keras.models.load_model(MODEL_PATH)
        else:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id="ebubekirkurtt/weather-quality-app", filename="weather_model.keras",
                                         repo_type="space")
            model = keras.models.load_model(model_path)

        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        pass


load_assets()

import time
from functools import wraps


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
                cache.pop(next(iter(cache)))
            cache[key] = (result, now)
            return result

        return wrapper

    return decorator


def get_weather_icon(code):
    if code == 0: return "fa-sun"
    if code in [1, 2, 3]: return "fa-cloud-sun"
    if code in [45, 48]: return "fa-smog"
    if code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: return "fa-cloud-rain"
    if code in [71, 73, 75, 77, 85, 86]: return "fa-snowflake"
    if code in [95, 96, 99]: return "fa-bolt"
    return "fa-cloud"


def get_quality_category(score):
    if score >= 90: return "Excellent", "#00b894"
    if score >= 70: return "Good", "#0984e3"
    if score >= 50: return "Moderate", "#fdcb6e"
    if score >= 30: return "Poor", "#e17055"
    return "Terrible", "#d63031"

    if score >= 30: return "Poor", "#e17055"
    return "Terrible", "#d63031"


def generate_explanation(temp, humidity, cloud_cover, score):
    """Generates a detailed 'Explainable AI' insight for the quality score."""
    reasons = []

    if temp < 10:
        reasons.append(f"Very Cold ({temp}°C)")
    elif temp < 18:
        reasons.append(f"Chilly ({temp}°C)")
    elif temp > 30:
        reasons.append(f"Very Hot ({temp}°C)")
    elif temp > 26:
        reasons.append(f"Warm ({temp}°C)")
    else:
                if score > 70: reasons.append("Ideal Temperature")

    if humidity > 80:
        reasons.append(f"High Humidity ({humidity}%)")
    elif humidity < 20:
        reasons.append(f"Dry Air ({humidity}%)")

    if cloud_cover > 80:
        reasons.append("Overcast")
    elif cloud_cover < 20 and score > 70:
        reasons.append("Clear Skies")

    if score >= 80:
        return f"Excellent conditions! {', '.join(reasons)} contributing to a high score."
    elif score >= 50:
        return f"Moderate quality. {', '.join(reasons)}."
    else:
        negatives = [r for r in reasons if "Ideal" not in r and "Clear" not in r]
        if not negatives: negatives = ["Unfavorable conditions"]
        return f"Score lowered by: {', '.join(negatives)}."


def get_activity_recommendations(forecasts):
    """Finds the best day for specific activities."""
    activities = {
        "Running": {"icon": "fa-person-running", "day": "N/A", "score": -1},
        "Picnic": {"icon": "fa-utensils", "day": "N/A", "score": -1},
        "Photography": {"icon": "fa-camera", "day": "N/A", "score": -1}
    }

    for day in forecasts:
        run_score = day['quality']
        if 10 <= day['temp'] <= 25 and day['quality'] > 60:
            if run_score > activities["Running"]["score"]:
                activities["Running"]["score"] = run_score
                activities["Running"]["day"] = day['day']

        picnic_score = day['quality']
        if 18 <= day['temp'] <= 30 and day['quality'] > 70:
            if picnic_score > activities["Picnic"]["score"]:
                activities["Picnic"]["score"] = picnic_score
                activities["Picnic"]["day"] = day['day']

        photo_score = (100 - (day['cloud_cover'] * 12.5))
        if day['quality'] > 60 and day['cloud_cover'] < 3:
            if photo_score > activities["Photography"]["score"]:
                activities["Photography"]["score"] = photo_score
                activities["Photography"]["day"] = day['day']

    return activities


def get_historical_data(lat, lon, start_date, end_date):
    """Fetches historical weather data for the same period over the last 10 years."""
    try:
        target_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        years_back = 10
        historical_temps = []
        years_to_fetch = 5
        trend_data = []

        for i in range(1, years_to_fetch + 1):
            past_start = target_date - datetime.timedelta(days=365 * i)
            past_end = past_start + datetime.timedelta(days=6)  # 1 week window

            params = {
                'latitude': lat,
                'longitude': lon,
                'start_date': past_start.strftime('%Y-%m-%d'),
                'end_date': past_end.strftime('%Y-%m-%d'),
                'daily': 'temperature_2m_mean',
                'timezone': 'auto'
            }

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
                continue

        trend_data.sort(key=lambda x: x['year'])
        return trend_data
    except Exception as e:
        return []


@ttl_cache(ttl=600)
def get_city_forecast(city_name, include_history=True):
    """Fetches weather and predicts quality for a given city."""
    try:
        geo_params = {'name': city_name, 'count': 1, 'language': 'en', 'format': 'json'}
        geo_res = requests.get(GEOCODING_API_URL, params=geo_params)
        geo_data = geo_res.json()

        if not geo_data.get('results'):
            return None, f"City '{city_name}' not found."

        lat = geo_data['results'][0]['latitude']
        lon = geo_data['results'][0]['longitude']

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

        input_batch = []
        for i in range(min(15, len(dates))):
            t = temps[i]
            h = humidities[i]
            c = clouds[i]
            c_oktas = (c / 100.0) * 8.0
            input_batch.append([t, h, c_oktas])

        if input_batch:
            input_data = np.array(input_batch)
            input_scaled = scaler.transform(input_data)
            predictions = model.predict(input_scaled, verbose=0)
        else:
            predictions = []

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

            score = float(predictions[i][0])
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

    main_city = "Istanbul"
    data, error = get_city_forecast(main_city)

    main_forecast = data['forecasts'] if data else None
    history = data['history'] if data else []
    activities = data['activities'] if data else {}

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
            pass

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

    data, error = get_city_forecast(search_city)

    main_forecast = data['forecasts'] if data else None
    history = data['history'] if data else []
    activities = data['activities'] if data else {}

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
            pass

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
        headers = {'User-Agent': 'WeatherApp/1.0'}
        reverse_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"

        res = requests.get(reverse_url, headers=headers, timeout=5)
        city_name = "Unknown Location"

        if res.status_code == 200:
            data = res.json()
            address = data.get('address', {})
            city_name = address.get('city') or address.get('town') or address.get('village') or address.get(
                'county') or "Unknown Location"

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
