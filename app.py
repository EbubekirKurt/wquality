import datetime
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import numpy as np
import requests
import tensorflow as tf
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, Response, url_for
from flask_compress import Compress
from tensorflow import keras

load_dotenv()

app = Flask(__name__, static_folder='static')
Compress(app)  # Enable gzip compression

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
        "Running": {"icon": "fa-person-running", "day": None, "score": -1},
        "Picnic": {"icon": "fa-utensils", "day": None, "score": -1},
        "Photography": {"icon": "fa-camera", "day": None, "score": -1}
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

    # Replace None with friendly message
    for activity_name, activity_data in activities.items():
        if activity_data["day"] is None:
            activity_data["day"] = "Belki haftaya?"

    return activities


def fetch_single_year_history(lat, lon, year_offset):
    """Fetches historical data for a single year (used for parallel execution)."""
    try:
        target_date = datetime.date.today()
        past_start = target_date - datetime.timedelta(days=365 * year_offset)
        past_end = past_start + datetime.timedelta(days=6)

        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': past_start.strftime('%Y-%m-%d'),
            'end_date': past_end.strftime('%Y-%m-%d'),
            'daily': 'temperature_2m_mean',
            'timezone': 'auto'
        }

        res = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=3)
        if res.status_code == 200:
            data = res.json()
            if 'daily' in data and 'temperature_2m_mean' in data['daily']:
                temps = data['daily']['temperature_2m_mean']
                if temps:
                    avg_temp = sum(temps) / len(temps)
                    return {
                        'year': past_start.year,
                        'temp': round(avg_temp, 1)
                    }
    except Exception:
        pass
    return None


def get_historical_data(lat, lon, start_date, end_date):
    """Fetches historical weather data for the same period over the last 5 years (parallel)."""
    try:
        years_to_fetch = 5
        trend_data = []

        # Parallel execution for faster historical data fetching
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch_single_year_history, lat, lon, i) 
                      for i in range(1, years_to_fetch + 1)]
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    trend_data.append(result)

        trend_data.sort(key=lambda x: x['year'])
        return trend_data
    except Exception as e:
        return []


@ttl_cache(ttl=600)
def get_city_forecast(city_name, include_history=False):
    """Fetches weather and predicts quality for a given city."""
    try:
        geo_params = {'name': city_name, 'count': 1, 'language': 'en', 'format': 'json'}
        geo_res = requests.get(GEOCODING_API_URL, params=geo_params, timeout=3)
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

        weather_res = requests.get(WEATHER_API_URL, params=weather_params, timeout=3)
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


@ttl_cache(ttl=300)  # Cache for 5 minutes (300 seconds)
def get_city_today_forecast(city_name):
    """Gets only today's forecast for a city (faster, for sidebar)."""
    try:
        geo_params = {'name': city_name, 'count': 1, 'language': 'en', 'format': 'json'}
        geo_res = requests.get(GEOCODING_API_URL, params=geo_params, timeout=2)
        geo_data = geo_res.json()

        if not geo_data.get('results'):
            return None

        lat = geo_data['results'][0]['latitude']
        lon = geo_data['results'][0]['longitude']

        weather_params = {
            'latitude': lat,
            'longitude': lon,
            'daily': 'temperature_2m_mean,relative_humidity_2m_mean,cloud_cover_mean,weathercode',
            'timezone': 'auto',
            'forecast_days': 1,
            'past_days': 0
        }

        weather_res = requests.get(WEATHER_API_URL, params=weather_params, timeout=2)
        weather_data = weather_res.json()

        if 'daily' not in weather_data:
            return None

        daily = weather_data['daily']
        if not daily['time']:
            return None

        t = daily['temperature_2m_mean'][0]
        h = daily['relative_humidity_2m_mean'][0]
        c = daily['cloud_cover_mean'][0]
        code = daily['weathercode'][0]
        c_oktas = (c / 100.0) * 8.0

        input_data = np.array([[t, h, c_oktas]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled, verbose=0)
        score = float(prediction[0][0])
        score = round(max(0, min(100, score)), 1)

        cat_name, cat_color = get_quality_category(score)

        return {
            'temp': round(t, 1),
            'humidity': round(h, 1),
            'cloud_cover': round(c_oktas, 1),
            'quality': score,
            'icon': get_weather_icon(code),
            'category': cat_name,
            'color': cat_color,
            'lat': lat,
            'lon': lon
        }
    except Exception:
        return None


DEFAULT_CITIES = [
    "New York", "Tokyo", "London", "Paris", "Berlin", "Sydney",
    "Dubai", "Singapore", "Los Angeles", "Toronto", "Mumbai",
    "Cairo", "Rio de Janeiro", "Moscow", "Cape Town", "Bangkok"
]


@app.route('/predict', methods=['POST'])
def predict():
    """Handle POST request and redirect to GET to avoid form resubmission."""
    search_city = request.form.get('city', 'Istanbul')
    # Redirect to GET request with city parameter
    return redirect(url_for('index', city=search_city))


@app.route('/', methods=['GET'])
def index():
    """Render empty page - all data will be loaded via JavaScript for instant page load."""
    # Get city from query parameter (from redirect) or default to Istanbul
    main_city = request.args.get('city', 'Istanbul')
    
    # Return empty template - JavaScript will fetch all data
    return render_template('index.html',
                           main_city=main_city,
                           main_forecast=None,
                           history=[],
                           activities={},
                           other_forecasts={},
                           error=None)


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

    include_history = request.args.get('history', 'false').lower() == 'true'
    forecast, error = get_city_forecast(city, include_history=include_history)
    if error:
        return {'error': error}, 404

    return {'city': city, 'forecast': forecast}


# Cache for all other cities (shared cache)
_other_cities_cache = {}
_other_cities_cache_time = None
_other_cities_cache_ttl = 600  # 10 minutes (increased)
_other_cities_cache_etag = None


@app.route('/api/other-cities', methods=['GET'])
def other_cities_api():
    """Returns forecasts for default cities (uses cache if available)."""
    global model, scaler, _other_cities_cache, _other_cities_cache_time, _other_cities_cache_etag
    
    if model is None or scaler is None:
        load_assets()

    # Check ETag from request
    if_none_match = request.headers.get('If-None-Match')
    
    # Check if cache is valid
    now = time.time()
    cache_age = 0
    if _other_cities_cache_time is not None:
        cache_age = now - _other_cities_cache_time
    
    # If ETag matches and cache is valid, return 304 Not Modified
    if (if_none_match and 
        _other_cities_cache_etag and 
        if_none_match == _other_cities_cache_etag and
        cache_age < _other_cities_cache_ttl and
        _other_cities_cache):
        response = Response(status=304)
        response.headers['Cache-Control'] = f'public, max-age={int(_other_cities_cache_ttl - cache_age)}'
        response.headers['ETag'] = _other_cities_cache_etag
        return response
    
    if (_other_cities_cache_time is not None and 
        cache_age < _other_cities_cache_ttl and
        _other_cities_cache):
        # Return cached data with cache headers
        cache_hash = hashlib.md5(json.dumps(_other_cities_cache, sort_keys=True).encode()).hexdigest()
        _other_cities_cache_etag = f'"{cache_hash}"'
        
        response = jsonify(_other_cities_cache)
        response.headers['Cache-Control'] = f'public, max-age={int(_other_cities_cache_ttl - cache_age)}, immutable'
        response.headers['ETag'] = _other_cities_cache_etag
        response.headers['X-Cache-Status'] = 'HIT'
        return response

    # Cache is empty or expired, fetch new data
    other_forecasts = {}
    cities_to_fetch = []
    
    # Check which cities need to be fetched (not in cache or expired)
    for city in DEFAULT_CITIES:
        # get_city_today_forecast has its own cache, so we can call it
        # It will return cached data if available
        cities_to_fetch.append(city)
    
    # Parallel execution for faster loading
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_city = {executor.submit(get_city_today_forecast, city): city 
                         for city in cities_to_fetch}
        
        for future in as_completed(future_to_city):
            city = future_to_city[future]
            try:
                result = future.result()
                if result:
                    other_forecasts[city] = result
            except Exception:
                pass

    # Update cache
    _other_cities_cache = other_forecasts
    _other_cities_cache_time = now
    cache_hash = hashlib.md5(json.dumps(other_forecasts, sort_keys=True).encode()).hexdigest()
    _other_cities_cache_etag = f'"{cache_hash}"'

    response = jsonify(other_forecasts)
    response.headers['Cache-Control'] = f'public, max-age={_other_cities_cache_ttl}, immutable'
    response.headers['ETag'] = _other_cities_cache_etag
    response.headers['X-Cache-Status'] = 'MISS'
    return response


@app.route('/api/history', methods=['GET'])
def history_api():
    """Returns historical data for a city."""
    city = request.args.get('city')
    if not city:
        return {'error': 'City parameter required'}, 400

    try:
        geo_params = {'name': city, 'count': 1, 'language': 'en', 'format': 'json'}
        geo_res = requests.get(GEOCODING_API_URL, params=geo_params, timeout=3)
        geo_data = geo_res.json()

        if not geo_data.get('results'):
            return {'error': 'City not found'}, 404

        lat = geo_data['results'][0]['latitude']
        lon = geo_data['results'][0]['longitude']
        
        # Get today's date for history calculation
        today = datetime.date.today()
        start_date = today.strftime('%Y-%m-%d')
        end_date = (today + datetime.timedelta(days=14)).strftime('%Y-%m-%d')
        
        history = get_historical_data(lat, lon, start_date, end_date)
        return jsonify({'history': history})
    except Exception as e:
        return {'error': str(e)}, 500


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


@app.route('/about')
def about():
    """Static page explaining how the weather quality is calculated."""
    return render_template('about.html')


@app.route('/api-docs')
def api_docs():
    """API documentation page."""
    return render_template('api_docs.html')


@app.route('/compare')
def compare():
    """City comparison page."""
    return render_template('compare.html')


@app.route('/api/locale/<lang>')
def get_locale(lang):
    """Get locale file for a language."""
    allowed_langs = ['en', 'tr', 'fr', 'de', 'kr']
    if lang not in allowed_langs:
        lang = 'en'
    
    try:
        locale_path = os.path.join('static', 'locales', f'{lang}.json')
        with open(locale_path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({'error': str(e)}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5055, debug=True)
