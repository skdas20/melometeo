import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import webbrowser
import time
import os
import psutil
import requests
import base64
import io
from PIL import Image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ---------- Spotify API Credentials ----------
CLIENT_ID = "846fad93136041818e267b71d006bb20"
CLIENT_SECRET = "1e20266cfc5a4c12aa21ebeaf5ff7503"
REDIRECT_URI = "http://localhost:8888/callback"
SCOPE = "user-read-playback-state,user-modify-playback-state,playlist-read-private,playlist-read-collaborative"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                               client_secret=CLIENT_SECRET,
                                               redirect_uri=REDIRECT_URI,
                                               scope=SCOPE))

# ---------- Load Models ----------
try:
    age_detection_model = load_model('models/age_model_finetuned.h5')
    mood_detection_model = load_model('models/mood_model_finetuned.h5')
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# ---------- Labels ----------
age_labels = ["1-10", "10-20", "20-30", "30-40", "40-50", "51-60", "60-70", "70-80", "80+"]
mood_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

# ---------- Gesture Variables (if needed later) ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

control_state = {
    "song_index": 0,
    "playlist": [],
    "uris": []
}

# ---------- Utility Functions ----------
def get_weather(lat=None, lon=None):
    try:
        if lat is None or lon is None:
            loc_response = requests.get("http://ip-api.com/json")
            loc_data = loc_response.json()
            lat = loc_data.get("lat")
            lon = loc_data.get("lon")
            city = loc_data.get("city", "Unknown")
        else:
            city = "Your Location"
        api_key = "de7b9ec62287a5b3e8cc155a611928ac"
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
        weather_response = requests.get(weather_url)
        weather_data = weather_response.json()
        weather_condition = weather_data["weather"][0]["main"]
        print(f"Detected Weather: {weather_condition} in {city}")
        return weather_condition
    except Exception as e:
        print("Error fetching weather data:", e)
        return "Clear"

def search_music_on_spotify(mood, age_group, location=None):
    if location:
        weather_condition = get_weather(lat=location.get('latitude'), lon=location.get('longitude'))
    else:
        weather_condition = get_weather()
    query = f"{mood} music for age {age_group} on a {weather_condition.lower()} day"
    print(f"Fetching Spotify playlists for: {query}")
    results = sp.search(q=query, type='playlist', limit=1)
    if (results and results.get('playlists') and 
        results['playlists'].get('items') and len(results['playlists']['items']) > 0):
        playlist_id = results['playlists']['items'][0]['id']
        tracks = sp.playlist_tracks(playlist_id)
        uris = [track['track']['uri'] for track in tracks['items']]
        print(f"Generated Spotify Playlist with {len(uris)} tracks.")
        return uris
    else:
        print("No playlists found for the given query. Returning empty list.")
        return []

# /play endpoint returns the external Spotify URL so the client can open it in a new tab.
@app.route('/play', methods=['POST'])
def play():
    data = request.get_json()
    mood = data.get('mood')
    age = data.get('age')
    location = data.get('location')
    if not mood or not age:
        return jsonify({'error': 'Mood or age missing'}), 400
    uris = search_music_on_spotify(mood, age, location)
    if uris:
        track_uri = uris[0]  # e.g., "spotify:track:xxx"
        track_id = track_uri.split(":")[-1]
        external_url = f"https://open.spotify.com/track/{track_id}"
        print(f"Returning Spotify URL: {external_url}")
        return jsonify({'status': 'success', 'external_url': external_url})
    else:
        return jsonify({'error': 'No tracks found'})

def play_song_on_spotify(uri):
    # Old function not used on client side now.
    try:
        devices = sp.devices()
        if devices["devices"]:
            device_id = devices["devices"][0]["id"]
            sp.start_playback(device_id=device_id, uris=[uri])
            print(f"Playing song on Spotify device: {uri}")
        else:
            web_url = f"https://open.spotify.com/track/{uri.split(':')[-1]}"
            print(f"No active devices. Opening song in web browser: {web_url}")
            webbrowser.open(web_url)
            time.sleep(3)
    except Exception as e:
        print(f"Error playing song: {e}")

def close_spotify_window():
    for proc in psutil.process_iter(['pid', 'name']):
        if 'chrome' in proc.info['name'].lower():
            try:
                for connection in proc.connections(kind='inet'):
                    if 'spotify' in connection.raddr.ip:
                        proc.kill()
                        print("Closed Spotify window.")
                        return
            except psutil.NoSuchProcess:
                continue
    print("No Spotify window found to close.")

# ---------- Inference Function ----------
def process_frame(img_cv):
    # Age detection on full image (resize to 126x126)
    age_input = cv2.resize(img_cv, (126, 126)) / 255.0
    age_input = np.expand_dims(age_input, axis=0)
    age_pred = age_detection_model.predict(age_input)
    age_label = age_labels[np.argmax(age_pred)]
    
    # Mood detection on grayscale image (resize to 48x48)
    gray_frame = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    mood_input = cv2.resize(gray_frame, (48, 48)) / 255.0
    mood_input = np.expand_dims(mood_input, axis=-1)
    mood_input = np.expand_dims(mood_input, axis=0)
    mood_pred = mood_detection_model.predict(mood_input)
    mood_label = mood_labels[np.argmax(mood_pred)]
    
    print(f"[DEBUG] Age: {age_label} | Mood: {mood_label}")
    return age_label, mood_label

# ---------- Flask Routes ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = data.get('image')
    user_location = data.get('location')
    if not img_data:
        return jsonify({'error': 'No image provided'})
    if ',' in img_data:
        img_str = img_data.split(',')[1]
    else:
        img_str = img_data
    try:
        img_bytes = base64.b64decode(img_str)
    except Exception as e:
        return jsonify({'error': f"Error decoding image: {e}"})
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({'error': f"Error opening image: {e}"})
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    age_label, mood_label = process_frame(img_cv)
    return jsonify({'age': age_label, 'mood': mood_label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
