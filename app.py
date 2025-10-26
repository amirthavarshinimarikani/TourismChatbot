import streamlit as st
import requests
import spacy

nlp = spacy.load("en_core_web_trf")


st.title("Pack & Play - Your Travel Buddy 🌍")

user_input = st.text_input("Ask me anything:", placeholder="e.g., Plan a trip to Chennai next weekend")

def extract_city(text):
    """Extracts city name from user input using spaCy NER"""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:  
            return ent.text
    return None

def get_weather(city):
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
    geo = requests.get(geo_url).json()
    if not geo.get("results"):
        return None

    result = geo["results"][0]
    lat, lon = result["latitude"], result["longitude"]

    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weathercode"
    weather = requests.get(weather_url).json()

    return result, weather

def get_weather_emoji(code):
    if code in [0, 1]: return "☀️ Sunny"
    elif code in [2, 3]: return "⛅ Partly Cloudy"
    elif code in [45, 48]: return "🌫 Foggy"
    elif code in [51, 53, 55]: return "🌦 Drizzle"
    elif code in [61, 63, 65]: return "🌧 Rainy"
    elif code in [71, 73, 75]: return "❄️ Snowy"
    elif code in [95, 96, 99]: return "⛈ Thunderstorm"
    else: return "☁️ Cloudy"

if user_input:
    city = extract_city(user_input)  # <-- use NLP to find the city automatically
    if city:
        data = get_weather(city)
        if data:
            location, weather = data
            temp = weather["current"]["temperature_2m"]
            desc = get_weather_emoji(weather["current"]["weathercode"])

            st.markdown("---")
            c1, c2 = st.columns([1, 3])
            with c1:
                st.markdown(f"<h1 style='text-align:center'>{desc.split()[0]}</h1>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='text-align:center; color:#FF6B6B'>{temp}°C</h2>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"### {location.get('name')}")
                st.caption(f"{location.get('admin1', '')}, {location.get('country', '')}")
                st.write(f"**Condition:** {desc}")
            st.markdown("---")
        else:
            st.error(f"Could not find weather info for **{city}**.")
    else:
        st.warning("I couldn’t detect any city in your message. Try again, e.g. 'Weather in Chennai' or 'Trip to Goa'.")
