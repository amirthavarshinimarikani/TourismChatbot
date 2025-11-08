import streamlit as st
import os
import re
from datetime import timedelta, datetime
import numpy as np
import feedparser
import requests
import spacy
from dateparser.search import search_dates
import tiktoken
from openai import OpenAI

# ========== INITIALIZATION ==========

@st.cache_resource
def init_llm():
    api_key = "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"
    base_url = 'https://openrouter.ai/api/v1'
    client = OpenAI(api_key=api_key, base_url=base_url)
    model = 'deepseek/deepseek-chat-v3-0324'
    return client, model

@st.cache_resource
def init_spacy():
    try:
        return spacy.load("en_core_web_trf")
    except:
        return spacy.load("en_core_web_sm")

# ========== HELPERS ==========

_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10}

def _word_to_int(s):
    s = s.lower().strip()
    if s.isdigit(): return int(s)
    return _WORD_NUM.get(s)

def _extract_duration_days(text):
    patterns = [
        r'for\s+(\d+)\s+days?', r'for\s+([a-zA-Z]+)\s+days?',
        r'(\d+)[-\s]?day', r'([a-zA-Z]+)[-\s]?day',
        r'for\s+(\d+)\s+nights?', r'for\s+([a-zA-Z]+)\s+nights?'
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            token = m.group(1)
            val = int(token) if token.isdigit() else _word_to_int(token)
            if val: return val
    return None

# ========== TRIP PARSER ==========

def extract_trip_details(text, nlp, prefer_future_dates=True):
    if not text:
        return {"source": None, "destination": None, "start_date": None, "return_date": None, "duration_days": None}
    doc = nlp(text)
    gpes = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")]
    src = dst = None
    text_stripped = text.strip()

    m = re.search(r'\bfrom\s+([A-Z][a-zA-Z\s]{1,60}?)\s+(?:to|->|-)\s+([A-Z][a-zA-Z\s]{1,60}?)\b', text_stripped)
    if m:
        src, dst = m.group(1).strip().title(), m.group(2).strip().title()
    else:
        m2 = re.search(r'\b([A-Z][a-zA-Z\s]{1,60}?)\s+(?:to|->|-)\s+([A-Z][a-zA-Z\s]{1,60}?)\b', text_stripped)
        if m2:
            src, dst = m2.group(1).strip().title(), m2.group(2).strip().title()

    if not src and gpes:
        src = gpes[0].title()
    if not dst and len(gpes) > 1:
        dst = gpes[1].title()
    elif len(gpes) == 1 and re.search(r'\b(to|visit|going to|trip to)\s+' + re.escape(gpes[0]), text, re.I):
        dst = gpes[0].title()

    dp_settings = {'PREFER_DATES_FROM': 'future'} if prefer_future_dates else {}
    try:
        sd = search_dates(text, settings=dp_settings) or []
    except:
        sd = []

    full_lower = text.lower()
    found = [(full_lower.find(mtext.lower()), mtext, dt) for mtext, dt in sd if full_lower.find(mtext.lower()) != -1]
    found.sort(key=lambda x: x[0])
    start_date = return_date = None
    duration = _extract_duration_days(text)

    if len(found) >= 2:
        start_date, return_date = found[0][2], found[1][2]
    elif len(found) == 1:
        start_date = found[0][2]
        if duration: return_date = start_date + timedelta(days=duration)

    if start_date and return_date and not duration:
        duration = (return_date - start_date).days

    return {"source": src, "destination": dst, "start_date": start_date, "return_date": return_date, "duration_days": duration}

# ========== WEATHER API ==========

@st.cache_data(ttl=1800)
def get_weather(city):
    if not city: return None
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(city)}&count=1&language=en&format=json"
    try:
        geo = requests.get(geo_url, timeout=6).json()
    except:
        return None
    if not geo.get("results"):
        return None
    result = geo["results"][0]
    lat, lon = result["latitude"], result["longitude"]
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&daily=temperature_2m_max,temperature_2m_min"
    try:
        weather = requests.get(weather_url, timeout=6).json()
    except:
        return result, None
    return result, weather

def get_weather_emoji(code):
    mapping = {
        0: "☀️ Sunny", 1: "☀️ Sunny", 2: "⛅ Partly Cloudy", 3: "⛅ Partly Cloudy",
        45: "🌫 Foggy", 48: "🌫 Foggy", 51: "🌦 Drizzle", 53: "🌦 Drizzle", 55: "🌦 Drizzle",
        61: "🌧 Rainy", 63: "🌧 Rainy", 65: "🌧 Rainy", 71: "❄️ Snowy", 73: "❄️ Snowy",
        75: "❄️ Snowy", 95: "⛈ Thunderstorm", 96: "⛈ Thunderstorm", 99: "⛈ Thunderstorm"
    }
    return mapping.get(code, "☁️ Cloudy")

# ========== HOTELS API (VEG FILTER ADDED) ==========

@st.cache_data(ttl=3600)
def get_hotels_by_city(city, vegetarian=False):
    try:
        if not city: return []
        auth_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        auth_data = {"grant_type": "client_credentials",
                     "client_id": "0v0IryggYObLYMgSXOktLUxK8sxk5RUo",
                     "client_secret": "ijLaSc2UVJq86G5T"}
        auth_res = requests.post(auth_url, data=auth_data)
        if auth_res.status_code != 200: return []
        token = auth_res.json()["access_token"]

        CITY_TO_IATA = {"Chennai": "MAA", "Trichy": "TRZ", "Goa": "GOI", "Delhi": "DEL",
                        "Mumbai": "BOM", "Hyderabad": "HYD", "Kolkata": "CCU",
                        "Bengaluru": "BLR", "Kodaikanal": "IXM"}
        iata = CITY_TO_IATA.get(city.title(), city[:3].upper())

        loc_url = f"https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city?cityCode={iata}"
        headers = {"Authorization": f"Bearer {token}"}
        loc_res = requests.get(loc_url, headers=headers).json()
        hotels = loc_res.get("data", [])
        if not hotels:
            return []

        results = []
        for h in hotels[:3]:
            name = h.get("name", "Unknown")
            address = ", ".join(filter(None, h.get("address", {}).get("lines", []) +
                                       [h.get("address", {}).get("cityName", "")]))
            if vegetarian:
                if any(word in name.lower() for word in ["veg", "vegetarian", "saravana", "woodlands", "bhavan"]):
                    results.append({"name": name, "price": "N/A", "address": address})
            else:
                results.append({"name": name, "price": "N/A", "address": address})
        return results
    except Exception as e:
        st.error(f"Hotel API error: {e}")
        return []

# ========== FLIGHTS API ==========

@st.cache_data(ttl=3600)
def get_flights_by_route(source, destination, date=None):
    try:
        if not (source and destination): return []
        auth_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        auth_data = {"grant_type": "client_credentials",
                     "client_id": "0v0IryggYObLYMgSXOktLUxK8sxk5RUo",
                     "client_secret": "ijLaSc2UVJq86G5T"}
        auth_res = requests.post(auth_url, data=auth_data)
        if auth_res.status_code != 200: return []
        token = auth_res.json()["access_token"]

        CITY_TO_IATA = {"Chennai": "MAA", "Trichy": "TRZ", "Goa": "GOI", "Delhi": "DEL",
                        "Mumbai": "BOM", "Hyderabad": "HYD", "Kolkata": "CCU",
                        "Bengaluru": "BLR", "Kodaikanal": "IXM"}
        origin, dest = CITY_TO_IATA.get(source.title()), CITY_TO_IATA.get(destination.title())
        if not (origin and dest): return []
        date_str = (date or datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        url = f"https://test.api.amadeus.com/v2/shopping/flight-offers?originLocationCode={origin}&destinationLocationCode={dest}&departureDate={date_str}&adults=1"
        headers = {"Authorization": f"Bearer {token}"}
        res = requests.get(url, headers=headers).json()
        if "errors" in res: return []

        flights = []
        for item in res.get("data", [])[:2]:
            price = item["price"]["total"]
            segments = item["itineraries"][0]["segments"]
            stops = len(segments) - 1
            flights.append({"type": f"{stops} stop{'s' if stops > 0 else ''}", "price": f"{price} {item['price']['currency']}"})
        return flights
    except Exception as e:
        st.error(f"Flight API error: {e}")
        return []

# ========== NEWS API ==========

@st.cache_data(ttl=1800)
def fetch_news(destination, max_articles=5):
    rss_url = f"https://news.google.com/rss/search?q={destination}+travel+tourism&hl=en-IN&gl=IN&ceid=IN:en&when=7d"
    feed = feedparser.parse(rss_url)
    return [{'title': e.title, 'summary': e.get('summary', '')[:300] + '...', 'link': e.link} for e in feed.entries[:max_articles]]

def simple_retrieve(query, articles, top_k=3):
    query_words = set(query.lower().split())
    scored = []
    for art in articles:
        art_words = set((art['title'] + ' ' + art['summary']).lower().split())
        score = len(query_words.intersection(art_words))
        scored.append((score, art))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [art for _, art in scored[:top_k]]

# ========== AGGREGATOR ==========

def fetch_all_apis(trip, client, model, vegetarian=False):
    dest, src, start_date = trip['destination'], trip['source'], trip['start_date']
    context_parts = []

    # Weather
    w = get_weather(dest)
    if w:
        loc, weather = w
        cw = weather.get('current_weather', {})
        temp = cw.get('temperature', 'N/A')
        code = cw.get('weathercode', 0)
        forecast = weather.get('daily', {})
        max_temp = forecast.get('temperature_2m_max', [None])[0]
        context_parts.append(f"Weather in {dest}: {get_weather_emoji(code)} {temp}°C, High: {max_temp}°C.")

    # Hotels
    hotels = get_hotels_by_city(dest, vegetarian=vegetarian)
    if hotels:
        hotel_str = " | ".join([f"{h['name']} ({h['price']})" for h in hotels])
        context_parts.append(f"Hotels in {dest}: {hotel_str}")

    # Flights
    flights = get_flights_by_route(src, dest, start_date)
    if flights:
        flight_str = " | ".join([f"{f['type']}: {f['price']}" for f in flights])
        context_parts.append(f"Flights from {src} to {dest}: {flight_str}")

    return "\n".join(context_parts) or "No data available."

# ========== LLM RESPONSE ==========

def generate_travel_response(user_input, client, model, nlp, last_trip=None):
    trip = extract_trip_details(user_input, nlp)

    # Memory-based fallback
    if last_trip:
        for k in trip:
            if not trip[k] and last_trip.get(k):
                trip[k] = last_trip[k]

    # Detect vegetarian preference
    vegetarian = bool(re.search(r"\bveg|vegetarian\b", user_input.lower()))

    # Display parsed details in an expandable section for interactivity
    with st.expander("📋 Parsed Trip Details", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("From", trip['source'] or 'N/A')
        with col2:
            st.metric("To", trip['destination'] or 'N/A')
        with col3:
            st.metric("Duration", f"{trip['duration_days']} days" if trip['duration_days'] else 'N/A')
        st.write(f"Start: {trip['start_date']} | Return: {trip['return_date']}")

    if not trip['destination']:
        st.warning("No destination detected — please rephrase your plan.")
        return trip, None

    # Show loading spinner for interactivity
    with st.spinner("Planning your adventure... 🧳"):
        api_context = fetch_all_apis(trip, client, model, vegetarian)
        prompt = f"You are a travel assistant. User wants: {user_input}. Trip details: {trip}. Context: {api_context}. Make a personalized response."
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a friendly travel planner."},
                      {"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.7,
        )
    return trip, response.choices[0].message.content

# ========== STREAMLIT UI ==========

# Custom CSS for enhanced look and feel
st.markdown("""
    <style>
    .stChatMessage { padding: 1rem; border-radius: 10px; margin: 0.5rem 0; }
    .user-message { background-color: #E3F2FD; color: #1976D2; }
    .assistant-message { background-color: #F3E5F5; color: #7B1FA2; }
    .chat-container { max-height: 500px; overflow-y: auto; }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Pack & Play", page_icon="🧳", layout="wide")
st.title("🧳 Pack & Play — Your Travel Buddy")
st.markdown("Chat naturally about your travel plans! Let's make it unforgettable. ✈️🌴")

# Sidebar for quick actions and trip summary
with st.sidebar:
    st.header("🚀 Quick Actions")
    if st.button("Start New Trip", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_trip = None
        st.rerun()
    if st.session_state.last_trip:
        st.header("📍 Current Trip")
        st.json(st.session_state.last_trip)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi 👋! I'm your travel buddy. Tell me your plan — like 'Trip from Trichy to Goa on Nov 9 for 3 days, vegetarian options please'."}]
if "last_trip" not in st.session_state:
    st.session_state.last_trip = None

# Chat container for scrolling
chat_container = st.container()
with chat_container:
    # Custom chat display for better separation and styling
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(f'<div class="user-message stChatMessage">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🧳"):
                st.markdown(f'<div class="assistant-message stChatMessage">{msg["content"]}</div>', unsafe_allow_html=True)

# Interactive chat input with placeholder
if user_input := st.chat_input("What's your next adventure? (e.g., 'From Chennai to Delhi next week')"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(f'<div class="user-message stChatMessage">{user_input}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="🧳"):
        client, model = init_llm()
        nlp = init_spacy()
        try:
            trip, reply = generate_travel_response(user_input, client, model, nlp, st.session_state.last_trip)
            if reply:
                st.session_state.last_trip = trip  # Update sidebar summary
                st.markdown(f'<div class="assistant-message stChatMessage">{reply}</div>', unsafe_allow_html=True)
                
                # Add quick reply buttons for interactivity
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("More Hotels 🏨", key="hotels_btn"):
                        st.session_state.messages.append({"role": "user", "content": "Show more hotel options"})
                        st.rerun()
                with col2:
                    if st.button("Flight Deals ✈️", key="flights_btn"):
                        st.session_state.messages.append({"role": "user", "content": "Find cheaper flights"})
                        st.rerun()
                with col3:
                    if st.button("Weather Update ☀️", key="weather_btn"):
                        st.session_state.messages.append({"role": "user", "content": "Latest weather forecast"})
                        st.rerun()
            else:
                st.warning("No valid itinerary generated.")
        except Exception as e:
            st.error(f"Error: {e}")
            reply = f"Oops! Something went wrong: {e}"

    st.session_state.messages.append({"role": "assistant", "content": reply})
