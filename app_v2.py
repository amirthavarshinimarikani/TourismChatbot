import streamlit as st
import os
import re
from datetime import timedelta, datetime
import numpy as np
import feedparser
import requests
import spacy
from dateparser.search import search_dates
from openai import OpenAI

# ==============================================
# 🧩 INITIALIZATION (SECURE)
# ==============================================

@st.cache_resource
def init_llm():
    """Initialize LLM client using Streamlit secrets."""
    api_key = st.secrets["OPENROUTER_API_KEY"]
    base_url = st.secrets.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)
    model = "deepseek/deepseek-chat-v3-0324"
    return client, model

@st.cache_resource
def init_spacy():
    """Load spaCy NLP model efficiently."""
    try:
        return spacy.load("en_core_web_trf")
    except:
        return spacy.load("en_core_web_sm")


# ==============================================
# 🔤 TEXT HELPERS
# ==============================================

_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10}

def _word_to_int(s):
    s = s.lower().strip()
    if s.isdigit():
        return int(s)
    return _WORD_NUM.get(s)

def _extract_duration_days(text):
    """Extract duration in days from text."""
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
            if val:
                return val
    return None


# ==============================================
# 🧠 TRIP PARSER
# ==============================================

def extract_trip_details(text, nlp, prefer_future_dates=True):
    """Extract source, destination, and dates from text."""
    if not text:
        return {"source": None, "destination": None, "start_date": None, "return_date": None, "duration_days": None}

    doc = nlp(text)
    gpes = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")]
    src = dst = None

    # Regex extraction
    text_stripped = text.strip()
    m = re.search(r'\bfrom\s+([A-Z][a-zA-Z\s]{1,60}?)\s+(?:to|->|-)\s+([A-Z][a-zA-Z\s]{1,60}?)\b', text_stripped)
    if m:
        src, dst = m.group(1).title(), m.group(2).title()
    else:
        m2 = re.search(r'\b([A-Z][a-zA-Z\s]{1,60}?)\s+(?:to|->|-)\s+([A-Z][a-zA-Z\s]{1,60}?)\b', text_stripped)
        if m2:
            src, dst = m2.group(1).title(), m2.group(2).title()

    # NLP-based fallback
    if not src and gpes:
        src = gpes[0].title()
    if not dst and len(gpes) > 1:
        dst = gpes[1].title()

    # Date extraction
    dp_settings = {'PREFER_DATES_FROM': 'future'} if prefer_future_dates else {}
    try:
        sd = search_dates(text, settings=dp_settings) or []
    except:
        sd = []

    found = [(text.lower().find(mtext.lower()), mtext, dt) for mtext, dt in sd]
    found.sort(key=lambda x: x[0])

    start_date = return_date = None
    duration = _extract_duration_days(text)

    if len(found) >= 2:
        start_date, return_date = found[0][2], found[1][2]
    elif len(found) == 1:
        start_date = found[0][2]
        if duration:
            return_date = start_date + timedelta(days=duration)

    if start_date and return_date and not duration:
        duration = (return_date - start_date).days

    return {"source": src, "destination": dst, "start_date": start_date, "return_date": return_date, "duration_days": duration}


# ==============================================
# 🌤 WEATHER API
# ==============================================

@st.cache_data(ttl=1800)
def get_weather(city):
    """Fetch weather info from Open-Meteo API."""
    if not city:
        return None
    try:
        geo = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(city)}&count=1&language=en&format=json",
            timeout=6).json()
    except:
        return None
    if not geo.get("results"):
        return None

    result = geo["results"][0]
    lat, lon = result["latitude"], result["longitude"]
    weather = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&daily=temperature_2m_max,temperature_2m_min",
        timeout=6).json()
    return result, weather

def get_weather_emoji(code):
    mapping = {
        0: "☀️ Sunny", 1: "☀️ Sunny", 2: "⛅ Partly Cloudy", 3: "⛅ Partly Cloudy",
        45: "🌫 Foggy", 51: "🌦 Drizzle", 61: "🌧 Rainy", 71: "❄️ Snowy", 95: "⛈ Thunderstorm"
    }
    return mapping.get(code, "☁️ Cloudy")


# ==============================================
# 🏨 HOTELS API (VEG FILTER)
# ==============================================

@st.cache_data(ttl=3600)
def get_hotels_by_city(city, vegetarian=False):
    """Fetch hotels (mocked or via Amadeus)"""
    # NOTE: Client credentials stored in Streamlit secrets
    client_id = st.secrets["AMADEUS_CLIENT_ID"]
    client_secret = st.secrets["AMADEUS_CLIENT_SECRET"]

    try:
        token_res = requests.post(
            "https://test.api.amadeus.com/v1/security/oauth2/token",
            data={"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret},
            timeout=6)
        token = token_res.json().get("access_token")
        if not token:
            return []

        CITY_TO_IATA = {"Chennai": "MAA", "Goa": "GOI", "Delhi": "DEL", "Mumbai": "BOM",
                        "Hyderabad": "HYD", "Kolkata": "CCU", "Bengaluru": "BLR"}
        iata = CITY_TO_IATA.get(city.title(), city[:3].upper())

        headers = {"Authorization": f"Bearer {token}"}
        res = requests.get(
            f"https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city?cityCode={iata}",
            headers=headers, timeout=6).json()
        hotels = res.get("data", [])[:3]

        result = []
        for h in hotels:
            name = h.get("name", "Unknown")
            if vegetarian and not any(v in name.lower() for v in ["veg", "vegetarian", "bhavan", "saravana", "woodlands"]):
                continue
            result.append({
                "name": name,
                "address": ", ".join(h.get("address", {}).get("lines", [])),
                "price": "N/A"
            })
        return result
    except Exception as e:
        st.error(f"Hotel API error: {e}")
        return []


# ==============================================
# ✈️ FLIGHTS API
# ==============================================

@st.cache_data(ttl=3600)
def get_flights_by_route(source, destination, date=None):
    """Fetch basic flight options."""
    client_id = st.secrets["AMADEUS_CLIENT_ID"]
    client_secret = st.secrets["AMADEUS_CLIENT_SECRET"]
    try:
        token_res = requests.post(
            "https://test.api.amadeus.com/v1/security/oauth2/token",
            data={"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret},
            timeout=6)
        token = token_res.json().get("access_token")
        if not token:
            return []

        CITY_TO_IATA = {"Chennai": "MAA", "Goa": "GOI", "Delhi": "DEL", "Mumbai": "BOM",
                        "Hyderabad": "HYD", "Kolkata": "CCU", "Bengaluru": "BLR"}
        origin, dest = CITY_TO_IATA.get(source.title()), CITY_TO_IATA.get(destination.title())
        if not (origin and dest):
            return []

        date_str = (date or datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        headers = {"Authorization": f"Bearer {token}"}
        res = requests.get(
            f"https://test.api.amadeus.com/v2/shopping/flight-offers?originLocationCode={origin}&destinationLocationCode={dest}&departureDate={date_str}&adults=1",
            headers=headers, timeout=6).json()
        flights = []
        for f in res.get("data", [])[:2]:
            price = f["price"]["total"]
            flights.append({"price": f"{price} {f['price']['currency']}", "type": f"{len(f['itineraries'][0]['segments']) - 1} stop(s)"})
        return flights
    except Exception as e:
        st.error(f"Flight API error: {e}")
        return []


# ==============================================
# 🗞️ NEWS API
# ==============================================

@st.cache_data(ttl=1800)
def fetch_news(destination, max_articles=3):
    rss_url = f"https://news.google.com/rss/search?q={destination}+travel&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    return [{"title": e.title, "link": e.link} for e in feed.entries[:max_articles]]


# ==============================================
# 🎨 STREAMLIT UI
# ==============================================

st.set_page_config(page_title="Pack & Play", page_icon="🧳", layout="wide")
st.title("🧳 Pack & Play — AI Travel Chatbot")
st.caption("Plan smarter trips with real-time info and vegetarian-friendly options 🌱")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi 👋! Tell me your trip plan, e.g., 'Trip from Trichy to Chennai on Nov 9 for 3 days'."}]
if "last_trip" not in st.session_state:
    st.session_state.last_trip = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Type your travel plan..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        client, model = init_llm()
        nlp = init_spacy()
        try:
            trip = extract_trip_details(user_input, nlp)
            if not trip["destination"]:
                st.warning("Couldn't detect destination. Try again with a clearer message.")
            else:
                # 🌤 Weather card
                if w := get_weather(trip["destination"]):
                    loc, weather = w
                    cw = weather["current_weather"]
                    with st.container(border=True):
                        st.markdown(f"### 🌤 Weather in {trip['destination']}")
                        st.write(f"**{get_weather_emoji(cw['weathercode'])} {cw['temperature']}°C**")

                # 🏨 Hotels card
                hotels = get_hotels_by_city(trip["destination"], vegetarian="veg" in user_input.lower())
                if hotels:
                    st.markdown("### 🏨 Top Hotels")
                    for h in hotels:
                        st.markdown(f"- **{h['name']}** — {h['address']}")

                # ✈️ Flights card
                flights = get_flights_by_route(trip["source"], trip["destination"], trip["start_date"])
                if flights:
                    st.markdown("### ✈️ Available Flights")
                    for f in flights:
                        st.markdown(f"- {f['type']} • {f['price']}")

                # 🗞️ News card
                news = fetch_news(trip["destination"])
                if news:
                    st.markdown("### 🗞️ Latest Travel News")
                    for n in news:
                        st.markdown(f"[{n['title']}]({n['link']})")

        except Exception as e:
            st.error(f"Error: {e}")

