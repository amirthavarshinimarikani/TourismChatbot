# app.py - Pack & Play Travel Chatbot with RAG + APIs + LLM
# Run in VSCode: streamlit run app.py
# No secrets.toml needed: API keys hardcoded as placeholders—replace with your own!
# OpenRouter: Replace 'your-openrouter-api-key-here' below.
# Google Maps: Buses skipped unless you add your key (optional).

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

# LLM Setup (hardcoded key—replace!)
@st.cache_resource
def init_llm():
    api_key = "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"  # Replace with your actual OpenRouter key!
    if api_key == "your-openrouter-api-key-here":
        st.error("Replace 'your-openrouter-api-key-here' in app.py with your OpenRouter API key!")
        st.stop()
    base_url = 'https://openrouter.ai/api/v1'
    client = OpenAI(api_key=api_key, base_url=base_url)
    model = 'deepseek/deepseek-chat-v3-0324'
    return client, model

# SpaCy Setup
@st.cache_resource
def init_spacy():
    try:
        return spacy.load("en_core_web_trf")
    except:
        return spacy.load("en_core_web_sm")

# Helpers (unchanged)
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
def _word_to_int(s):
    s = s.lower().strip()
    if s.isdigit(): return int(s)
    return _WORD_NUM.get(s)

def _extract_duration_days(text):
    patterns = [r'for\s+(\d+)\s+days?', r'for\s+([a-zA-Z]+)\s+days?', r'(\d+)[-\s]?day', r'([a-zA-Z]+)[-\s]?day',
                r'for\s+(\d+)\s+nights?', r'for\s+([a-zA-Z]+)\s+nights?']
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            token = m.group(1)
            val = int(token) if token.isdigit() else _word_to_int(token)
            if val: return val
    return None

# Trip Parser (unchanged)
def extract_trip_details(text, nlp, prefer_future_dates=True):
    if not text: return {"source": None, "destination": None, "start_date": None, "return_date": None, "duration_days": None}
    doc = nlp(text)
    gpes = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")]
    src = dst = None
    text_stripped = text.strip()

    m = re.search(r'\bfrom\s+([A-Z][a-zA-Z\s]{1,60}?)\s+(?:to|->|-)\s+([A-Z][a-zA-Z\s]{1,60}?)\b', text_stripped)
    if m: src, dst = m.group(1).strip().title(), m.group(2).strip().title()
    else:
        m2 = re.search(r'\b([A-Z][a-zA-Z\s]{1,60}?)\s+(?:to|->|-)\s+([A-Z][a-zA-Z\s]{1,60}?)\b', text_stripped)
        if m2: src, dst = m2.group(1).strip().title(), m2.group(2).strip().title()

    if not src and gpes: src = gpes[0].title()
    if not dst and len(gpes) > 1: dst = gpes[1].title()
    elif len(gpes) == 1 and re.search(r'\b(to|visit|going to|trip to)\s+' + re.escape(gpes[0]), text, re.I):
        dst = gpes[0].title()

    dp_settings = {'PREFER_DATES_FROM': 'future'} if prefer_future_dates else {}
    try: sd = search_dates(text, settings=dp_settings) or []
    except: sd = []
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

# Weather API (unchanged)
@st.cache_data(ttl=1800)
def get_weather(city):
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(city)}&count=1&language=en&format=json"
    try: geo = requests.get(geo_url, timeout=6).json()
    except: return None
    if not geo.get("results"): return None
    result = geo["results"][0]
    lat, lon = result["latitude"], result["longitude"]
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&daily=temperature_2m_max,temperature_2m_min"
    try: weather = requests.get(weather_url, timeout=6).json()
    except: return result, None
    return result, weather

def get_weather_emoji(code):
    mapping = {0: "☀️ Sunny", 1: "☀️ Sunny", 2: "⛅ Partly Cloudy", 3: "⛅ Partly Cloudy", 45: "🌫 Foggy", 48: "🌫 Foggy",
               51: "🌦 Drizzle", 53: "🌦 Drizzle", 55: "🌦 Drizzle", 61: "🌧 Rainy", 63: "🌧 Rainy", 65: "🌧 Rainy",
               71: "❄️ Snowy", 73: "❄️ Snowy", 75: "❄️ Snowy", 95: "⛈ Thunderstorm", 96: "⛈ Thunderstorm", 99: "⛈ Thunderstorm"}
    return mapping.get(code, "☁️ Cloudy")

# Hotels API (unchanged)
@st.cache_data(ttl=3600)
def get_hotels_by_city(city):
    try:
        auth_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        auth_data = {"grant_type": "client_credentials", "client_id": "PdGKUOnybGoAzwjfP4xg93lD3pv2L89k", "client_secret": "z39TseJgdeG0w2Oh"}
        auth_res = requests.post(auth_url, data=auth_data)
        if auth_res.status_code != 200: return []
        token = auth_res.json()["access_token"]

        CITY_TO_IATA = {"Chennai": "MAA", "Trichy": "TRZ", "Goa": "GOI", "Delhi": "DEL", "Mumbai": "BOM", "Hyderabad": "HYD", "Kolkata": "CCU", "Bengaluru": "BLR", "Kodaikanal": "IXM"}  # Madurai for Kodaikanal
        iata = CITY_TO_IATA.get(city.title(), city[:3].upper())

        loc_url = f"https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city?cityCode={iata}"
        headers = {"Authorization": f"Bearer {token}"}
        loc_res = requests.get(loc_url, headers=headers).json()
        hotels = loc_res.get("data", [])
        if not hotels: return []

        results = []
        for h in hotels[:3]:
            name = h.get("name", "Unknown")
            address = ", ".join(filter(None, h.get("address", {}).get("lines", []) + [h.get("address", {}).get("cityName", "")]))
            hotel_id = h["hotelId"]
            offer_url = f"https://test.api.amadeus.com/v3/shopping/hotel-offers?hotelIds={hotel_id}"
            offer_res = requests.get(offer_url, headers=headers).json()
            if "errors" not in offer_res and offer_res.get("data"):
                offer = offer_res["data"][0]
                price = offer["offers"][0]["price"]["total"]
                results.append({"name": name, "price": f"{price} {offer['offers'][0]['price']['currency']}", "address": address})
        return results
    except Exception as e:
        st.error(f"Hotel API error: {e}")
        return []

# Flights API (unchanged)
@st.cache_data(ttl=3600)
def get_flights_by_route(source, destination, date=None):
    try:
        auth_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        auth_data = {"grant_type": "client_credentials", "client_id": "PdGKUOnybGoAzwjfP4xg93lD3pv2L89k", "client_secret": "z39TseJgdeG0w2Oh"}
        auth_res = requests.post(auth_url, data=auth_data)
        if auth_res.status_code != 200: return []
        token = auth_res.json()["access_token"]

        CITY_TO_IATA = {"Chennai": "MAA", "Trichy": "TRZ", "Goa": "GOI", "Delhi": "DEL", "Mumbai": "BOM", "Hyderabad": "HYD", "Kolkata": "CCU", "Bengaluru": "BLR", "Kodaikanal": "IXM", "Virudhunagar": "VGA"}
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
            flights.append({"type": f"{stops} stops", "price": f"{price} {item['price']['currency']}"})
        return flights
    except Exception as e:
        st.error(f"Flight API error: {e}")
        return []

# Bus API (hardcoded fallback—no secrets)


# RAG News (unchanged, destination-specific)
@st.cache_data(ttl=1800)
def fetch_news(destination, max_articles=5):
    rss_url = f"https://news.google.com/rss/search?q={destination}+travel+tourism&hl=en-IN&gl=IN&ceid=IN:en&when=7d"
    feed = feedparser.parse(rss_url)
    articles = []
    for entry in feed.entries[:max_articles]:
        articles.append({
            'title': entry.title,
            'summary': entry.get('summary', '')[:300] + '...' if entry.get('summary') else '',
            'link': entry.link
        })
    return articles

def simple_retrieve(query, articles, top_k=3):
    query_words = set(query.lower().split())
    scored = []
    for art in articles:
        art_words = set((art['title'] + ' ' + art['summary']).lower().split())
        score = len(query_words.intersection(art_words))
        scored.append((score, art))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [art for _, art in scored[:top_k]]

# API Aggregator (updated bus display)
def fetch_all_apis(trip, client, model):
    dest = trip['destination']
    src = trip['source']
    start_date = trip['start_date']
    context_parts = []

    # Weather
    with st.spinner("Fetching weather..."):
        w = get_weather(dest)
        if w:
            loc, weather = w
            cw = weather.get('current_weather', {})
            temp = cw.get('temperature', 'N/A')
            code = cw.get('weathercode', 0)
            forecast = weather.get('daily', {})
            max_temp = forecast.get('temperature_2m_max', [None])[0]
            context_parts.append(f"Weather in {dest}: Current {get_weather_emoji(code)} {temp}°C. Forecast high: {max_temp}°C.")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Temp", f"{temp}°C", f"{get_weather_emoji(code)}")
            with col2:
                st.metric("Forecast High", f"{max_temp}°C")

    # News (RAG)
    with st.spinner("Fetching news..."):
        news = fetch_news(dest)
        relevant_news = simple_retrieve(trip['destination'] or '', news)
        news_context = "\n".join([f"News: {n['title']} - {n['summary']} (Source: {n['link']})" for n in relevant_news])
        if news_context: context_parts.append(f"Recent News: {news_context}")
        if relevant_news:
            st.subheader("📰 Recent News")
            for n in relevant_news:
                st.write(f"**{n['title']}**")
                st.caption(n['summary'])
                st.caption(f"[Source]({n['link']})")
                st.divider()

    # Hotels
    with st.spinner("Fetching hotels..."):
        hotels = get_hotels_by_city(dest)
        if hotels:
            hotel_str = " | ".join([f"{h['name']} ({h['price']})" for h in hotels])
            context_parts.append(f"Hotel Options in {dest}: {hotel_str}")
            st.subheader("🏨 Hotel Options")
            for h in hotels:
                st.write(f"**{h['name']}** - 💵 {h['price']}")
                st.caption(f"📍 {h['address']}")

    # Flights
    if src and dest:
        with st.spinner("Fetching flights..."):
            flights = get_flights_by_route(src, dest, start_date)
            if flights:
                flight_str = " | ".join([f"{f['type']}: {f['price']}" for f in flights])
                context_parts.append(f"Flight Options from {src} to {dest}: {flight_str}")
                st.subheader("✈️ Flight Options")
                for f in flights:
                    st.write(f"**{f['type']}** - 💰 {f['price']}")

    return "\n\n".join(context_parts) if context_parts else "No additional data available."

# LLM Generation (unchanged)
def generate_travel_response(user_input, client, model, nlp):
    trip = extract_trip_details(user_input, nlp)
    
    st.subheader("✨ Parsed Trip Details")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**From:** {trip['source'] or 'Not detected'}")
        if trip['start_date']:
            st.success(f"**Start:** {trip['start_date'].strftime('%a, %d %b %Y')}")
        else:
            st.warning("**Start Date:** Not detected")
        st.write(f"**Duration:** {trip['duration_days'] or 'Not detected'} days")
    with col2:
        st.info(f"**To:** {trip['destination'] or 'Not detected'}")
        if trip['return_date']:
            st.success(f"**Return:** {trip['return_date'].strftime('%a, %d %b %Y')}")
        else:
            st.warning("**Return Date:** Not detected")

    if not trip['destination']:
        st.warning("No destination detected—try rephrasing!")
        return

    api_context = fetch_all_apis(trip, client, model)

    system_prompt = "You are a helpful travel assistant. Use the parsed trip details and provided context (news, weather, flights, hotels) to create a concise, personalized itinerary. Include tips, costs, and warnings. Structure: Overview, Itinerary, Recommendations."
    
    user_prompt = f"""
    User Plan: {user_input}
    
    Parsed Details: Source: {trip['source']}, Destination: {trip['destination']}, 
    Start: {trip['start_date'].strftime('%Y-%m-%d') if trip['start_date'] else 'N/A'}, 
    Return: {trip['return_date'].strftime('%Y-%m-%d') if trip['return_date'] else 'N/A'}, 
    Duration: {trip['duration_days']} days.
    
    Context from APIs & News: {api_context}
    
    Generate a helpful response.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    encoding = tiktoken.get_encoding("cl100k_base")
    prompt_tokens = len(encoding.encode("\n".join([m['content'] for m in messages])))
    if prompt_tokens > 3000:
        user_prompt = user_prompt[:2000]
        messages[1]['content'] = user_prompt
        st.warning("Prompt truncated for token limit.")

    with st.spinner("Generating personalized itinerary..."):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=800,
            temperature=0.7,
        )
    
    return response.choices[0].message.content

# Streamlit UI (no secrets checks)
st.set_page_config(page_title="Pack & Play - Travel Chatbot", page_icon="🧳", layout="wide")
st.title("🧳 Pack & Play — Travel Chatbot")
st.markdown("Type your travel plan in natural language. Example: `I am planning to go to Kodaikanal from Chennai on 31st Oct for 3 days`")

user_input = st.text_input("Enter travel plan:", placeholder="e.g., Trip Chennai to Kodaikanal 31 Oct - 2 Nov")

if st.button("🚀 Parse & Plan", type="primary"):
    if not user_input:
        st.warning("Please enter a travel plan first.")
    else:
        client, model = init_llm()
        nlp = init_spacy()
        full_response = generate_travel_response(user_input, client, model, nlp)
        
        st.subheader("🤖 Personalized Itinerary")
        st.markdown(full_response)

st.markdown("---")
st.caption("Powered by Streamlit, OpenRouter, Amadeus, Open-Meteo & Google News. Tip: Use future dates for best results!")
