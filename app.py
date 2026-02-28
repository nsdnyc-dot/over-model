import streamlit as st
import requests

BASE_URL = "https://api.sportmonks.com/v3/football"

st.title("Sportmonks API Test")

api_key = st.secrets.get("SPORTMONKS_API_KEY")

if not api_key:
    st.error("API key not found in Streamlit Secrets.")
    st.stop()

team_name = st.text_input("Enter team name", value="Manchester City")

if st.button("Search"):
    url = f"{BASE_URL}/teams/search/{team_name}"
    params = {"api_token": api_key}

    r = requests.get(url, params=params, timeout=30)
    st.write("Status:", r.status_code)
    st.write(r.json())
