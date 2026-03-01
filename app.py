import streamlit as st
import requests
import math

st.set_page_config(page_title="Over 2.5 Model (FootyStats)", layout="centered")

API_KEY = st.secrets["FOOTYSTATS_API_KEY"]
BASE_URL = "https://api.football-data-api.com"

st.title("Over 2.5 Betting Model (FootyStats)")

# --------- Step 1: Search League ----------
search = st.text_input("Search League (example: premier, mls, scotland)")

league_id = None

if search:
    res = requests.get(f"{BASE_URL}/league-list?key={API_KEY}")
    leagues = res.json()["data"]

    filtered = [l for l in leagues if search.lower() in l["name"].lower()]

    if filtered:
        choice = st.selectbox("Select League Season", filtered, format_func=lambda x: f"{x['name']} ({x['season']})")
        league_id = choice["id"]
    else:
        st.warning("No leagues found")

# --------- Step 2: Load Teams ----------
home_team = None
away_team = None

if league_id:
    team_res = requests.get(f"{BASE_URL}/league-teams?key={API_KEY}&league_id={league_id}")
    teams = team_res.json()["data"]

    team_names = [t["name"] for t in teams]

    home_team = st.selectbox("Home Team", team_names)
    away_team = st.selectbox("Away Team", team_names)

# --------- Step 3: Odds + Matches ----------
odds = st.number_input("Over 2.5 American Odds", value=-110)
num_matches = st.slider("Number of Matches to Use", 5, 30, 10)

# --------- Step 4: Calculation ----------
def american_to_prob(o):
    if o > 0:
        return 100 / (o + 100)
    else:
        return -o / (-o + 100)

def poisson_prob(lam):
    prob = 0
    for i in range(3):
        prob += (math.exp(-lam) * lam**i) / math.factorial(i)
    return 1 - prob

if st.button("Calculate") and league_id and home_team and away_team:

    matches_res = requests.get(f"{BASE_URL}/league-matches?key={API_KEY}&league_id={league_id}")
    matches = matches_res.json()["data"]

    team_goals = []

    for m in matches:
        if m["status"] == "complete":
            if m["home_name"] == home_team:
                team_goals.append(m["homeGoalCount"])
            if m["away_name"] == home_team:
                team_goals.append(m["awayGoalCount"])

    team_goals = team_goals[:num_matches]

    if len(team_goals) < 5:
        st.error("Not enough completed matches.")
    else:
        avg_goals = sum(team_goals) / len(team_goals)
        lam = avg_goals * 2

        model_prob = poisson_prob(lam)
        market_prob = american_to_prob(odds)
        edge = model_prob - market_prob

        st.subheader("Model Output")
        st.write("Expected Total Goals (λ):", round(lam,2))
        st.write("Model Probability Over 2.5:", round(model_prob*100,2),"%")
        st.write("Market Implied Probability:", round(market_prob*100,2),"%")
        st.write("Edge:", round(edge*100,2),"%")

        if edge > 0.03:
            st.success("EDGE FOUND")
        else:
            st.warning("No Edge")
