import streamlit as st
import requests
import math

BASE_URL = "https://api.football-data-api.com"

st.set_page_config(page_title="Over 2.5 Model (FootyStats)", layout="centered")
st.title("Over 2.5 Betting Model (FootyStats)")

# Get API key from Streamlit Secrets
API_KEY = st.secrets.get("FOOTYSTATS_API_KEY")

if not API_KEY:
    st.error("Missing FOOTYSTATS_API_KEY in Streamlit Secrets.")
    st.stop()


def american_to_probability(odds):
    if odds < 0:
        return (-odds) / ((-odds) + 100)
    else:
        return 100 / (odds + 100)


def poisson_over_2_5(lam):
    p0 = math.exp(-lam)
    p1 = p0 * lam
    p2 = p1 * lam / 2
    return 1 - (p0 + p1 + p2)


league_id = st.number_input("League ID (season id)", value=1625)
home_team_name = st.text_input("Home Team Name", "Manchester City")
away_team_name = st.text_input("Away Team Name", "Liverpool")
odds = st.number_input("Over 2.5 American Odds", value=-110)
lookback_matches = st.slider("Number of Matches to Use", 5, 30, 10)

if st.button("Calculate"):

    # Get teams
    teams_url = f"{BASE_URL}/league-teams"
    teams_params = {"key": API_KEY, "league_id": league_id}
    teams_response = requests.get(teams_url, params=teams_params)
    teams_data = teams_response.json()

    if not teams_data.get("success"):
        st.error(teams_data.get("message", "API Error"))
        st.stop()

    teams = teams_data.get("data", [])

    def find_team(name):
        for t in teams:
            if name.lower() in t.get("cleanName", "").lower():
                return t
        return None

    home_team = find_team(home_team_name)
    away_team = find_team(away_team_name)

    if not home_team:
        st.error("Home team not found in this league.")
        st.stop()

    if not away_team:
        st.error("Away team not found in this league.")
        st.stop()

    st.write(f"Home ID: {home_team['id']}")
    st.write(f"Away ID: {away_team['id']}")

    # Get matches
    matches_url = f"{BASE_URL}/league-matches"
    matches_params = {"key": API_KEY, "league_id": league_id}
    matches_response = requests.get(matches_url, params=matches_params)
    matches_data = matches_response.json()

    if not matches_data.get("success"):
        st.error(matches_data.get("message", "API Error"))
        st.stop()

    matches = matches_data.get("data", [])

    completed = [
        m for m in matches
        if m.get("status") == "complete"
        and isinstance(m.get("totalGoalCount"), int)
    ]

    if len(completed) == 0:
        st.error("No completed matches found.")
        st.stop()

    # Use recent matches
    completed = sorted(completed, key=lambda x: x.get("date_unix", 0), reverse=True)
    recent = completed[:lookback_matches]

    total_goals = [m["totalGoalCount"] for m in recent]
    lam = sum(total_goals) / len(total_goals)

    model_prob = poisson_over_2_5(lam)
    market_prob = american_to_probability(odds)
    edge = model_prob - market_prob

    st.subheader("Model Output")
    st.write(f"Expected Goals (λ): {lam:.2f}")
    st.write(f"Model Probability Over 2.5: {model_prob*100:.2f}%")
    st.write(f"Market Implied Probability: {market_prob*100:.2f}%")
    st.write(f"Edge: {edge*100:.2f}%")

    if edge > 0.03:
        st.success("EDGE FOUND")
    else:
        st.warning("No Edge")
