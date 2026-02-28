import math
import requests
import streamlit as st

BASE_URL = "https://api.sportmonks.com/v3/football"

# ---------- odds + poisson ----------
def american_to_prob(odds: float) -> float:
    odds = float(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)

def poisson_over_25(lam: float) -> float:
    # P(X >= 3) for Poisson(lam)
    p0 = math.exp(-lam)
    p1 = lam * p0
    p2 = (lam**2 / 2.0) * p0
    return 1.0 - (p0 + p1 + p2)

# ---------- Sportmonks helpers ----------
def sm_get(path: str, api_key: str, params: dict | None = None):
    if params is None:
        params = {}
    params["api_token"] = api_key
    url = f"{BASE_URL}/{path}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# LEAGUE IDS (Sportmonks standard)
LEAGUES = {
    "Premier League (England)": 8,
    "Championship (England)": 9,
    "La Liga (Spain)": 564,
    "Serie A (Italy)": 384,
    "Bundesliga (Germany)": 82,
    "Ligue 1 (France)": 301
}

def find_team_in_league(api_key: str, league_id: int, team_name: str):
    q = team_name.strip().lower()

    data = sm_get(f"leagues/{league_id}/teams", api_key)
    teams = data.get("data", [])

    for t in teams:
        name = (t.get("name") or "").lower()
        if q in name:
            return t["id"], t["name"]

    st.error(f"Team not found in selected league: {team_name}")
    st.stop()

def last_n_fixtures(api_key: str, team_id: int, n: int = 10):
    # Pull last N fixtures for team (with scores)
    params = {
        "filters": f"team:{team_id}",
        "per_page": n,
        "sort": "-starting_at",
        "include": "scores"
    }
    data = sm_get("fixtures", api_key, params=params)
    return data.get("data", [])

def avg_goals_from_fixtures(fixtures: list, team_id: int):
    gf = 0.0
    ga = 0.0
    games = 0

    for fx in fixtures:
        scores = fx.get("scores", [])
        if not scores:
            continue

        home_id = fx.get("home_team_id")
        away_id = fx.get("away_team_id")

        home_goals = None
        away_goals = None

        # Prefer FT/final if available
        for s in scores:
            desc = (s.get("description") or "").upper()
            if "FT" in desc or "FINAL" in desc:
                part = s.get("score", {}).get("participant")
                goals = s.get("score", {}).get("goals")
                if part == "home":
                    home_goals = goals
                elif part == "away":
                    away_goals = goals

        # Fallback: any score
        if home_goals is None or away_goals is None:
            for s in scores:
                part = s.get("score", {}).get("participant")
                goals = s.get("score", {}).get("goals")
                if part == "home":
                    home_goals = goals
                elif part == "away":
                    away_goals = goals

        if home_goals is None or away_goals is None:
            continue

        games += 1
        home_goals = float(home_goals)
        away_goals = float(away_goals)

        if team_id == home_id:
            gf += home_goals
            ga += away_goals
        elif team_id == away_id:
            gf += away_goals
            ga += home_goals

    if games == 0:
        return 0.0, 0.0, 0
    return gf / games, ga / games, games

# ---------- Streamlit App ----------
st.set_page_config(page_title="Over 2.5 Model (Sportmonks)", layout="centered")
st.title("Over 2.5 Betting Model (Sportmonks)")

api_key = st.secrets.get("SPORTMONKS_API_KEY")
if not api_key:
    st.error("Missing SPORTMONKS_API_KEY in Streamlit Secrets.")
    st.stop()

league_name = st.selectbox("League", list(LEAGUES.keys()), index=0)
league_id = LEAGUES[league_name]

home_team = st.text_input("Home Team", value="Bournemouth")
away_team = st.text_input("Away Team", value="Sunderland")
odds = st.number_input("Over 2.5 American Odds", value=-110, step=1)

if st.button("Calculate"):
    home_id, home_name = find_team_in_league(api_key, league_id, home_team)
    away_id, away_name = find_team_in_league(api_key, league_id, away_team)

    home_fx = last_n_fixtures(api_key, home_id, n=10)
    away_fx = last_n_fixtures(api_key, away_id, n=10)

    home_gf, home_ga, home_n = avg_goals_from_fixtures(home_fx, home_id)
    away_gf, away_ga, away_n = avg_goals_from_fixtures(away_fx, away_id)

    # Simple lambda estimate using recent GF/GA
    if home_n == 0 or away_n == 0:
        st.error("Not enough recent fixtures with scores returned to calculate.")
        st.stop()

    home_exp = (home_gf + away_ga) / 2
    away_exp = (away_gf + home_ga) / 2
    lam = home_exp + away_exp

    model_prob = poisson_over_25(lam)
    market_prob = american_to_prob(odds)
    edge = model_prob - market_prob

    st.subheader("Recent form (last 10 fixtures, goals-based)")
    st.write(f"**{home_name}** avg GF: **{home_gf:.2f}**, avg GA: **{home_ga:.2f}** (games used: {home_n})")
    st.write(f"**{away_name}** avg GF: **{away_gf:.2f}**, avg GA: **{away_ga:.2f}** (games used: {away_n})")

    st.subheader("Model output")
    st.write(f"λ (expected total goals): **{lam:.2f}**")
    st.write(f"Model P(Over 2.5): **{model_prob:.2%}**")
    st.write(f"Market implied P: **{market_prob:.2%}**")
    st.write(f"Edge: **{edge:.2%}**")

    if edge > 0.03:
        st.success("VALUE BET (edge > 3%)")
    else:
        st.error("NO EDGE")
