import os
import difflib
import requests
import streamlit as st
from datetime import date
from math import exp, factorial

# ----------------------------
# Config
# ----------------------------
BASE_URL = "https://api.sportmonks.com/v3/football"
DEFAULT_LOOKBACK_DAYS = 200
LAST_N_MATCHES = 10

# Common league IDs (Sportmonks league IDs can vary by product/version; keep editable)
LEAGUES = {
    "Premier League (England)": 8,
    "Serie A (Italy)": 384,
    "La Liga (Spain)": 564,
    "Bundesliga (Germany)": 82,
    "Ligue 1 (France)": 301,
}

# ----------------------------
# Helpers
# ----------------------------
def get_api_token() -> str:
    # Streamlit Cloud -> st.secrets is best
    if "SPORTMONKS_API_TOKEN" in st.secrets:
        return st.secrets["SPORTMONKS_API_TOKEN"]
    # local env fallback
    return os.getenv("SPORTMONKS_API_TOKEN", "").strip()

def sm_get(path: str, token: str, params: dict | None = None) -> dict:
    if params is None:
        params = {}
    params = {**params, "api_token": token}

    url = f"{BASE_URL}/{path.lstrip('/')}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def find_season_id_for_year(token: str, league_id: int, season_year: int) -> int | None:
    """
    Uses league endpoint with include=seasons, then picks the season whose 'starting_at' year matches.
    Sportmonks recommends leagues + seasons/currentSeason for season discovery.  [oai_citation:1‡docs.sportmonks.com](https://docs.sportmonks.com/v3/welcome/what-can-you-do-with-sportmonks-data?utm_source=chatgpt.com)
    """
    data = sm_get(f"leagues/{league_id}", token, params={"include": "seasons"})
    seasons = (data.get("data") or {}).get("seasons") or data.get("data", {}).get("seasons")

    # Depending on API response shape, seasons might be in "included"
    if not seasons:
        # Try included
        included = data.get("included", [])
        seasons = [x for x in included if x.get("type") == "season"] or []

    # Normalize: each season should have starting_at or starting_at_timestamp, etc.
    for s in seasons:
        starting_at = s.get("starting_at") or ""
        # starting_at often like "2024-08-16"
        if isinstance(starting_at, str) and len(starting_at) >= 4:
            if int(starting_at[:4]) == int(season_year):
                return int(s["id"])

        # fallback: name might contain 2024/2025
        name = str(s.get("name", ""))
        if str(season_year) in name:
            return int(s["id"])

    return None

def get_teams_by_season(token: str, season_id: int) -> list[dict]:
    """
    Correct endpoint: Teams by Season ID  [oai_citation:2‡docs.sportmonks.com](https://docs.sportmonks.com/v3/tutorials-and-guides/tutorials/teams-players-coaches-and-referees/teams?utm_source=chatgpt.com)
    """
    data = sm_get(f"teams/seasons/{season_id}", token)
    return data.get("data", []) or []

def best_team_match(teams: list[dict], user_text: str) -> dict | None:
    """
    Fuzzy match against team 'name' and 'short_code' if present.
    """
    if not user_text.strip():
        return None

    names = []
    key_to_team = {}

    for t in teams:
        name = str(t.get("name", "")).strip()
        short_code = str(t.get("short_code", "")).strip()
        # Build searchable keys
        for k in [name, short_code]:
            if k:
                low = k.lower()
                names.append(low)
                key_to_team[low] = t

    query = user_text.strip().lower()

    # Exact contains shortcut
    exact = [k for k in names if k == query]
    if exact:
        return key_to_team[exact[0]]

    # Close matches
    matches = difflib.get_close_matches(query, names, n=5, cutoff=0.55)
    if matches:
        return key_to_team[matches[0]]

    # Contains fallback
    contains = [k for k in names if query in k]
    if contains:
        return key_to_team[contains[0]]

    return None

def american_to_implied_prob(odds: int) -> float:
    # American odds -> implied probability (no-vig not handled)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    else:
        return 100.0 / (odds + 100.0)

def poisson_pmf(k: int, lam: float) -> float:
    return (exp(-lam) * (lam ** k)) / factorial(k)

def prob_total_over_2_5(lam_total: float) -> float:
    # P(Total goals >= 3) = 1 - P(0,1,2)
    p0 = poisson_pmf(0, lam_total)
    p1 = poisson_pmf(1, lam_total)
    p2 = poisson_pmf(2, lam_total)
    return max(0.0, min(1.0, 1.0 - (p0 + p1 + p2)))

def get_last_fixtures_for_team(token: str, team_id: int, lookback_days: int, league_id: int | None) -> list[dict]:
    """
    Uses fixtures by date range for team (official endpoint).  [oai_citation:3‡docs.sportmonks.com](https://docs.sportmonks.com/v3/endpoints-and-entities/endpoints/fixtures/get-fixtures-by-date-range-for-team?utm_source=chatgpt.com)
    We then filter to finished matches and (optionally) a league.
    """
    end = date.today()
    start = end.fromordinal(end.toordinal() - lookback_days)

    data = sm_get(
        f"fixtures/between/{start.isoformat()}/{end.isoformat()}/{team_id}",
        token,
        params={"include": "participants;scores"}  # participants helps identify home/away  [oai_citation:4‡docs.sportmonks.com](https://docs.sportmonks.com/v3/tutorials-and-guides/tutorials/includes/participants?utm_source=chatgpt.com)
    )

    fixtures = data.get("data", []) or []

    # finished: state_id commonly indicates status; we’ll keep those with scores present
    filtered = []
    for fx in fixtures:
        # Optional league filter
        if league_id is not None:
            if int(fx.get("league_id", -1)) != int(league_id):
                continue

        # must have scores
        scores = fx.get("scores") or []
        if not scores:
            # some responses put scores in included; we already requested include=scores
            pass

        # keep if result_info present or scores exist
        if fx.get("result_info") or scores:
            filtered.append(fx)

    # sort newest first by starting_at
    def fx_sort_key(fx):
        return fx.get("starting_at") or ""

    filtered.sort(key=fx_sort_key, reverse=True)
    return filtered[:LAST_N_MATCHES]

def goals_for_against_from_fixture(fx: dict, team_id: int) -> tuple[int, int] | None:
    """
    Extract goals for/against from fixture 'scores' and 'participants'.
    This can vary by Sportmonks schema; we handle common cases.
    """
    participants = fx.get("participants") or []
    scores = fx.get("scores") or []

    # Identify home/away team IDs via participants meta.location (recommended)  [oai_citation:5‡docs.sportmonks.com](https://docs.sportmonks.com/v3/tutorials-and-guides/tutorials/includes/participants?utm_source=chatgpt.com)
    home_id = None
    away_id = None
    for p in participants:
        meta = p.get("meta") or {}
        loc = meta.get("location")
        if loc == "home":
            home_id = int(p.get("id"))
        elif loc == "away":
            away_id = int(p.get("id"))

    if home_id is None or away_id is None:
        return None

    # Try common score types:
    # Often there is a "current" or "FT" entry. We’ll take the max/min in scores where description mentions "FT" or "Full Time"
    # If not found, take last score item.
    ft = None
    for s in scores:
        desc = str(s.get("description", "")).lower()
        if "full" in desc or "ft" in desc:
            ft = s
            break
    if ft is None and scores:
        ft = scores[-1]

    if not ft:
        return None

    home_goals = ft.get("score", {}).get("home")
    away_goals = ft.get("score", {}).get("away")

    if home_goals is None or away_goals is None:
        return None

    home_goals = int(home_goals)
    away_goals = int(away_goals)

    if int(team_id) == int(home_id):
        return home_goals, away_goals
    if int(team_id) == int(away_id):
        return away_goals, home_goals

    return None

def avg_goals_last_n(fixtures: list[dict], team_id: int) -> tuple[float, float] | None:
    gf = []
    ga = []
    for fx in fixtures:
        res = goals_for_against_from_fixture(fx, team_id)
        if res is None:
            continue
        g_for, g_against = res
        gf.append(g_for)
        ga.append(g_against)

    if len(gf) < 3:
        return None
    return sum(gf) / len(gf), sum(ga) / len(ga)

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Over 2.5 Betting Model (Sportmonks)", layout="centered")
st.title("Over 2.5 Betting Model (Sportmonks)")
st.caption("Select league + season, enter teams + Over 2.5 American odds. Uses last 10 finished fixtures to estimate total goals, then Poisson for P(Over 2.5).")

token = get_api_token()
if not token:
    st.error("Missing SPORTMONKS_API_TOKEN. Add it in Streamlit: App → Settings → Secrets, or set env var locally.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    league_name = st.selectbox("League", list(LEAGUES.keys()))
    league_id = LEAGUES[league_name]
with col2:
    season_year = st.number_input("Season start year (YYYY)", min_value=2000, max_value=2100, value=2024, step=1)

home_team_text = st.text_input("Home Team", value="Bournemouth")
away_team_text = st.text_input("Away Team", value="Sunderland")
odds = st.number_input("Over 2.5 American Odds", value=-110, step=1)

lookback_days = st.slider("Lookback window (days) for fixtures", 60, 365, DEFAULT_LOOKBACK_DAYS)

if st.button("Calculate"):
    try:
        season_id = find_season_id_for_year(token, league_id, int(season_year))
        if not season_id:
            st.error("Could not find season_id for that league + year. Try a different year or check your league ID mapping.")
            st.stop()

        teams = get_teams_by_season(token, int(season_id))
        if not teams:
            st.error("No teams returned for that season. This usually means your plan doesn’t cover it or the season_id is wrong.")
            st.stop()

        home_team = best_team_match(teams, home_team_text)
        away_team = best_team_match(teams, away_team_text)

        if not home_team:
            st.error(f"Home team not found in that season: {home_team_text}")
            st.stop()
        if not away_team:
            st.error(f"Away team not found in that season: {away_team_text}")
            st.stop()

        st.write(f"Matched Home: **{home_team.get('name')}** (team_id: {home_team.get('id')})")
        st.write(f"Matched Away: **{away_team.get('name')}** (team_id: {away_team.get('id')})")

        # Fetch last fixtures (date range endpoint)
        home_fx = get_last_fixtures_for_team(token, int(home_team["id"]), int(lookback_days), league_id)
        away_fx = get_last_fixtures_for_team(token, int(away_team["id"]), int(lookback_days), league_id)

        home_avgs = avg_goals_last_n(home_fx, int(home_team["id"]))
        away_avgs = avg_goals_last_n(away_fx, int(away_team["id"]))

        if not home_avgs or not away_avgs:
            st.error("Not enough finished fixtures with scores found to compute averages. Increase lookback days or verify league/season.")
            st.stop()

        home_gf, home_ga = home_avgs
        away_gf, away_ga = away_avgs

        st.subheader("Recent Form (Last 10 finished fixtures in selected league)")
        st.write(f"**{home_team.get('name')}** — Avg Goals For: **{home_gf:.2f}**, Avg Goals Against: **{home_ga:.2f}**")
        st.write(f"**{away_team.get('name')}** — Avg Goals For: **{away_gf:.2f}**, Avg Goals Against: **{away_ga:.2f}**")

        # Simple total-goals expectation:
        # Expected home goals ~ average(home GF, away GA)
        # Expected away goals ~ average(away GF, home GA)
        exp_home = (home_gf + away_ga) / 2.0
        exp_away = (away_gf + home_ga) / 2.0
        lam_total = max(0.05, exp_home + exp_away)

        p_model = prob_total_over_2_5(lam_total)
        p_market = american_to_implied_prob(int(odds))
        edge = p_model - p_market

        st.subheader("Model Output")
        st.write(f"Expected Total Goals (λ): **{lam_total:.2f}**")
        st.write(f"Model Probability Over 2.5: **{p_model*100:.2f}%**")
        st.write(f"Market Implied Probability: **{p_market*100:.2f}%**")
        st.write(f"Edge (Model − Market): **{edge*100:.2f}%**")

        if edge > 0.02:
            st.success("VALUE (positive edge)")
        else:
            st.warning("NO EDGE")

    except requests.HTTPError as e:
        # Important: do NOT print token or full URL with token
        st.error(f"HTTP error from Sportmonks: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
