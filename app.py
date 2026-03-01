import streamlit as st
import requests
import math

st.set_page_config(page_title="Over 2.5 Model (FootyStats)", layout="centered")

API_KEY = st.secrets.get("FOOTYSTATS_API_KEY", "")
BASE_URL = "https://api.football-data-api.com"

st.title("Over 2.5 Betting Model (FootyStats)")
st.caption("Search league → pick a season → pick teams → calculate O2.5 with Poisson.")

if not API_KEY:
    st.error("Missing FOOTYSTATS_API_KEY in Streamlit Secrets.")
    st.stop()

# ---------------- Utils ----------------
def api_get(path: str, params: dict | None = None) -> dict:
    params = params or {}
    params["key"] = API_KEY
    r = requests.get(f"{BASE_URL}/{path.lstrip('/')}", params=params, timeout=25)
    r.raise_for_status()
    j = r.json()
    return j

def american_to_prob(o: int) -> float:
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)

def poisson_over_25(lam: float) -> float:
    # P(X >= 3) = 1 - (P0 + P1 + P2)
    p_le_2 = 0.0
    for k in range(0, 3):
        p_le_2 += (math.exp(-lam) * (lam ** k)) / math.factorial(k)
    return 1.0 - p_le_2

def safe_get(d: dict, keys: list[str], default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default

# ---------------- Step 1: Search league ----------------
query = st.text_input("Search League (examples: premier, mls, scotland)", value="premier")

league_obj = None
season_obj = None
season_id = None

try:
    league_list = api_get("league-list")
    leagues = league_list.get("data", [])
except Exception as e:
    st.error(f"Could not load league list: {e}")
    st.stop()

filtered = []
if query.strip():
    q = query.lower().strip()
    for L in leagues:
        name = safe_get(L, ["name", "league_name", "competition_name"], "")
        if name and q in str(name).lower():
            filtered.append(L)

if not filtered:
    st.warning("No leagues found. Try a different search term.")
    st.stop()

def league_label(L: dict) -> str:
    name = safe_get(L, ["name", "league_name", "competition_name"], "Unknown League")
    country = safe_get(L, ["country", "country_name"], "")
    return f"{name}" + (f" ({country})" if country else "")

league_obj = st.selectbox(
    "Select League",
    filtered,
    format_func=league_label
)

# ---------------- Step 2: Pick season from that league ----------------
seasons = league_obj.get("seasons") or league_obj.get("season") or []
if not isinstance(seasons, list) or len(seasons) == 0:
    st.error("This league object has no seasons list in the API response.")
    st.stop()

# sort seasons newest first if year exists
def season_year(s):
    return safe_get(s, ["year", "season", "season_year"], 0)

seasons_sorted = sorted(seasons, key=season_year, reverse=True)

def season_label(s: dict) -> str:
    y = safe_get(s, ["year", "season", "season_year"], "Unknown")
    sid = safe_get(s, ["id", "season_id", "league_id"], "")
    return f"{y} (season_id: {sid})"

season_obj = st.selectbox(
    "Select Season",
    seasons_sorted,
    format_func=season_label
)

season_id = safe_get(season_obj, ["id", "season_id", "league_id"])
if season_id is None:
    st.error("Could not read season_id from selected season.")
    st.stop()

st.success(f"Using season_id = {season_id}")

# ---------------- Step 3: Load teams from that season ----------------
try:
    teams_json = api_get("league-teams", {"league_id": season_id})
    teams = teams_json.get("data", [])
except Exception as e:
    st.error(f"Could not load teams for season_id {season_id}: {e}")
    st.stop()

if not teams:
    st.error("No teams returned for this season. Pick a different season.")
    st.stop()

team_names = [t.get("name") for t in teams if t.get("name")]
team_names = sorted(list(set(team_names)))

home_team = st.selectbox("Home Team", team_names)
away_team = st.selectbox("Away Team", team_names, index=min(1, len(team_names)-1))

odds = st.number_input("Over 2.5 American Odds", value=-110, step=5)
num_matches = st.slider("Number of recent completed matches to use", 5, 40, 10)

# ---------------- Step 4: Calculate ----------------
if st.button("Calculate"):
    try:
        matches_json = api_get("league-matches", {"league_id": season_id})
        matches = matches_json.get("data", [])
    except Exception as e:
        st.error(f"Could not load matches: {e}")
        st.stop()

    # keep completed matches involving either team
    completed = []
    for m in matches:
        if str(m.get("status", "")).lower() != "complete":
            continue
        if m.get("home_name") in [home_team, away_team] or m.get("away_name") in [home_team, away_team]:
            completed.append(m)

    # build recent totals list (total goals in match)
    totals = []
    for m in completed:
        hg = m.get("homeGoalCount")
        ag = m.get("awayGoalCount")
        if isinstance(hg, int) and isinstance(ag, int):
            totals.append(hg + ag)

    totals = totals[:num_matches]

    if len(totals) < 5:
        st.error("Not enough completed matches with usable scores in this season. Choose another season or league.")
        st.stop()

    avg_total = sum(totals) / len(totals)
    lam = avg_total  # expected total goals
    model_prob = poisson_over_25(lam)
    market_prob = american_to_prob(int(odds))
    edge = model_prob - market_prob

    st.subheader("Model Output")
    st.write("Matches used:", len(totals))
    st.write("Avg total goals (from matches used):", round(avg_total, 2))
    st.write("Expected Total Goals (λ):", round(lam, 2))
    st.write("Model Probability Over 2.5:", f"{model_prob*100:.2f}%")
    st.write("Market Implied Probability:", f"{market_prob*100:.2f}%")
    st.write("Edge (Model − Market):", f"{edge*100:.2f}%")

    if edge > 0.03:
        st.success("EDGE ✅ (model > market by 3%+)")
    else:
        st.warning("NO EDGE")

    with st.expander("Debug details"):
        st.json({
            "season_id_used": season_id,
            "home_team": home_team,
            "away_team": away_team,
            "odds": odds,
            "totals_sample": totals[:10],
        })
