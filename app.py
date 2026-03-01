import streamlit as st
import requests
import math

st.set_page_config(page_title="Over 2.5 Model (FootyStats)", layout="centered")

API_KEY = st.secrets.get("FOOTYSTATS_API_KEY", "")
BASE_URL = "https://api.football-data-api.com"

st.title("Over 2.5 Betting Model (FootyStats)")
st.caption("Search league → pick season → pick teams → Poisson for P(Over 2.5).")

if not API_KEY:
    st.error("Missing FOOTYSTATS_API_KEY in Streamlit Secrets.")
    st.stop()

# ---------- helpers ----------
def api_get(path: str, params: dict | None = None) -> dict:
    params = params or {}
    params["key"] = API_KEY
    r = requests.get(f"{BASE_URL}/{path.lstrip('/')}", params=params, timeout=25)
    r.raise_for_status()
    return r.json()

def american_to_prob(o: int) -> float:
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)

def poisson_over_25(lam: float) -> float:
    p_le_2 = 0.0
    for k in range(0, 3):
        p_le_2 += (math.exp(-lam) * (lam ** k)) / math.factorial(k)
    return 1.0 - p_le_2

def safe_get(d: dict, keys: list[str], default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default

def to_int(x):
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        return int(x)
    except Exception:
        return None

def norm_status(s):
    s = str(s or "").lower().strip()
    return s

COMPLETE_STATUSES = {"complete", "completed", "finished", "ft"}

# ---------- load leagues ----------
query = st.text_input("Search League (examples: premier, mls, scotland)", value="premier")

try:
    league_list = api_get("league-list")
    leagues = league_list.get("data", [])
except Exception as e:
    st.error(f"Could not load league list: {e}")
    st.stop()

filtered = []
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

league_obj = st.selectbox("Select League", filtered, format_func=league_label)

# ---------- seasons ----------
seasons = league_obj.get("seasons") or league_obj.get("season") or []
if not isinstance(seasons, list) or not seasons:
    st.error("This league has no seasons list in the API response.")
    st.stop()

def season_year(s):
    return safe_get(s, ["year", "season", "season_year"], 0)

seasons_sorted = sorted(seasons, key=season_year, reverse=True)

def season_label(s: dict) -> str:
    y = safe_get(s, ["year", "season", "season_year"], "Unknown")
    sid = safe_get(s, ["id", "season_id", "league_id"], "")
    return f"{y} (season_id: {sid})"

season_obj = st.selectbox("Select Season", seasons_sorted, format_func=season_label)
season_id = safe_get(season_obj, ["id", "season_id", "league_id"])

if season_id is None:
    st.error("Could not read season_id from selected season.")
    st.stop()

st.success(f"Using season_id = {season_id}")

# ---------- teams ----------
try:
    teams_json = api_get("league-teams", {"league_id": season_id})
    teams = teams_json.get("data", [])
except Exception as e:
    st.error(f"Could not load teams for season_id {season_id}: {e}")
    st.stop()

if not teams:
    st.error("No teams returned for this season. Pick a different season.")
    st.stop()

team_names = sorted({t.get("name") for t in teams if t.get("name")})

home_team = st.selectbox("Home Team", team_names)
away_team = st.selectbox("Away Team", team_names, index=min(1, len(team_names) - 1))

odds = st.number_input("Over 2.5 American Odds", value=-110, step=5)
num_matches = st.slider("Number of recent completed matches to use", 5, 40, 10)

# ---------- calculate ----------
if st.button("Calculate"):
    try:
        matches_json = api_get("league-matches", {"league_id": season_id})
        matches = matches_json.get("data", [])
    except Exception as e:
        st.error(f"Could not load matches: {e}")
        st.stop()

    # filter completed matches involving either team
    completed = []
    for m in matches:
        status = norm_status(m.get("status"))
        if status not in COMPLETE_STATUSES:
            continue

        hn = m.get("home_name")
        an = m.get("away_name")
        if hn not in (home_team, away_team) and an not in (home_team, away_team):
            continue

        dt = to_int(m.get("date_unix")) or to_int(m.get("dateTimestamp")) or 0
        hg = to_int(m.get("homeGoalCount"))
        ag = to_int(m.get("awayGoalCount"))
        if hg is None or ag is None:
            continue

        completed.append((dt, hg + ag))

    # newest first
    completed.sort(key=lambda x: x[0], reverse=True)
    totals = [t for _, t in completed][:num_matches]

    if len(totals) < 5:
        st.error(
            "Not enough completed matches with usable scores in this season.\n"
            "Pick an older season (finished season) or increase matches if available."
        )
        with st.expander("Debug details"):
            st.write("Completed usable matches found:", len([t for _, t in completed]))
            st.write("Tip: Current seasons often have few 'complete' matches early on.")
        st.stop()

    avg_total = sum(totals) / len(totals)
    lam = avg_total
    model_prob = poisson_over_25(lam)
    market_prob = american_to_prob(int(odds))
    edge = model_prob - market_prob

    st.subheader("Model Output")
    st.write("Matches used:", len(totals))
    st.write("Avg total goals:", round(avg_total, 2))
    st.write("Expected Total Goals (λ):", round(lam, 2))
    st.write("Model P(Over 2.5):", f"{model_prob*100:.2f}%")
    st.write("Market implied:", f"{market_prob*100:.2f}%")
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
            "complete_statuses_accepted": sorted(list(COMPLETE_STATUSES)),
        })
