import os
import math
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional

import requests
import streamlit as st

API_BASE = "https://api.football-data-api.com"


# ----------------------------
# Odds + Poisson helpers
# ----------------------------
def american_to_implied_prob(odds: float) -> float:
    if odds == 0:
        return float("nan")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def poisson_p_over_2_5(lmbda: float) -> float:
    if lmbda <= 0:
        return 0.0
    p0 = math.exp(-lmbda)
    p1 = p0 * lmbda
    p2 = p1 * lmbda / 2.0
    return max(0.0, 1.0 - (p0 + p1 + p2))


# ----------------------------
# API helpers
# ----------------------------
def get_api_key() -> Optional[str]:
    if "FOOTYSTATS_API_KEY" in st.secrets:
        return st.secrets["FOOTYSTATS_API_KEY"]
    return os.getenv("FOOTYSTATS_API_KEY")


def fs_get(path: str, params: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    url = f"{API_BASE}/{path}"
    r = requests.get(url, params=params, timeout=timeout)
    if not r.ok:
        try:
            payload = r.json()
        except Exception:
            payload = {"text": r.text[:500]}
        raise requests.HTTPError(f"{r.status_code} {r.reason} for {r.url} | {payload}")
    return r.json()


def _as_list(payload: Any) -> List[Dict[str, Any]]:
    """
    Normalize FootyStats responses to a list of dicts.
    Some endpoints return:
      - {"data": [...]}
      - {"response": [...]}
      - {"data": {"items": [...]}}
      - [...]
    """
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return [x for x in payload["data"] if isinstance(x, dict)]
        if isinstance(payload.get("response"), list):
            return [x for x in payload["response"] if isinstance(x, dict)]

        data = payload.get("data")
        if isinstance(data, dict):
            for k in ("items", "leagues", "seasons", "results"):
                if isinstance(data.get(k), list):
                    return [x for x in data[k] if isinstance(x, dict)]

    return []


def _to_int(val: Any) -> Optional[int]:
    if isinstance(val, int):
        return val
    if isinstance(val, float) and val.is_integer():
        return int(val)
    if isinstance(val, str):
        s = val.strip()
        if s.isdigit():
            return int(s)
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_league_list(api_key: str) -> List[Dict[str, Any]]:
    payload = fs_get("league-list", {"key": api_key})
    return _as_list(payload)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_league_teams(api_key: str, season_id: int) -> List[Dict[str, Any]]:
    payload = fs_get("league-teams", {"key": api_key, "season_id": season_id})
    return _as_list(payload)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_league_matches_all(api_key: str, season_id: int, max_per_page: int = 1000) -> List[Dict[str, Any]]:
    all_matches: List[Dict[str, Any]] = []
    page = 1

    while True:
        payload = fs_get("league-matches", {"key": api_key, "season_id": season_id, "page": page, "max_per_page": max_per_page})
        matches = _as_list(payload)

        if not matches:
            break

        all_matches.extend(matches)

        if len(matches) < max_per_page:
            break

        page += 1
        if page > 50:
            break

    return all_matches


def extract_match_time_unix(m: Dict[str, Any]) -> Optional[int]:
    for key in ("date_unix", "timestamp", "time_unix", "kickoff_unix", "unix"):
        v = m.get(key)
        t = _to_int(v)
        if t and t > 0:
            return t
    return None


def match_is_complete(m: Dict[str, Any]) -> bool:
    status = (m.get("status") or "").lower()
    return status in {"complete", "completed", "finished"}


def get_team_matches_last_n(matches: List[Dict[str, Any]], team_id: int, n: int, lookback_days: int) -> List[Dict[str, Any]]:
    now_unix = int(datetime.now(timezone.utc).timestamp())
    min_unix = now_unix - lookback_days * 86400

    filtered = []
    for m in matches:
        if not match_is_complete(m):
            continue

        if m.get("homeID") != team_id and m.get("awayID") != team_id:
            continue

        t = extract_match_time_unix(m)
        if t is None or t < min_unix:
            continue

        hg = m.get("homeGoals")
        ag = m.get("awayGoals")
        if not isinstance(hg, (int, float)) or not isinstance(ag, (int, float)):
            continue

        filtered.append((t, m))

    filtered.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in filtered[:n]]


def last_n_no_time_filter(matches: List[Dict[str, Any]], team_id: int, n: int) -> List[Dict[str, Any]]:
    arr = []
    for m in matches:
        if not match_is_complete(m):
            continue
        if m.get("homeID") != team_id and m.get("awayID") != team_id:
            continue
        t = extract_match_time_unix(m)
        if t is None:
            continue
        hg = m.get("homeGoals")
        ag = m.get("awayGoals")
        if not isinstance(hg, (int, float)) or not isinstance(ag, (int, float)):
            continue
        arr.append((t, m))
    arr.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in arr[:n]]


def goals_for_against_from_matches(team_id: int, team_matches: List[Dict[str, Any]]) -> Tuple[float, float, int]:
    gf = 0.0
    ga = 0.0
    count = 0
    for m in team_matches:
        hg = float(m["homeGoals"])
        ag = float(m["awayGoals"])
        if m.get("homeID") == team_id:
            gf += hg
            ga += ag
        else:
            gf += ag
            ga += hg
        count += 1
    return gf, ga, count


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Over 2.5 Betting Model (FootyStats)", layout="centered")
st.title("Over 2.5 Betting Model (FootyStats)")
st.caption("Search league → pick season_id → pick teams → Poisson for P(Over 2.5).")

api_key = get_api_key()
if not api_key:
    st.error("Missing FOOTYSTATS_API_KEY. Add it in Streamlit Secrets (App → Settings → Secrets) or set env var FOOTYSTATS_API_KEY.")
    st.stop()

with st.spinner("Loading league list..."):
    leagues = fetch_league_list(api_key)

if not leagues:
    st.error("league-list returned no leagues. Either your API key has no access to this endpoint, or the response format changed.")
    st.stop()


def league_label(x: Dict[str, Any]) -> str:
    name = x.get("name") or x.get("league_name") or x.get("competition_name") or "Unknown League"
    country = x.get("country") or x.get("country_name") or ""
    season = x.get("season") or x.get("season_name") or x.get("year") or ""
    sid = x.get("season_id") or x.get("id") or ""
    parts = [p for p in [country, name, str(season)] if p]
    return f"{' • '.join(parts)}  (season_id: {sid})"


# Build season items (accept int OR numeric string IDs)
season_items: List[Dict[str, Any]] = []
for x in leagues:
    sid = _to_int(x.get("season_id") or x.get("id"))
    if sid is not None:
        season_items.append(x)

query = st.text_input("Search league (type: premier, mls, scotland, etc.)", value="premier")
filtered = [x for x in season_items if query.lower() in league_label(x).lower()] if query else season_items

# ✅ Critical guard: if none, STOP and show helpful message
if not filtered:
    st.warning("No options to select. Try different search words: 'england', 'epl', 'premier league', 'mls', 'usa', 'scotland'.")
    st.stop()

choice = st.selectbox("League season", filtered, format_func=league_label)

# ✅ Another guard (paranoia)
if not isinstance(choice, dict):
    st.error("League season selection failed. Try refreshing the page.")
    st.stop()

season_id = _to_int(choice.get("season_id") or choice.get("id"))
if season_id is None:
    st.error("Could not parse season_id from selected league item.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    last_n = st.slider("Use last N completed matches per team", min_value=5, max_value=25, value=10, step=1)
with col2:
    lookback_days = st.slider("Lookback window (days)", min_value=30, max_value=730, value=365, step=10)

with st.spinner("Loading teams for this season..."):
    teams = fetch_league_teams(api_key, season_id)

if not teams:
    st.error("No teams returned for that season_id. Pick a different league season.")
    st.stop()

team_options = []
for t in teams:
    tid = _to_int(t.get("id") or t.get("team_id"))
    nm = t.get("name") or t.get("english_name") or t.get("full_name")
    if tid is not None and nm:
        team_options.append({"id": tid, "name": str(nm)})

team_options = sorted(team_options, key=lambda x: x["name"].lower())

home_team = st.selectbox("Home team", team_options, format_func=lambda x: x["name"])
away_team = st.selectbox("Away team", team_options, format_func=lambda x: x["name"], index=min(1, len(team_options) - 1))
odds = st.number_input("Over 2.5 American odds", value=-110, step=1)

run = st.button("Calculate")

if run:
    if home_team["id"] == away_team["id"]:
        st.error("Home and Away team cannot be the same.")
        st.stop()

    with st.spinner("Loading matches for this season..."):
        matches = fetch_league_matches_all(api_key, season_id)

    if not matches:
        st.error("No matches returned for this season. Pick a different league season.")
        st.stop()

    home_ms = get_team_matches_last_n(matches, home_team["id"], last_n, lookback_days)
    away_ms = get_team_matches_last_n(matches, away_team["id"], last_n, lookback_days)

    if len(home_ms) < 5:
        home_ms = last_n_no_time_filter(matches, home_team["id"], last_n)
    if len(away_ms) < 5:
        away_ms = last_n_no_time_filter(matches, away_team["id"], last_n)

    home_gf, home_ga, home_cnt = goals_for_against_from_matches(home_team["id"], home_ms)
    away_gf, away_ga, away_cnt = goals_for_against_from_matches(away_team["id"], away_ms)

    if home_cnt < 3 or away_cnt < 3:
        st.error("Not enough completed matches with usable scores for one or both teams. Try a different season or increase lookback days.")
        st.stop()

    home_gf_avg = home_gf / home_cnt
    home_ga_avg = home_ga / home_cnt
    away_gf_avg = away_gf / away_cnt
    away_ga_avg = away_ga / away_cnt

    exp_home_goals = (home_gf_avg + away_ga_avg) / 2.0
    exp_away_goals = (away_gf_avg + home_ga_avg) / 2.0
    lmbda_total = max(0.0, exp_home_goals + exp_away_goals)

    model_p = poisson_p_over_2_5(lmbda_total)
    market_p = american_to_implied_prob(float(odds))
    edge = model_p - market_p

    st.subheader("Model Output")
    st.write(f"**Selected season_id:** {season_id}")
    st.write(f"**Expected total goals (λ):** {lmbda_total:.2f}")
    st.write(f"**Model P(Over 2.5):** {model_p*100:.2f}%")
    st.write(f"**Market implied P:** {market_p*100:.2f}%")
    st.write(f"**Edge (Model − Market):** {edge*100:.2f}%")

    if edge >= 0.03:
        st.success("EDGE ✅ (model > market by > 3%)")
    else:
        st.warning("NO EDGE")

    with st.expander("Debug details"):
        st.write("Home sample:")
        st.write(f"- {home_team['name']} (id={home_team['id']})")
        st.write(f"- Matches used: {home_cnt} | Avg GF: {home_gf_avg:.2f} | Avg GA: {home_ga_avg:.2f}")

        st.write("Away sample:")
        st.write(f"- {away_team['name']} (id={away_team['id']})")
        st.write(f"- Matches used: {away_cnt} | Avg GF: {away_gf_avg:.2f} | Avg GA: {away_ga_avg:.2f}")

        st.write("Expected goals split:")
        st.write(f"- Expected home goals: {exp_home_goals:.2f}")
        st.write(f"- Expected away goals: {exp_away_goals:.2f}")
