import os
import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st


# ----------------------------
# CONFIG
# ----------------------------
BASE_URL = "https://api.sportmonks.com/v3/football"
TOKEN_ENV_KEYS = ["SPORTMONKS_API_TOKEN", "SPORTMONKS_TOKEN", "SPORTMONKS_API_KEY"]


def get_token() -> Optional[str]:
    # Streamlit secrets first
    if "SPORTMONKS_API_TOKEN" in st.secrets:
        return st.secrets["SPORTMONKS_API_TOKEN"]
    # Env fallback
    for k in TOKEN_ENV_KEYS:
        v = os.getenv(k)
        if v:
            return v
    return None


# ----------------------------
# HTTP HELPERS
# ----------------------------
def sm_get(path: str, token: str, params: Optional[Dict] = None) -> Dict:
    """GET wrapper with clean error messages."""
    if params is None:
        params = {}
    params = dict(params)
    params["api_token"] = token

    url = f"{BASE_URL}/{path.lstrip('/')}"
    r = requests.get(url, params=params, timeout=30)

    # Keep a small debug payload for Streamlit display
    debug = {
        "url": r.url,
        "status": r.status_code,
    }

    try:
        j = r.json()
    except Exception:
        j = {"raw_text": r.text[:1000]}

    if r.status_code >= 400:
        # Raise with info
        msg = j.get("message") if isinstance(j, dict) else None
        raise requests.HTTPError(
            f"Sportmonks HTTP {r.status_code}. {msg or 'Request failed.'}",
            response=r,
        )

    # attach debug (non-sportmonks standard)
    if isinstance(j, dict):
        j["_debug"] = debug
    return j


# ----------------------------
# ODDS HELPERS
# ----------------------------
def american_to_implied_prob(american: int) -> float:
    """Convert American odds to implied probability (no vig removal)."""
    if american == 0:
        return 0.0
    if american < 0:
        return (-american) / ((-american) + 100.0)
    return 100.0 / (american + 100.0)


# ----------------------------
# POISSON HELPERS
# ----------------------------
def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def prob_over_25_from_lambda(lam: float) -> float:
    """
    Total goals ~ Poisson(lam).
    Over 2.5 = P(G >= 3) = 1 - (P0 + P1 + P2)
    """
    if lam <= 0:
        return 0.0
    p0 = poisson_pmf(0, lam)
    p1 = poisson_pmf(1, lam)
    p2 = poisson_pmf(2, lam)
    return max(0.0, min(1.0, 1.0 - (p0 + p1 + p2)))


# ----------------------------
# SPORTMONKS DATA PARSING
# ----------------------------
@dataclass
class League:
    id: int
    name: str
    country: str


@dataclass
class Season:
    id: int
    name: str
    year: Optional[int]
    starting_at: Optional[str]
    is_current: bool


def safe_get(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


@st.cache_data(ttl=3600)
def fetch_leagues(token: str) -> List[League]:
    # Per Sportmonks, leagues endpoint is core; we keep it simple and paginate if needed.
    # Many plans return a manageable list.
    out: List[League] = []
    page = 1
    while True:
        j = sm_get("leagues", token, params={"page": page})
        data = j.get("data", [])
        if not data:
            break
        for row in data:
            out.append(
                League(
                    id=int(row["id"]),
                    name=str(row.get("name", "")),
                    country=str(row.get("country", {}).get("name", "")) if isinstance(row.get("country"), dict) else str(row.get("country_name", "")),
                )
            )
        pagination = j.get("pagination") or j.get("meta", {}).get("pagination")
        if isinstance(pagination, dict):
            has_more = pagination.get("has_more")
            if has_more is False:
                break
        # fallback: if less than typical page size, stop
        if len(data) < 25:
            break
        page += 1
        if page > 30:
            break
    # remove empty names
    out = [l for l in out if l.name]
    return out


@st.cache_data(ttl=3600)
def fetch_seasons_for_league(token: str, league_id: int) -> List[Season]:
    """
    Use /seasons with filter on league.
    Docs: seasons endpoints provide Season ID, Name, League ID, Year, Active.  [oai_citation:1‡docs.sportmonks.com](https://docs.sportmonks.com/v3/endpoints-and-entities/endpoints/seasons?utm_source=chatgpt.com)
    """
    seasons: List[Season] = []
    page = 1
    while True:
        j = sm_get(
            "seasons",
            token,
            params={
                "page": page,
                "filters": f"seasonLeagues:{league_id}",
            },
        )
        data = j.get("data", [])
        if not data:
            break

        for s in data:
            seasons.append(
                Season(
                    id=int(s["id"]),
                    name=str(s.get("name", "")),
                    year=int(s["year"]) if s.get("year") not in (None, "") else None,
                    starting_at=s.get("starting_at"),
                    is_current=bool(s.get("is_current", False)),
                )
            )

        pagination = j.get("pagination") or j.get("meta", {}).get("pagination")
        if isinstance(pagination, dict) and pagination.get("has_more") is False:
            break
        if len(data) < 25:
            break
        page += 1
        if page > 50:
            break

    # Sort newest first by year/starting_at
    def sort_key(x: Season):
        y = x.year or 0
        return (y, x.starting_at or "")

    seasons.sort(key=sort_key, reverse=True)
    return seasons


def normalize_name(s: str) -> str:
    return "".join(ch.lower() for ch in s.strip() if ch.isalnum() or ch.isspace()).strip()


def best_fuzzy_match(target: str, options: List[str]) -> Optional[str]:
    """
    Light fuzzy match without extra libraries.
    Strategy: exact (normalized) contains / startswith, else token overlap.
    """
    t = normalize_name(target)
    if not t:
        return None
    norm_opts = [(opt, normalize_name(opt)) for opt in options]

    # exact
    for opt, n in norm_opts:
        if n == t:
            return opt

    # startswith
    for opt, n in norm_opts:
        if n.startswith(t) or t.startswith(n):
            return opt

    # contains
    for opt, n in norm_opts:
        if t in n or n in t:
            return opt

    # token overlap
    tset = set(t.split())
    best = None
    best_score = 0
    for opt, n in norm_opts:
        oset = set(n.split())
        score = len(tset & oset)
        if score > best_score:
            best_score = score
            best = opt
    return best if best_score > 0 else None


@st.cache_data(ttl=900)
def fetch_fixtures_for_season(token: str, season_id: int, lookback_days: int) -> List[Dict]:
    """
    Pull finished fixtures for a season within lookback window.
    Use /fixtures + filters (Sportmonks supports filtering fixtures by leagues etc).  [oai_citation:2‡docs.sportmonks.com](https://docs.sportmonks.com/v3/endpoints-and-entities/endpoints/fixtures/get-all-fixtures?utm_source=chatgpt.com)
    We'll filter by season using fixtureSeasons.
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days)

    fixtures: List[Dict] = []
    page = 1
    while True:
        j = sm_get(
            "fixtures",
            token,
            params={
                "page": page,
                "include": "participants;scores",
                "filters": f"fixtureSeasons:{season_id}",
                # Date filters can be plan-dependent; we still apply if supported.
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            },
        )
        data = j.get("data", [])
        if not data:
            break

        fixtures.extend(data)

        pagination = j.get("pagination") or j.get("meta", {}).get("pagination")
        if isinstance(pagination, dict) and pagination.get("has_more") is False:
            break
        if len(data) < 25:
            break
        page += 1
        if page > 50:
            break

    # Keep only finished with scores present
    finished = []
    for fx in fixtures:
        scores = fx.get("scores") or []
        # If no scores array, skip
        if not scores:
            continue
        finished.append(fx)
    return finished


def extract_total_goals_from_scores(fixture: Dict) -> Optional[int]:
    """
    Sportmonks scores format varies by include.
    We try common patterns:
      - scores entries with 'description' like 'CURRENT' or 'FT'
      - scores entries per participant (home/away)
    We’ll attempt robust extraction:
      sum of participant scores at FT/Current.
    """
    scores = fixture.get("scores") or []
    if not scores:
        return None

    # prefer CURRENT / FT-like scores
    preferred = []
    for s in scores:
        desc = str(s.get("description", "")).upper()
        if "CURRENT" in desc or "FT" in desc or "FULL" in desc:
            preferred.append(s)
    use = preferred if preferred else scores

    # many responses have score 'score' dict with 'goals'
    goals = []
    for s in use:
        sc = s.get("score")
        if isinstance(sc, dict):
            g = sc.get("goals")
            if isinstance(g, int):
                goals.append(g)

    if goals:
        # Often two entries: home and away goals
        # If more entries exist, take last two
        if len(goals) >= 2:
            return int(goals[-1] + goals[-2])
        return int(goals[0])

    return None


def participant_names(fixture: Dict) -> List[str]:
    parts = fixture.get("participants") or []
    names = []
    for p in parts:
        name = p.get("name")
        if name:
            names.append(str(name))
    return names


def find_team_ids_from_fixtures(fixtures: List[Dict]) -> Dict[str, int]:
    """
    Build map {team_name: team_id} from participants in fixtures.
    """
    m: Dict[str, int] = {}
    for fx in fixtures:
        parts = fx.get("participants") or []
        for p in parts:
            pid = p.get("id")
            pname = p.get("name")
            if pid and pname and pname not in m:
                m[str(pname)] = int(pid)
    return m


def filter_team_recent_fixtures(fixtures: List[Dict], team_id: int, n: int = 10) -> List[Dict]:
    """
    Take most recent N fixtures for a team (based on starting_at).
    """
    team_fx = []
    for fx in fixtures:
        parts = fx.get("participants") or []
        ids = [p.get("id") for p in parts if isinstance(p, dict)]
        if team_id in ids:
            team_fx.append(fx)

    def dt_key(fx: Dict):
        s = fx.get("starting_at") or fx.get("starting_at_timestamp") or ""
        return str(s)

    team_fx.sort(key=dt_key, reverse=True)
    return team_fx[:n]


def estimate_lambda_from_recent(fixtures: List[Dict], home_id: int, away_id: int, n_each: int = 10) -> Tuple[float, Dict]:
    """
    Estimate expected total goals (lambda) from:
      - last n fixtures for home team
      - last n fixtures for away team
    We take average total goals from each set, then average them.
    """
    h_fx = filter_team_recent_fixtures(fixtures, home_id, n=n_each)
    a_fx = filter_team_recent_fixtures(fixtures, away_id, n=n_each)

    def avg_totals(fxs: List[Dict]) -> Tuple[float, int, List[int]]:
        totals = []
        for fx in fxs:
            tg = extract_total_goals_from_scores(fx)
            if tg is not None:
                totals.append(int(tg))
        if not totals:
            return (0.0, 0, [])
        return (sum(totals) / len(totals), len(totals), totals)

    h_avg, h_cnt, h_totals = avg_totals(h_fx)
    a_avg, a_cnt, a_totals = avg_totals(a_fx)

    # combine
    usable = [x for x in [h_avg, a_avg] if x > 0]
    lam = sum(usable) / len(usable) if usable else 0.0

    debug = {
        "home_fixtures_found": len(h_fx),
        "away_fixtures_found": len(a_fx),
        "home_totals_used": h_cnt,
        "away_totals_used": a_cnt,
        "home_avg_total_goals": h_avg,
        "away_avg_total_goals": a_avg,
        "lambda": lam,
    }
    return lam, debug


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Over 2.5 Betting Model (Sportmonks)", layout="centered")
st.title("Over 2.5 Betting Model (Sportmonks)")
st.caption("Pick league + season, enter teams + Over 2.5 American odds. Uses recent finished fixtures to estimate λ and Poisson P(Over 2.5).")

token = get_token()
if not token:
    st.error("Missing SPORTMONKS_API_TOKEN. Add it in Streamlit: App → Settings → Secrets as SPORTMONKS_API_TOKEN.")
    st.stop()

# Leagues
try:
    leagues = fetch_leagues(token)
except Exception as e:
    st.error(f"Could not load leagues. {e}")
    st.stop()

league_label_map = {f"{l.name} ({l.country})".strip(): l.id for l in leagues}
league_choices = sorted(league_label_map.keys())

league_choice = st.selectbox("League", league_choices, index=0)

league_id = league_label_map[league_choice]

# Seasons for league
try:
    seasons = fetch_seasons_for_league(token, league_id)
except Exception as e:
    st.error(f"Could not load seasons for league. {e}")
    st.stop()

if not seasons:
    st.warning("No seasons returned for this league in your Sportmonks plan.")
    st.stop()

season_labels = []
season_id_by_label = {}
for s in seasons:
    yr = f"{s.year}" if s.year else (s.starting_at[:4] if s.starting_at else "")
    cur = " (current)" if s.is_current else ""
    label = f"{s.name} — {yr}{cur}".strip()
    season_labels.append(label)
    season_id_by_label[label] = s.id

season_choice = st.selectbox("Season", season_labels, index=0)
season_id = season_id_by_label[season_choice]

home_team = st.text_input("Home Team", value="Bournemouth")
away_team = st.text_input("Away Team", value="Sunderland")
american_odds = st.number_input("Over 2.5 American Odds", value=-110, step=5)

lookback_days = st.slider("Lookback window (days) for finished fixtures", min_value=60, max_value=730, value=200, step=10)

calc = st.button("Calculate", type="primary")

if calc:
    try:
        fixtures = fetch_fixtures_for_season(token, season_id, lookback_days)
    except Exception as e:
        st.error(f"Could not fetch fixtures. {e}")
        st.stop()

    if not fixtures:
        st.error("No finished fixtures returned for this season in that lookback window.")
        st.stop()

    # Build team map from participants found in fixtures
    name_to_id = find_team_ids_from_fixtures(fixtures)
    options = list(name_to_id.keys())

    home_match = best_fuzzy_match(home_team, options)
    away_match = best_fuzzy_match(away_team, options)

    if not home_match:
        st.error(f"Home team not found in this league/season fixtures: {home_team}")
        st.info("Tip: try the exact name Sportmonks uses (as shown in fixtures/participants).")
        st.stop()

    if not away_match:
        st.error(f"Away team not found in this league/season fixtures: {away_team}")
        st.info("Tip: try the exact name Sportmonks uses (as shown in fixtures/participants).")
        st.stop()

    home_id = name_to_id[home_match]
    away_id = name_to_id[away_match]

    st.caption(f"Matched Home: **{home_match}** (team_id: {home_id})")
    st.caption(f"Matched Away: **{away_match}** (team_id: {away_id})")

    lam, dbg = estimate_lambda_from_recent(fixtures, home_id, away_id, n_each=10)

    if lam <= 0:
        st.error("Not enough finished fixtures with usable scores for these teams in your lookback window.")
        st.json(dbg)
        st.stop()

    model_p = prob_over_25_from_lambda(lam)
    market_p = american_to_implied_prob(int(american_odds))
    edge = model_p - market_p

    st.subheader("Model Output")
    st.write(f"**Expected Total Goals (λ):** {lam:.2f}")
    st.write(f"**Model Probability Over 2.5:** {model_p*100:.2f}%")
    st.write(f"**Market Implied Probability:** {market_p*100:.2f}%")
    st.write(f"**Edge (Model − Market):** {edge*100:.2f}%")

    if edge > 0.03:
        st.success("EDGE ✅ (model > market by > 3%)")
    else:
        st.warning("NO EDGE (or too small)")

    with st.expander("Debug details"):
        st.json(dbg)
        st.write(f"Fixtures fetched: {len(fixtures)}")
