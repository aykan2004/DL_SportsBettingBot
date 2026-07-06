"""HTTP clients for the stats API (api-football) and the Odds API.

All requests go through one Session with retries and a hard timeout; the
original code had neither, so a single hung socket could stall the nightly
cron forever.
"""

import logging
from datetime import datetime, timedelta
from difflib import SequenceMatcher

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import (
    DEFAULT_LEAGUE_WEIGHT,
    LEAGUE_MAP,
    LEAGUE_WEIGHTS,
    NAME_OVERRIDES,
    ODDS_API_KEY,
    ODDS_URL,
    REQUEST_TIMEOUT,
    STATS_API_KEY,
    STATS_URL,
)
from .features import SEQ_LEN, pad_history

log = logging.getLogger(__name__)

FINISHED_STATUSES = ("FT", "AET", "PEN")
NAME_MATCH_THRESHOLD = 0.6

_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503])
    ),
)


def _stats_get(path: str, params: dict) -> dict:
    r = _session.get(
        f"{STATS_URL}/{path}",
        headers={"x-rapidapi-key": STATS_API_KEY},
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def current_season(league_id: int, today: datetime | None = None) -> int:
    """Season label for a league. European seasons are labelled by their start
    year (a March 2026 fixture belongs to season 2025); calendar-year leagues
    (e.g. Brazilian Série A) use the year itself. Computed locally instead of
    burning an API call per league per run."""
    today = today or datetime.now()
    if LEAGUE_MAP.get(league_id, {}).get("calendar_season"):
        return today.year
    return today.year if today.month >= 7 else today.year - 1


def get_upcoming_fixtures(league_id: int, season: int, days_ahead: int = 3) -> list[dict]:
    now = datetime.now()
    data = _stats_get(
        "fixtures",
        {
            "league": league_id,
            "season": season,
            "from": now.strftime("%Y-%m-%d"),
            "to": (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d"),
            "timezone": "UTC",
        },
    )
    return data.get("response", [])


def _extract_fulltime_goals(fixture: dict) -> tuple[int | None, int | None]:
    """Prefer the 90-minute score; extra time and shootouts say little about
    league form. Fall back to the headline goals if fulltime is absent."""
    score = fixture.get("score", {}).get("fulltime", {})
    h, a = score.get("home"), score.get("away")
    if h is None or a is None:
        h, a = fixture["goals"]["home"], fixture["goals"]["away"]
    return h, a


def get_team_recent_stats(team_id: int) -> tuple[list[float], list[list[float]]] | None:
    """Weighted form points, per-game GD and the GF/GA history over the last
    SEQ_LEN finished matches. Returns None when the API has no history —
    fabricating a neutral prior (the old behaviour) just feeds the model
    made-up features, so callers should skip the fixture instead."""
    data = _stats_get(
        "fixtures",
        {"team": team_id, "last": SEQ_LEN, "status": "-".join(FINISHED_STATUSES)},
    )
    fixtures = data.get("response", [])
    if not fixtures:
        return None

    # The API returns newest-first; flip to chronological so the LSTM sees
    # oldest -> newest and left-padding lands on the oldest slots.
    fixtures = list(reversed(fixtures[:SEQ_LEN]))

    points = 0.0
    gd_total = 0.0
    history = []
    for f in fixtures:
        weight = LEAGUE_WEIGHTS.get(f["league"]["id"], DEFAULT_LEAGUE_WEIGHT)
        h_goals, a_goals = _extract_fulltime_goals(f)
        h_goals = h_goals if h_goals is not None else 0
        a_goals = a_goals if a_goals is not None else 0

        is_home = f["teams"]["home"]["id"] == team_id
        gf, ga = (h_goals, a_goals) if is_home else (a_goals, h_goals)
        history.append([gf, ga])

        if gf > ga:
            points += 3 * weight
        elif gf == ga:
            points += 1 * weight
        gd_total += (gf - ga) * weight

    return [points, gd_total / len(fixtures)], pad_history(history)


def _names_match(search_name: str, api_name: str) -> bool:
    return SequenceMatcher(None, search_name, api_name).ratio() > NAME_MATCH_THRESHOLD


def get_match_odds(sport_key: str, home_team: str, away_team: str) -> list[float] | None:
    """Best-effort 1X2 decimal odds [home, draw, away] from the first
    bookmaker listing the fixture, or None if it can't be matched."""
    search_h = NAME_OVERRIDES.get(home_team, home_team)
    search_a = NAME_OVERRIDES.get(away_team, away_team)

    r = _session.get(
        f"{ODDS_URL}/{sport_key}/odds",
        params={
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "decimal",
        },
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    events = r.json()
    if not isinstance(events, list):
        log.warning("Odds API returned non-list payload for %s", sport_key)
        return None

    for event in events:
        if not (
            _names_match(search_h, event["home_team"])
            and _names_match(search_a, event["away_team"])
        ):
            continue
        if not event.get("bookmakers"):
            return None
        outcomes = {
            o["name"]: o["price"] for o in event["bookmakers"][0]["markets"][0]["outcomes"]
        }
        return [
            outcomes.get(event["home_team"], 0.0),
            outcomes.get("Draw", 0.0),
            outcomes.get(event["away_team"], 0.0),
        ]
    return None
