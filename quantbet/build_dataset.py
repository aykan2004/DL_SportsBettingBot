"""Build the historical odds dataset used by the backtest.

Walks each league season chronologically, tracks rolling 5-match form, and —
for matches inside the target window — fetches a pre-kickoff (T-2h) odds
snapshot from the Odds API's paid /historical endpoint. Matches without
5 games of prior form or without clean odds are skipped.

Note: historical odds credits are expensive; the window below is deliberately
narrow. Widen START/END only when you mean to spend the credits.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher

import requests

from .config import (
    HISTORICAL_ODDS_JSON,
    LEAGUE_MAP,
    MAPPINGS_FILE,
    NAME_OVERRIDES,
    ODDS_API_KEY,
    REQUEST_TIMEOUT,
    STATS_API_KEY,
    STATS_URL,
)
from .features import Outcome, result_from_goals

log = logging.getLogger(__name__)

START = "2024-09-01"
END = "2024-10-31"
SEASON = 2024
FORM_WINDOW = 5


def get_finished_fixtures(league_id: int, season: int) -> list[dict]:
    r = requests.get(
        f"{STATS_URL}/fixtures",
        headers={"x-rapidapi-key": STATS_API_KEY},
        params={"league": league_id, "season": season},
        timeout=REQUEST_TIMEOUT,
    )
    games = [
        g
        for g in r.json().get("response", [])
        if g["fixture"]["status"]["short"] in ("FT", "AET", "PEN")
    ]
    games.sort(key=lambda g: g["fixture"]["timestamp"])
    return games


def get_historical_odds(sport_key: str, home: str, away: str, kickoff_iso: str) -> list[float] | None:
    """1X2 odds snapshot two hours before kickoff."""
    search_h = NAME_OVERRIDES.get(home, home)
    search_a = NAME_OVERRIDES.get(away, away)
    snap = (datetime.fromisoformat(kickoff_iso) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        r = requests.get(
            f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/odds",
            params={
                "apiKey": ODDS_API_KEY,
                "regions": "eu",
                "markets": "h2h",
                "date": snap,
                "oddsFormat": "decimal",
            },
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 429:
            log.error("Odds API rate limit hit — stopping odds lookups.")
            return None
        data = r.json().get("data", [])
    except requests.RequestException as e:
        log.warning("Odds request failed: %s", e)
        return None

    for ev in data:
        if (
            SequenceMatcher(None, search_h, ev["home_team"]).ratio() > 0.6
            and SequenceMatcher(None, search_a, ev["away_team"]).ratio() > 0.6
        ):
            bookmakers = ev.get("bookmakers", [])
            if not bookmakers:
                continue
            outcomes = {o["name"]: o["price"] for o in bookmakers[0]["markets"][0]["outcomes"]}
            return [
                outcomes.get(ev["home_team"], 0.0),
                outcomes.get("Draw", 0.0),
                outcomes.get(ev["away_team"], 0.0),
            ]
    return None


def build() -> None:
    print(f"Building historical odds dataset: {START} .. {END}")
    with open(MAPPINGS_FILE) as f:
        team_map = json.load(f)["teams"]

    dataset = []
    odds_calls = 0

    for league_id, info in LEAGUE_MAP.items():
        games = get_finished_fixtures(league_id, SEASON)
        history: dict[int, list[dict]] = {}
        saved = 0

        for g in games:
            home, away = g["teams"]["home"]["name"], g["teams"]["away"]["name"]
            if home not in team_map or away not in team_map:
                continue

            h_id, a_id = g["teams"]["home"]["id"], g["teams"]["away"]["id"]
            history.setdefault(h_id, [])
            history.setdefault(a_id, [])

            fulltime = g.get("score", {}).get("fulltime", {})
            h_goals = fulltime.get("home", g["goals"]["home"])
            a_goals = fulltime.get("away", g["goals"]["away"])
            if h_goals is None or a_goals is None:
                continue

            # Canonical encoding: 0=Home, 1=Draw, 2=Away — matching the model,
            # the training CSV and the odds array ordering.
            label = int(result_from_goals(h_goals, a_goals))

            date = g["fixture"]["date"][:10]
            if len(history[h_id]) >= FORM_WINDOW and len(history[a_id]) >= FORM_WINDOW and START <= date <= END:
                odds = get_historical_odds(info["key"], home, away, g["fixture"]["date"])
                odds_calls += 1
                if odds and all(o > 1.01 for o in odds):
                    h_form = history[h_id][-FORM_WINDOW:]
                    a_form = history[a_id][-FORM_WINDOW:]
                    dataset.append(
                        {
                            "features": {
                                "h_id": team_map[home],
                                "a_id": team_map[away],
                                "l_id": league_id,
                                "h_pts": sum(m["pts"] for m in h_form),
                                "h_gd": sum(m["gd"] for m in h_form) / FORM_WINDOW,
                                "a_pts": sum(m["pts"] for m in a_form),
                                "a_gd": sum(m["gd"] for m in a_form) / FORM_WINDOW,
                                "h_imp": 0.5,
                                "a_imp": 0.5,
                            },
                            "label": label,
                            "odds": odds,
                            "meta": f"{home} vs {away} ({date})",
                        }
                    )
                    saved += 1
                time.sleep(0.5)  # respect rate limits

            h_pts = 3 if label == Outcome.HOME else (1 if label == Outcome.DRAW else 0)
            a_pts = 3 if label == Outcome.AWAY else (1 if label == Outcome.DRAW else 0)
            history[h_id].append({"pts": h_pts, "gd": h_goals - a_goals})
            history[a_id].append({"pts": a_pts, "gd": a_goals - h_goals})

        print(f"  {info['name']}: {saved} matches with odds")

    with open(HISTORICAL_ODDS_JSON, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"Done: {len(dataset)} matches saved ({odds_calls} historical odds calls used).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build()
