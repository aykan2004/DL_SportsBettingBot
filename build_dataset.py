import json
import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import requests

from config import STATS_API_KEY, STATS_URL, ODDS_API_KEY, LEAGUE_MAP, \
    NAME_OVERRIDES

START = "2024-09-01"
END = "2024-10-31"


def get_fixtures(l_id, year):
    print(f"Pulling fixtures for league {l_id}...")
    res = requests.get(
        f"{STATS_URL}/fixtures",
        headers={'x-rapidapi-key': STATS_API_KEY},
        params={'league': l_id, 'season': year}
    ).json().get('response', [])

    # Just grab finished games and sort chronologically
    games = [g for g in res if
             g['fixture']['status']['short'] in ['FT', 'AET', 'PEN']]
    games.sort(key=lambda x: x['fixture']['timestamp'])
    return games


def get_odds(sport, h_team, a_team, date_str):
    h_search = NAME_OVERRIDES.get(h_team, h_team)
    a_search = NAME_OVERRIDES.get(a_team, a_team)

    kickoff = datetime.fromisoformat(date_str)
    snap = (kickoff - timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%SZ')

    try:
        r = requests.get(
            f"https://api.the-odds-api.com/v4/historical/sports/{sport}/odds",
            params={
                'apiKey': ODDS_API_KEY,
                'regions': 'eu',
                'markets': 'h2h',
                'date': snap,
                'oddsFormat': 'decimal'
            }
        )

        if r.status_code == 429:
            print("Hit the Odds API limit! Bailing.")
            return None

        data = r.json().get('data', [])
    except:
        return None

    for ev in data:
        if SequenceMatcher(None, h_search, ev['home_team']).ratio() > 0.6 and \
                SequenceMatcher(None, a_search, ev['away_team']).ratio() > 0.6:

            bks = ev.get('bookmakers', [])
            if not bks: continue

            # Just grab the first bookie's 1x2 market
            outcomes = {o['name']: o['price'] for o in
                        bks[0]['markets'][0]['outcomes']}
            return [outcomes.get(ev['home_team'], 0), outcomes.get('Draw', 0),
                    outcomes.get(ev['away_team'], 0)]

    return None


def main():
    print(f"Building dataset from {START} to {END}...")

    with open('mappings.json', 'r') as f:
        t_map = json.load(f)['teams']

    dataset = []
    api_hits = 0

    for l_id, info in LEAGUE_MAP.items():
        games = get_fixtures(l_id, 2024)
        if not games: continue

        history = {}
        saved = 0

        for g in games:
            h = g['teams']['home']['name']
            a = g['teams']['away']['name']
            date = g['fixture']['date'][:10]

            if h not in t_map or a not in t_map: continue

            h_id = g['teams']['home']['id']
            a_id = g['teams']['away']['id']

            if h_id not in history: history[h_id] = []
            if a_id not in history: history[a_id] = []

            try:
                h_goals = g['score']['fulltime']['home']
                a_goals = g['score']['fulltime']['away']
            except:
                h_goals = g['goals']['home']
                a_goals = g['goals']['away']

            if h_goals is None or a_goals is None: continue

            if h_goals > a_goals:
                res = 0
            elif a_goals > h_goals:
                res = 1
            else:
                res = 2

            # Require 5 games of form
            if len(history[h_id]) >= 5 and len(history[a_id]) >= 5:
                if START <= date <= END:
                    odds = get_odds(info['key'], h, a, g['fixture']['date'])
                    api_hits += 1

                    # Filter out suspended markets where odds drop to 1.01
                    if odds and all(o > 1.01 for o in odds):
                        h_form = history[h_id][-5:]
                        a_form = history[a_id][-5:]

                        dataset.append({
                            "features": {
                                "h_id": t_map[h], "a_id": t_map[a],
                                "l_id": l_id,
                                "h_pts": sum(m['pts'] for m in h_form),
                                "h_gd": sum(m['gd'] for m in h_form) / 5.0,
                                "a_pts": sum(m['pts'] for m in a_form),
                                "a_gd": sum(m['gd'] for m in a_form) / 5.0,
                                "h_imp": 0.5, "a_imp": 0.5
                            },
                            "label": res,
                            "odds": odds,
                            "meta": f"{h} v {a} ({date})"
                        })
                        saved += 1

                    time.sleep(0.5)  # Respect rate limits

            h_pts = 3 if res == 0 else (1 if res == 1 else 0)
            a_pts = 3 if res == 2 else (1 if res == 1 else 0)

            history[h_id].append({'pts': h_pts, 'gd': h_goals - a_goals})
            history[a_id].append({'pts': a_pts, 'gd': a_goals - h_goals})

        print(f"Grabbed {saved} matches for {info['name']}")

    with open('historical_dataset_2024.json', 'w') as f:
        json.dump(dataset, f, indent=4)

    print(
        f"Done. {len(dataset)} matches saved. Burned ~{api_hits * 10} odds API credits.")


if __name__ == "__main__":
    main()