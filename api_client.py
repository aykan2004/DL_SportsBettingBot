# api_client.py
import requests
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from config import STATS_API_KEY, STATS_URL, ODDS_API_KEY, ODDS_URL, \
    LEAGUE_WEIGHTS, NAME_OVERRIDES


def get_current_season(league_id):
    headers = {'x-rapidapi-key': STATS_API_KEY}
    params = {'league': league_id, 'season': 2025, 'next': 1}
    r = requests.get(f"{STATS_URL}/fixtures", headers=headers, params=params)
    return 2025 if r.json().get('results', 0) > 0 else 2026


def get_fixtures_next_5_days(league_id, season):
    headers = {'x-rapidapi-key': STATS_API_KEY}
    now = datetime.now()
    params = {
        'league': league_id,
        'season': season,
        'from': now.strftime("%Y-%m-%d"),
        'to': (now + timedelta(days=5)).strftime("%Y-%m-%d"),
        'timezone': 'Europe/London'
    }
    r = requests.get(f"{STATS_URL}/fixtures", headers=headers, params=params)
    return r.json().get('response', [])


def get_stats_for_match(h_id, a_id):
    headers = {'x-rapidapi-key': STATS_API_KEY}

    def fetch(t_id):
        params = {'team': t_id, 'last': 5, 'status': 'FT-AET-PEN'}
        r = requests.get(f"{STATS_URL}/fixtures", headers=headers,
                         params=params)
        fixtures = r.json().get('response', [])
        if not fixtures: return [5.0, 0.0]

        total_points = 0
        total_gd = 0

        for f in fixtures:
            comp_id = f['league']['id']
            weight = LEAGUE_WEIGHTS.get(comp_id, 0.5)

            # Strict 90-minute fulltime score extraction
            try:
                h_g = f['score']['fulltime']['home']
                a_g = f['score']['fulltime']['away']
            except (KeyError, TypeError):
                h_g, a_g = None, None

            if h_g is None or a_g is None:
                h_g = f['goals']['home']
                a_g = f['goals']['away']

            h_g = h_g if h_g is not None else 0
            a_g = a_g if a_g is not None else 0

            is_home = (f['teams']['home']['id'] == t_id)

            raw_pts = 0
            if h_g > a_g:
                raw_pts = 3 if is_home else 0
            elif a_g > h_g:
                raw_pts = 3 if not is_home else 0
            else:
                raw_pts = 1

            total_points += (raw_pts * weight)

            raw_gd = h_g - a_g
            if not is_home: raw_gd = -raw_gd
            total_gd += (raw_gd * weight)

        return [float(total_points), float(total_gd / len(fixtures))]

    return fetch(h_id), fetch(a_id)


def get_match_odds(sport_key, h_team, a_team):
    search_h = NAME_OVERRIDES.get(h_team, h_team)
    search_a = NAME_OVERRIDES.get(a_team, a_team)

    params = {'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': 'h2h',
              'oddsFormat': 'decimal'}
    r = requests.get(f"{ODDS_URL}/{sport_key}/odds", params=params)
    events = r.json()

    if not isinstance(events, list): return None

    for event in events:
        # Match teams using SequenceMatcher
        if SequenceMatcher(None, search_h,
                           event['home_team']).ratio() > 0.6 and \
                SequenceMatcher(None, search_a,
                                event['away_team']).ratio() > 0.6:
            bk = event['bookmakers'][0] if event['bookmakers'] else None
            if not bk: return None

            o = {outcome['name']: outcome['price'] for outcome in
                 bk['markets'][0]['outcomes']}
            return [o.get(event['home_team'], 0), o.get('Draw', 0),
                    o.get(event['away_team'], 0)]

    return None