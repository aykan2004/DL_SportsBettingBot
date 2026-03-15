# config.py
import os
from dotenv import load_dotenv

load_dotenv()

SPORTS_API_KEY = os.getenv("SPORTS_API_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
STATS_URL = "https://v3.football.api-sports.io"
ODDS_URL = "https://api.the-odds-api.com/v4/sports"

HISTORY_FILE = "bet_history.json"

LEAGUE_MAP = {
    39: {'key': 'soccer_epl', 'name': '🇬🇧 Premier League'},
    2: {'key': 'soccer_uefa_champs_league', 'name': '🇪🇺 Champions League'},
    3: {'key': 'soccer_uefa_europa_league', 'name': '🇪🇺 Europa League'},
    203: {'key': 'soccer_turkey_super_league', 'name': '🇹🇷 Süper Lig'},
    71: {'key': 'soccer_brazil_campeonato', 'name': '🇧🇷 Série A'}
}

LEAGUE_WEIGHTS = {
    39: 1.0, 140: 1.0, 2: 1.0,
    78: 0.9, 135: 0.9, 3: 0.9,
    203: 0.7, 71: 0.7
}

NAME_OVERRIDES = {
    "Manchester United": "Man Utd",
    "Fenerbahçe": "Fenerbahce",
    "Besiktas": "Besiktas JK"
}