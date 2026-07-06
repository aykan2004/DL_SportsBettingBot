"""Central configuration: paths, API endpoints, league metadata."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths (anchored to the repo root so cron jobs work from any cwd) ---
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

TRAINING_CSV = DATA_DIR / "soccer_data_full.csv"
HISTORICAL_ODDS_JSON = DATA_DIR / "historical_dataset_2024.json"
HISTORY_FILE = DATA_DIR / "bet_history.json"
MODEL_FILE = MODELS_DIR / "soccer_model_full.pth"
MAPPINGS_FILE = MODELS_DIR / "mappings.json"

# --- API credentials / endpoints ---
STATS_API_KEY = os.getenv("STATS_API_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
STATS_URL = "https://v3.football.api-sports.io"
ODDS_URL = "https://api.the-odds-api.com/v4/sports"

REQUEST_TIMEOUT = 10  # seconds; every outbound HTTP call must set one

# --- Leagues we generate slips for ---
# `calendar_season`: Brazilian Série A runs Jan-Dec, so the API season label
# is the calendar year rather than the European YYYY/YY+1 convention.
LEAGUE_MAP = {
    39: {"key": "soccer_epl", "name": "Premier League", "calendar_season": False},
    2: {"key": "soccer_uefa_champs_league", "name": "Champions League", "calendar_season": False},
    3: {"key": "soccer_uefa_europa_league", "name": "Europa League", "calendar_season": False},
    203: {"key": "soccer_turkey_super_league", "name": "Süper Lig", "calendar_season": False},
    71: {"key": "soccer_brazil_campeonato", "name": "Série A (BR)", "calendar_season": True},
}

# Strength weights applied to recent-form points/GD. This map is wider than
# LEAGUE_MAP on purpose: a team's last-5 window can include competitions we
# never bet on (La Liga 140, Bundesliga 78, Serie A 135, domestic cups...).
# Unknown competitions fall back to 0.5 in the form calculation.
LEAGUE_WEIGHTS = {
    39: 1.0, 140: 1.0, 2: 1.0,
    78: 0.9, 135: 0.9, 3: 0.9,
    203: 0.7, 71: 0.7,
}
DEFAULT_LEAGUE_WEIGHT = 0.5

# Stats-API team names -> Odds-API team names, for fixtures where fuzzy
# matching alone is not reliable.
NAME_OVERRIDES = {
    "Manchester United": "Man Utd",
    "Fenerbahçe": "Fenerbahce",
    "Besiktas": "Besiktas JK",
}


def require_api_keys() -> None:
    """Fail fast with a clear message instead of sending unauthenticated requests."""
    missing = [
        name
        for name, value in (("STATS_API_KEY", STATS_API_KEY), ("ODDS_API_KEY", ODDS_API_KEY))
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your API keys."
        )
