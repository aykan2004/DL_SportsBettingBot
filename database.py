# database.py
import json
import os
from datetime import datetime
from config import HISTORY_FILE

def save_bet_to_ledger(bet_data, stake):
    if not os.path.exists(HISTORY_FILE) or os.path.getsize(HISTORY_FILE) == 0:
        with open(HISTORY_FILE, 'w') as f:
            json.dump([], f)

    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    except (json.JSONDecodeError, ValueError):
        history = []

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "home_team": str(bet_data['h_name']),
        "home_id": int(bet_data['h_id']),
        "away_team": str(bet_data['a_name']),
        "away_id": int(bet_data['a_id']),
        "league_id": int(bet_data['l_id']),
        "bet_placed": str(bet_data['bet']),
        "odds_taken": float(bet_data['odds']),
        "stake": float(stake),
        "model_prob": float(bet_data['prob']) / 100,
        "model_ev": float(bet_data['ev']) / 100,
        "status": "pending",
        "profit": 0.0
    }
    history.append(record)

    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"   [SUCCESS] Saved: {bet_data['h_name']} vs {bet_data['a_name']} ({bet_data['bet']} @ {bet_data['odds']:.2f})")