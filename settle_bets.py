# settle_bets.py
import json
import requests
import os

# --- MODULAR IMPORTS ---
from config import STATS_API_KEY, STATS_URL, HISTORY_FILE


def get_match_result(home_id, away_id):
    headers = {'x-rapidapi-key': STATS_API_KEY}
    params = {'team': home_id, 'last': 10}

    try:
        response = requests.get(f"{STATS_URL}/fixtures", headers=headers,
                                params=params)
        data = response.json()
    except Exception as e:
        print(f"[ERROR] API Connection Error: {e}")
        return "Pending"

    for fixture in data.get('response', []):
        if fixture['teams']['away']['id'] == away_id:
            status = fixture['fixture']['status']['short']

            if status in ['FT', 'AET', 'PEN']:
                h_g = fixture['goals']['home']
                a_g = fixture['goals']['away']
                if h_g > a_g:
                    return "Home"
                elif a_g > h_g:
                    return "Away"
                else:
                    return "Draw"
            elif status in ['CANC', 'PSTP', 'ABD']:
                return "Void"

    return "Pending"


def run_settlement():
    if not os.path.exists(HISTORY_FILE) or os.path.getsize(HISTORY_FILE) == 0:
        print(f"[WARNING] No {HISTORY_FILE} found. Go place some bets first!")
        return

    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    except (json.JSONDecodeError, ValueError):
        print("[ERROR] Ledger is corrupted. Cannot settle bets.")
        return

    print("\n--- DAILY AUTOMATION: SETTLEMENT ENGINE ---")

    updates_made = False
    new_profit = 0.0

    for bet in history:
        if bet.get('status') == 'pending':
            actual_result = get_match_result(bet['home_id'], bet['away_id'])

            if actual_result == "Pending":
                continue

            updates_made = True
            placed_bet = bet['bet_placed']
            stake = float(bet['stake'])
            odds = float(bet['odds_taken'])

            # Grading Engine
            if actual_result == "Void":
                bet['status'] = "refunded"
                bet['profit'] = 0.0
                print(
                    f"[REFUND] {bet['home_team']} vs {bet['away_team']} (Match Voided)")

            elif "(DNB)" in placed_bet and actual_result == "Draw":
                bet['status'] = "refunded"
                bet['profit'] = 0.0
                print(
                    f"[REFUND] {bet['home_team']} vs {bet['away_team']} (DNB Draw)")

            elif (
                    placed_bet == "Home" or placed_bet == "Home (DNB)") and actual_result == "Home":
                bet['status'] = "won"
                bet['profit'] = round(stake * (odds - 1), 2)
                print(
                    f"[WIN] {bet['home_team']} vs {bet['away_team']} (+${bet['profit']})")

            elif (
                    placed_bet == "Away" or placed_bet == "Away (DNB)") and actual_result == "Away":
                bet['status'] = "won"
                bet['profit'] = round(stake * (odds - 1), 2)
                print(
                    f"[WIN] {bet['home_team']} vs {bet['away_team']} (+${bet['profit']})")

            elif placed_bet == "Draw" and actual_result == "Draw":
                bet['status'] = "won"
                bet['profit'] = round(stake * (odds - 1), 2)
                print(
                    f"[WIN] {bet['home_team']} vs {bet['away_team']} (+${bet['profit']})")

            else:
                bet['status'] = "lost"
                bet['profit'] = -stake
                print(
                    f"[LOSS] {bet['home_team']} vs {bet['away_team']} (-${stake})")

            new_profit += bet['profit']

    if updates_made:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"\n[SUCCESS] Ledger updated. Daily PnL: ${new_profit:.2f}")
    else:
        print("[INFO] No pending matches have finished yet.")

    total_profit = sum(float(b['profit']) for b in history if
                       b.get('status') in ['won', 'lost'])
    total_pending = sum(
        float(b['stake']) for b in history if b.get('status') == 'pending')

    print("\n--- LIFETIME BANKROLL SUMMARY ---")
    print(f"Total Profit/Loss: ${total_profit:.2f}")
    print(f"Money in Flight:   ${total_pending:.2f}")
    print("-" * 33)


if __name__ == "__main__":
    run_settlement()