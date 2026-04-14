import torch
import json
import numpy as np
import builtins
import time

# --- MODULAR IMPORTS ---
from config import LEAGUE_MAP
from model import SuperSoccerNet
from database import save_bet_to_ledger
from api_client import get_current_season, get_fixtures_next_5_days, \
    get_stats_for_match, get_match_odds


def filter_correlated_bets(candidates):
    """
    If multiple lines (e.g., 1X2 and DNB) are found for the same fixture,
    it only keeps the line with the highest Expected Value (EV).
    """
    unique_bets = {}

    for bet in candidates:
        match_name = bet['match']

        if match_name not in unique_bets:
            unique_bets[match_name] = bet
        else:
            if bet['ev'] > unique_bets[match_name]['ev']:
                unique_bets[match_name] = bet
    return list(unique_bets.values())


def generate_value_slip():
    # 1. System Initialization
    try:
        with open('mappings.json', 'r') as f:
            m = json.load(f)
            t_map = m['teams']
            l_map = {int(k): v for k, v in m['leagues'].items()}
            s_mean = np.array(m['scaler_mean'])
            s_scale = np.array(m['scaler_scale'])

        model = SuperSoccerNet(len(t_map), len(l_map))
        model.load_state_dict(torch.load('soccer_model_full.pth'))
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model or mappings: {e}")
        return

    print("\n--- HIGH VALUE BET GENERATOR (5-Day Outlook) ---")
    candidates = []

    # 2. Data Gathering & Prediction
    for l_id, info in LEAGUE_MAP.items():
        season = get_current_season(l_id)
        print(f"[INFO] Scanning {info['name']} for value anomalies...")

        fixtures = get_fixtures_next_5_days(l_id, season)

        for f in fixtures:
            h, a = f['teams']['home']['name'], f['teams']['away']['name']

            if h in t_map and a in t_map:
                h_s, a_s = get_stats_for_match(f['teams']['home']['id'],
                                               f['teams']['away']['id'])

                cont = torch.tensor(np.hstack(((np.array(
                    [[h_s[0], a_s[0], 0, 0]]) - s_mean[:4]) / s_scale[:4],
                                               [[0.5, 0.5]])),
                                    dtype=torch.float32)
                cat = torch.tensor([[t_map[h], t_map[a], l_map.get(l_id, 0)]],
                                   dtype=torch.long)

                h_idx_t = cat[:, 0]
                a_idx_t = cat[:, 1]
                c_idx_t = cat[:, 2]
                seq_tensor = torch.zeros((1, 5, 4), dtype=torch.float32)

                with torch.no_grad():
                    probs = torch.softmax(model(h_idx_t, a_idx_t, c_idx_t, seq_tensor, cont), dim=1).numpy()[0]

                odds = get_match_odds(info['key'], h, a)

                if odds:
                    for i, label in enumerate(['Home', 'Draw', 'Away']):
                        if odds[i] <= 1.01: continue
                        ev = (probs[i] * odds[i]) - 1

                        # High Value Filters: >= 25% Win Prob, Odds >= 2.50, +EV >= 0.15
                        if probs[i] >= 0.25 and odds[i] >= 2.50 and ev >= 0.15:
                            candidates.append({
                                'h_name': h, 'a_name': a,
                                'h_id': f['teams']['home']['id'],
                                'a_id': f['teams']['away']['id'], 'l_id': l_id,
                                'league': info['name'],
                                'date': f['fixture']['date'][:16],
                                'match': f"{h} vs {a}", 'bet': label,
                                'prob': probs[i] * 100,
                                'odds': odds[i], 'ev': ev * 100
                            })
            time.sleep(0.1)

    # 3. Final Slip Output
    # Difference from main_safe: Sorting by EV (Expected Value) instead of Model Probability
    candidates = filter_correlated_bets(candidates)
    candidates.sort(key=lambda x: x['ev'], reverse=True)
    final_slip = candidates[:10]

    print("-" * 50)
    if not final_slip:
        print("[WARNING] No high-value anomalies found in the current market.")
        return

    for idx, bet in enumerate(final_slip):
        print(f"[{idx + 1}] {bet['league']} | {bet['match']}")
        print(f"    Target: {bet['bet']} @ {bet['odds']:.2f} (EV: {bet['ev']:.1f}% | Conf: {bet['prob']:.1f}%)")

    while True:
        choice_raw = input("\n[INPUT] Enter match # to log (or 'q' to quit): ").strip()
        if choice_raw.lower() == 'q': break

        choice_clean = "".join(filter(str.isdigit, choice_raw))
        if not choice_clean: continue

        idx = int(choice_clean) - 1
        if 0 <= idx < len(final_slip):
            selected = final_slip[idx]
            print(f"[ACTION] Targeting: {selected['match']} ({selected['bet']})")

            raw_stake = input("[INPUT] Enter stake (Press Enter for 5): ").strip()

            if not raw_stake:
                stake = 5.0
            else:
                try:
                    scrubbed = "".join(c for c in raw_stake if c.isdigit() or c == '.')
                    stake = builtins.float(scrubbed)
                except Exception:
                    print("[WARNING] System input error. Defaulting to $5.00.")
                    stake = 10.0

            save_bet_to_ledger(selected, stake)

            if input("\n[INPUT] Log another bet? (y/n): ").strip().lower() != 'y': break

if __name__ == "__main__":
    generate_value_slip()