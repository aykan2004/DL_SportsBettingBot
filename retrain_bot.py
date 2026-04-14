import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import os
import random
import pandas as pd

# --- MODULAR IMPORTS ---
from config import HISTORY_FILE
from model import SuperSoccerNet


def retrain_model():
    print("\n--- DAILY AUTOMATION: ML RETRAINER (WITH EXPERIENCE REPLAY) ---")

    try:
        with open('mappings.json', 'r') as f:
            m = json.load(f)
    except FileNotFoundError:
        print("[ERROR] Cannot find mappings.json. Aborting.")
        return

    if not os.path.exists(HISTORY_FILE):
        print(f"[ERROR] No {HISTORY_FILE} found. Aborting.")
        return

    with open(HISTORY_FILE, 'r') as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            print("[ERROR] Ledger is corrupted. Aborting.")
            return

    trainable_bets = [b for b in history if b.get('status') in ['won', 'lost']]
    skipped_bets = len(history) - len(trainable_bets)

    if len(trainable_bets) < 2:
        print(f"[INFO] Found {len(trainable_bets)} finished bet(s).")
        print(
            "[WARNING] The neural network requires at least 2 finished bets to calculate batch variance.")
        print("[INFO] Going back to sleep until more matches finish.")
        return

    print(
        f"[INFO] Found {len(trainable_bets)} new actionable bets in ledger. (Skipped {skipped_bets} pending/refunded).")

    cat_data = []
    labels = []

    # 1. Load New Data (from Ledger)
    for bet in trainable_bets:
        h_name = bet['home_team']
        a_name = bet['away_team']

        if h_name not in m['teams'] or a_name not in m['teams']:
            continue

        h_idx = m['teams'][h_name]
        a_idx = m['teams'][a_name]
        l_idx = m['leagues'].get(str(bet['league_id']), 0)

        bet_placed = bet['bet_placed'].replace(" (DNB)", "")
        status = bet['status']

        if status == 'won':
            if bet_placed == 'Home':
                true_label = 0
            elif bet_placed == 'Draw':
                true_label = 1
            else:
                true_label = 2
        else:
            if bet_placed == 'Home':
                true_label = 2
            elif bet_placed == 'Draw':
                true_label = 0
            else:
                true_label = 0

        cat_data.append([h_idx, a_idx, l_idx])
        labels.append(true_label)

    new_bets_count = len(cat_data)

    # 2. Load Historical Data (Experience Replay Buffer)
    print("[INFO] Building Experience Replay Buffer from historical data...")
    try:
        df = pd.read_csv('soccer_data_full.csv', encoding='utf-8')
        df = df.dropna(subset=['result', 'home_team', 'away_team'])

        replay_size = 400
        if len(df) > replay_size:
            df = df.sample(n=replay_size)

        hist_count = 0
        for _, row in df.iterrows():
            h_name = row['home_team']
            a_name = row['away_team']

            if h_name in m['teams'] and a_name in m['teams']:
                h_idx = m['teams'][h_name]
                a_idx = m['teams'][a_name]
                l_idx = m['leagues'].get(str(row.get('league_id', 0)), 0)

                cat_data.append([h_idx, a_idx, l_idx])
                labels.append(int(row['result']))
                hist_count += 1

        print(
            f"[SUCCESS] Buffer built: {new_bets_count} new matches + {hist_count} historical matches.")
    except Exception as e:
        print(
            f"[WARNING] Could not load historical data for replay buffer: {e}")
        print(
            "[WARNING] Proceeding to train ONLY on new ledger data (High risk of overfitting).")

    if not cat_data:
        print("[WARNING] No valid team mappings found. Aborting.")
        return

    # 3. Prepare Tensors
    combined = list(zip(cat_data, labels))
    random.shuffle(combined)
    cat_data, labels = zip(*combined)

    cat_tensor = torch.tensor(cat_data, dtype=torch.long)

    h_idx_t = cat_tensor[:, 0]
    a_idx_t = cat_tensor[:, 1]
    c_idx_t = cat_tensor[:, 2]

    cont_tensor = torch.zeros((len(cat_data), 6), dtype=torch.float32)
    seq_tensor = torch.zeros((len(cat_data), 5, 4), dtype=torch.float32)

    target_tensor = torch.tensor(labels, dtype=torch.long)

    # 4. Retrain
    model = SuperSoccerNet(len(m['teams']), len(m['leagues']))
    model.load_state_dict(torch.load('soccer_model_full.pth'))
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    optimizer.zero_grad()

    outputs = model(h_idx_t, a_idx_t, c_idx_t, seq_tensor, cont_tensor)

    loss = criterion(outputs, target_tensor)
    loss.backward()
    optimizer.step()

    torch.save(model.state_dict(), 'soccer_model_full.pth')
    print(
        f"[SUCCESS] Retraining complete. Model weights updated! (Loss: {loss.item():.4f})")


if __name__ == "__main__":
    retrain_model()