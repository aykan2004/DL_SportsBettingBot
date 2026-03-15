# retrain_bot.py
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import os

# --- MODULAR IMPORTS ---
from config import HISTORY_FILE
from model import SuperSoccerNet


def retrain_model():
    print("\n--- DAILY AUTOMATION: ML RETRAINER ---")

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

    # Batch Norm Fix
    if len(trainable_bets) < 2:
        print(f"[INFO] Found {len(trainable_bets)} finished bet(s).")
        print(
            "[WARNING] The neural network requires at least 2 finished bets to calculate batch variance.")
        print("[INFO] Going back to sleep until more matches finish.")
        return

    print(
        f"[INFO] Found {len(trainable_bets)} actionable bets. (Skipped {skipped_bets} pending/refunded).")

    cat_data = []
    labels = []

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

    if not cat_data:
        print(
            "[WARNING] No valid team mappings found in the ledger. Aborting.")
        return

    cat_tensor = torch.tensor(cat_data, dtype=torch.long)
    cont_tensor = torch.zeros((len(cat_data), 6), dtype=torch.float32)
    target_tensor = torch.tensor(labels, dtype=torch.long)

    # Load Model cleanly from model.py
    model = SuperSoccerNet(len(m['teams']), len(m['leagues']))
    model.load_state_dict(torch.load('soccer_model_full.pth'))
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = model(cat_tensor, cont_tensor)
    loss = criterion(outputs, target_tensor)
    loss.backward()
    optimizer.step()

    torch.save(model.state_dict(), 'soccer_model_full.pth')
    print(
        f"[SUCCESS] Retraining complete. Model weights updated! (Loss: {loss.item():.4f})")


if __name__ == "__main__":
    retrain_model()