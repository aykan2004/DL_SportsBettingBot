import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from model import SuperSoccerNet


def build_core_model():
    # TODO: The API sometimes crashes on Turkish characters (e.g., Fenerbahçe). Forcing utf-8.
    df = pd.read_csv('soccer_data_full.csv', encoding='utf-8')

    # Late season games where neither team has anything to play for (importance < 0.2)
    # usually feature heavily rotated squads and ruin the form metrics. Trash them.
    init_len = len(df)
    df = df[(df['home_importance'] > 0.2) | (df['away_importance'] > 0.2)]
    print(f"Dropped {init_len - len(df)} dead rubber matches.")

    # Drop API ghosts and postponed matches
    df = df.dropna(subset=['result', 'home_team', 'away_team'])

    # Build vocabularies
    all_clubs = sorted(pd.concat([df['home_team'], df['away_team']]).unique())
    club_vocab = {club: i for i, club in enumerate(all_clubs)}
    comp_vocab = {comp: i for i, comp in enumerate(df['league_id'].unique())}

    df['h_idx'] = df['home_team'].map(club_vocab)
    df['a_idx'] = df['away_team'].map(club_vocab)
    df['comp_idx'] = df['league_id'].map(comp_vocab)

    fixture_cats = df[['h_idx', 'a_idx', 'comp_idx']].values.astype(int)

    df['home_gd'] = df['home_gd'].clip(-4, 4)
    df['away_gd'] = df['away_gd'].clip(-4, 4)

    raw_form = df[['home_form', 'away_form', 'home_gd', 'away_gd']].values

    scaler = StandardScaler()
    scaled_form = scaler.fit_transform(raw_form)

    # Motivation metrics
    motivation = df[['home_importance', 'away_importance']].values

    # Smash continuous features together
    fixture_conts = np.hstack((scaled_form, motivation))
    match_outcomes = df['result'].values

    # --- NEW: DUMMY SEQUENCE FOR LSTM ---
    # Shape: (Total Matches, 5 matches in history, 4 stats per match)
    fixture_seqs = np.zeros((len(df), 5, 4))

    with open('mappings.json', 'w') as f:
        json.dump({
            'teams': club_vocab,
            'leagues': {str(k): v for k, v in comp_vocab.items()},
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist()
        }, f, indent=4)

    # --- CHRONOLOGICAL SPLIT ---
    # Sklearn's random split causes future-data leakage.
    # We must train on the past to predict the future.
    split_idx = int(len(df) * 0.8)

    train_cats = torch.LongTensor(fixture_cats[:split_idx])
    train_seqs = torch.FloatTensor(fixture_seqs[:split_idx])
    train_conts = torch.FloatTensor(fixture_conts[:split_idx])
    train_labels = torch.LongTensor(match_outcomes[:split_idx])

    test_cats = torch.LongTensor(fixture_cats[split_idx:])
    test_seqs = torch.FloatTensor(fixture_seqs[split_idx:])
    test_conts = torch.FloatTensor(fixture_conts[split_idx:])
    test_labels = torch.LongTensor(match_outcomes[split_idx:])

    net = SuperSoccerNet(len(club_vocab), len(comp_vocab))
    opt = optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    print("Cracking the weights...")

    for epoch in range(100):
        net.train()
        opt.zero_grad()

        h_idx = train_cats[:, 0]
        a_idx = train_cats[:, 1]
        c_idx = train_cats[:, 2]

        # Forward pass with the sequence data added
        out = net(h_idx, a_idx, c_idx, train_seqs, train_conts)

        loss = loss_fn(out, train_labels)
        loss.backward()
        opt.step()

        if epoch in [0, 49, 99]:
            print(f"Ep {epoch:02d} | Loss: {loss.item():.3f}")

    net.eval()
    with torch.no_grad():
        h_idx_test = test_cats[:, 0]
        a_idx_test = test_cats[:, 1]
        c_idx_test = test_cats[:, 2]

        val_out = net(h_idx_test, a_idx_test, c_idx_test, test_seqs,
                      test_conts)
        hits = (torch.argmax(val_out, dim=1) == test_labels).sum().item()
        print(
            f"Holdout Hit Rate (Chronological): {(hits / len(test_labels)) * 100:.1f}%")

    torch.save(net.state_dict(), 'soccer_model_full.pth')


if __name__ == '__main__':
    build_core_model()