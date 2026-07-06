"""Nightly fine-tuning on settled bets, with an experience-replay buffer.

Fixes over the original implementation:
- Trains on the *feature snapshot stored at bet time* rather than re-fetching
  each team's current form for a match played weeks earlier (look-ahead bias).
  Legacy ledger rows without a snapshot are skipped.
- Marks consumed rows (`used_in_training`) so the same bets aren't re-learned
  every night — the old loop re-trained on the full ledger daily, drifting
  the weights toward a tiny, model-selected sample.

Known caveat, documented rather than hidden: bets in the ledger are chosen
by the model itself, so this data is not an unbiased sample of matches. The
replay buffer of random historical matches limits, but does not eliminate,
that feedback loop.
"""

import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import MAPPINGS_FILE, MODEL_FILE, TRAINING_CSV
from .features import Outcome
from .ledger import load_ledger, save_ledger
from .model import Mappings, load_model
from .train import load_training_frame

log = logging.getLogger(__name__)

MIN_NEW_BETS = 4
REPLAY_SIZE = 400  # historical matches mixed in to anchor the weights
FINE_TUNE_EPOCHS = 3
FINE_TUNE_LR = 1e-4
RESULT_TO_LABEL = {"Home": Outcome.HOME, "Draw": Outcome.DRAW, "Away": Outcome.AWAY}


def _ledger_samples(history: list[dict], mappings: Mappings) -> tuple[list, list[dict]]:
    """Extract (cat, cont, seq, label) rows from settled, unconsumed bets."""
    samples, consumed = [], []
    for bet in history:
        if bet.get("status") not in ("won", "lost") or bet.get("used_in_training"):
            continue
        snapshot = bet.get("features")
        label = RESULT_TO_LABEL.get(bet.get("actual_result", ""))
        if snapshot is None or label is None:
            continue  # legacy row: no leak-free features available
        if bet["home_team"] not in mappings.teams or bet["away_team"] not in mappings.teams:
            continue
        samples.append(
            (
                [
                    mappings.teams[bet["home_team"]],
                    mappings.teams[bet["away_team"]],
                    mappings.leagues.get(int(bet["league_id"]), 0),
                ],
                snapshot["cont"],
                snapshot["seq"],
                int(label),
            )
        )
        consumed.append(bet)
    return samples, consumed


def _replay_samples(mappings: Mappings, size: int = REPLAY_SIZE) -> list:
    # load_training_frame already attaches leak-free "seq_data" per row.
    df = load_training_frame(TRAINING_CSV)
    if len(df) > size:
        df = df.sample(n=size, random_state=random.randrange(1 << 30))

    samples = []
    for row in df.itertuples():
        if row.home_team not in mappings.teams or row.away_team not in mappings.teams:
            continue
        scaled = (
            np.array([row.home_form, row.away_form, row.home_gd, row.away_gd])
            - mappings.scaler_mean[:4]
        ) / mappings.scaler_scale[:4]
        cont = np.concatenate([scaled, [row.home_importance, row.away_importance]])
        samples.append(
            (
                [
                    mappings.teams[row.home_team],
                    mappings.teams[row.away_team],
                    mappings.leagues.get(int(row.league_id), 0),
                ],
                cont.tolist(),
                row.seq_data,
                int(row.result),
            )
        )
    return samples


def retrain() -> None:
    print("\n--- ML RETRAINER (experience replay) ---")
    mappings = Mappings.load(MAPPINGS_FILE)
    history = load_ledger()

    new_samples, consumed = _ledger_samples(history, mappings)
    if len(new_samples) < MIN_NEW_BETS:
        print(f"[INFO] Only {len(new_samples)} new settled bet(s); need {MIN_NEW_BETS}. Skipping.")
        return

    replay = _replay_samples(mappings)
    samples = new_samples + replay
    random.shuffle(samples)
    print(f"[INFO] Fine-tuning on {len(new_samples)} new + {len(replay)} replay matches.")

    cats = torch.tensor([s[0] for s in samples], dtype=torch.long)
    cont = torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32)
    seqs = torch.tensor(np.array([s[2] for s in samples]), dtype=torch.float32)
    labels = torch.tensor([s[3] for s in samples], dtype=torch.long)

    net = load_model(MODEL_FILE, mappings)
    net.train()
    opt = optim.Adam(net.parameters(), lr=FINE_TUNE_LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(FINE_TUNE_EPOCHS):
        opt.zero_grad()
        loss = loss_fn(net(cats[:, 0], cats[:, 1], cats[:, 2], seqs, cont), labels)
        loss.backward()
        opt.step()
        log.info("fine-tune epoch %d | loss %.4f", epoch, loss.item())

    torch.save(net.state_dict(), MODEL_FILE)

    for bet in consumed:
        bet["used_in_training"] = True
    save_ledger(history)
    print(f"[SUCCESS] Weights updated (final loss {loss.item():.4f}).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    retrain()
