"""Train the outcome model from scratch on the historical match CSV.

Methodology notes (the details that make results trustworthy):
- Chronological split (train -> val -> test). A random split lets the model
  "predict" matches using information from later matches by the same teams.
- The feature scaler is fit on the *training* slice only. Fitting it on the
  full dataset (the original implementation) leaks test-set statistics.
- Early stopping on validation log-loss, not accuracy: we bet on
  probabilities, so probability quality is the objective.
- Temperature scaling fit on the validation set. An uncalibrated softmax
  destroys a betting strategy even when accuracy is fine — EV thresholds
  compare model probabilities directly against market prices.
"""

import argparse
import copy
import logging
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .config import MAPPINGS_FILE, MODEL_FILE, TRAINING_CSV
from .features import GD_CLIP, build_match_sequences
from .model import Mappings, SuperSoccerNet

log = logging.getLogger(__name__)

SEED = 42
MAX_EPOCHS = 200
PATIENCE = 15
BATCH_SIZE = 128
LR = 5e-4
WEIGHT_DECAY = 1e-5
TRAIN_FRAC, VAL_FRAC = 0.70, 0.10  # remainder is the test slice
CONT_COLS = ["home_form", "away_form", "home_gd", "away_gd"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_training_frame(csv_path=TRAINING_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)

    # Sequences must be built before any row filtering so each team's rolling
    # window still sees all of its past matches.
    df["seq_data"] = build_match_sequences(df)

    # Dead rubbers (nothing at stake for either side) feature heavily rotated
    # squads and add label noise.
    before = len(df)
    df = df[(df["home_importance"] > 0.2) | (df["away_importance"] > 0.2)]
    df = df.dropna(subset=["result", "home_team", "away_team"]).reset_index(drop=True)
    log.info("Dropped %d dead-rubber/invalid rows.", before - len(df))

    df["home_gd"] = df["home_gd"].clip(-GD_CLIP, GD_CLIP)
    df["away_gd"] = df["away_gd"].clip(-GD_CLIP, GD_CLIP)
    return df


def build_vocabs(df: pd.DataFrame) -> tuple[dict[str, int], dict[int, int]]:
    # Vocabularies cover the full dataset so live inference can index any team
    # we have ever seen. Embeddings for teams that only appear late in the
    # data get few gradient updates — a known small-data limitation.
    teams = sorted(pd.concat([df["home_team"], df["away_team"]]).unique())
    team_vocab = {t: i for i, t in enumerate(teams)}
    league_vocab = {int(league): i for i, league in enumerate(sorted(df["league_id"].unique()))}
    return team_vocab, league_vocab


def _tensors(df: pd.DataFrame, mean: np.ndarray, scale: np.ndarray) -> TensorDataset:
    cats = torch.LongTensor(df[["h_idx", "a_idx", "l_idx"]].values)
    scaled = (df[CONT_COLS].values - mean) / scale
    cont = torch.FloatTensor(
        np.hstack([scaled, df[["home_importance", "away_importance"]].values])
    )
    seqs = torch.FloatTensor(np.array(df["seq_data"].tolist()))
    labels = torch.LongTensor(df["result"].values.astype(int))
    return TensorDataset(cats, seqs, cont, labels)


@torch.no_grad()
def collect_logits(net: SuperSoccerNet, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    net.eval()
    logits, labels = [], []
    for cats, seqs, cont, y in loader:
        logits.append(net(cats[:, 0], cats[:, 1], cats[:, 2], seqs, cont))
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Post-hoc calibration (Guo et al. 2017): one scalar T minimizing val NLL."""
    log_t = torch.zeros(1, requires_grad=True)
    nll = nn.CrossEntropyLoss()
    opt = optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nll(logits / torch.exp(log_t), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).item())


def evaluate(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> dict:
    probs = torch.softmax(logits / temperature, dim=1)
    n = len(labels)
    onehot = torch.zeros_like(probs).scatter_(1, labels.unsqueeze(1), 1.0)
    return {
        "n": n,
        "accuracy": (probs.argmax(dim=1) == labels).float().mean().item(),
        "log_loss": nn.functional.nll_loss(torch.log(probs.clamp_min(1e-12)), labels).item(),
        "brier": ((probs - onehot) ** 2).sum(dim=1).mean().item(),
    }


def train(csv_path=TRAINING_CSV, model_path=MODEL_FILE, mappings_path=MAPPINGS_FILE) -> dict:
    set_seed()
    df = load_training_frame(csv_path)
    team_vocab, league_vocab = build_vocabs(df)

    df["h_idx"] = df["home_team"].map(team_vocab)
    df["a_idx"] = df["away_team"].map(team_vocab)
    df["l_idx"] = df["league_id"].astype(int).map(league_vocab)

    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    train_df, val_df, test_df = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

    # Scaler statistics come from the training slice only.
    mean = train_df[CONT_COLS].values.mean(axis=0)
    scale = train_df[CONT_COLS].values.std(axis=0)
    scale[scale == 0] = 1.0

    train_loader = DataLoader(_tensors(train_df, mean, scale), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(_tensors(val_df, mean, scale), batch_size=512)
    test_loader = DataLoader(_tensors(test_df, mean, scale), batch_size=512)

    net = SuperSoccerNet(len(team_vocab), len(league_vocab))
    opt = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state, since_best = float("inf"), None, 0
    print(f"Training on {len(train_df)} matches (val {len(val_df)}, test {len(test_df)})...")

    for epoch in range(MAX_EPOCHS):
        net.train()
        for cats, seqs, cont, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(net(cats[:, 0], cats[:, 1], cats[:, 2], seqs, cont), y)
            loss.backward()
            opt.step()

        val_logits, val_labels = collect_logits(net, val_loader)
        val_loss = loss_fn(val_logits, val_labels).item()
        if val_loss < best_val - 1e-4:
            best_val, best_state, since_best = val_loss, copy.deepcopy(net.state_dict()), 0
        else:
            since_best += 1
        if epoch % 10 == 0 or since_best == 0:
            log.info("epoch %03d | val log-loss %.4f%s", epoch, val_loss, " *" if since_best == 0 else "")
        if since_best >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best val log-loss {best_val:.4f}).")
            break

    net.load_state_dict(best_state)

    val_logits, val_labels = collect_logits(net, val_loader)
    temperature = fit_temperature(val_logits.detach(), val_labels)

    test_logits, test_labels = collect_logits(net, test_loader)
    raw = evaluate(test_logits, test_labels)
    cal = evaluate(test_logits, test_labels, temperature)
    prior = train_df["result"].value_counts(normalize=True).sort_index().values
    prior_ll = -float(np.log(np.maximum(prior[test_labels.numpy()], 1e-12)).mean())

    print(f"\nTemperature: {temperature:.3f}")
    print(f"Test ({raw['n']} matches, chronological holdout):")
    print(f"  accuracy      {raw['accuracy']:.3f}")
    print(f"  log-loss raw  {raw['log_loss']:.4f} -> calibrated {cal['log_loss']:.4f}")
    print(f"  brier raw     {raw['brier']:.4f} -> calibrated {cal['brier']:.4f}")
    print(f"  class-prior baseline log-loss: {prior_ll:.4f}")

    torch.save(net.state_dict(), model_path)
    Mappings(team_vocab, league_vocab, mean, scale, temperature).save(mappings_path)
    print(f"\nSaved weights -> {model_path}\nSaved mappings -> {mappings_path}")

    return {"temperature": temperature, "test_raw": raw, "test_calibrated": cal,
            "baseline_log_loss": prior_ll}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=TRAINING_CSV)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    train(csv_path=args.csv)
