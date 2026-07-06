"""Backtest the modelling + staking approach on a historical odds dataset.

The dataset (data/historical_dataset_2024.json, built by build_dataset.py)
pairs each match with pre-kickoff 1X2 odds snapshots, so we can evaluate the
model against the market — not just against the scoreboard.

What was wrong with the original backtest, all fixed here:
- Labels were encoded 0=H/1=A/2=D while the model and odds arrays assumed
  0=H/1=D/2=A, so wins/losses were graded against scrambled outcomes.
- The model was called with 2 arguments after the architecture grew to 5;
  the script simply crashed.
- Bets were picked by raw argmax confidence with no reference to price.
  Here selection is EV-based with fractional-Kelly staking, matching the
  live strategy.

Honest limitation: this dataset stores aggregate 5-match form but not the
per-match GF/GA sequences, so the LSTM branch receives zeros consistently in
both training and evaluation. The backtest therefore measures the
embeddings + form model, a slightly weaker sibling of the live model.
"""

import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import HISTORICAL_ODDS_JSON, MAPPINGS_FILE, REPORTS_DIR
from .features import SEQ_LEN
from .model import Mappings, SuperSoccerNet
from .strategy import VALUE, StrategyProfile, expected_value
from .train import fit_temperature, set_seed

log = logging.getLogger(__name__)

EPOCHS = 200
PATIENCE = 20
LR = 1e-3
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2
START_BANKROLL = 1000.0


def _load_rows(mappings: Mappings) -> list[dict]:
    with open(HISTORICAL_ODDS_JSON) as f:
        rows = json.load(f)
    for r in rows:
        f_ = r["features"]
        scaled = (
            np.array([f_["h_pts"], f_["a_pts"], f_["h_gd"], f_["a_gd"]])
            - mappings.scaler_mean[:4]
        ) / mappings.scaler_scale[:4]
        r["cont"] = np.concatenate([scaled, [f_["h_imp"], f_["a_imp"]]]).astype(np.float32)
        r["l_idx"] = mappings.leagues.get(int(f_["l_id"]), 0)
    return rows


def _batch(rows: list[dict]):
    cats = torch.tensor(
        [[r["features"]["h_id"], r["features"]["a_id"], r["l_idx"]] for r in rows],
        dtype=torch.long,
    )
    cont = torch.tensor(np.array([r["cont"] for r in rows]), dtype=torch.float32)
    seqs = torch.zeros(len(rows), SEQ_LEN, 4)  # see module docstring
    labels = torch.tensor([r["label"] for r in rows], dtype=torch.long)
    return cats, seqs, cont, labels


def _forward(net, rows):
    cats, seqs, cont, labels = _batch(rows)
    return net(cats[:, 0], cats[:, 1], cats[:, 2], seqs, cont), labels


def implied_probabilities(odds: list[float]) -> np.ndarray:
    """Bookmaker probabilities with the overround removed (basic normalization)."""
    inv = np.array([1.0 / o if o > 1.0 else 0.0 for o in odds])
    return inv / inv.sum() if inv.sum() > 0 else np.full(3, 1 / 3)


def log_loss_of(probs: np.ndarray, labels: np.ndarray) -> float:
    return -float(np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-12)).mean())


def simulate_bankroll(
    probs: np.ndarray, rows: list[dict], profile: StrategyProfile = VALUE
) -> dict:
    bankroll = START_BANKROLL
    peak = bankroll
    max_drawdown = 0.0
    curve = [bankroll]
    n_bets = wins = 0

    for p_row, row in zip(probs, rows, strict=True):
        # Pick the single best +EV outcome per match, if any clears filters.
        best_i, best_ev = None, 0.0
        for i in range(3):
            odds_i = row["odds"][i]
            if profile.accepts(float(p_row[i]), float(odds_i)):
                ev = expected_value(float(p_row[i]), float(odds_i))
                if ev > best_ev:
                    best_i, best_ev = i, ev
        if best_i is None:
            continue

        stake = bankroll * profile.stake_fraction(float(p_row[best_i]), float(row["odds"][best_i]))
        n_bets += 1
        if best_i == row["label"]:
            wins += 1
            bankroll += stake * (row["odds"][best_i] - 1.0)
        else:
            bankroll -= stake
        peak = max(peak, bankroll)
        max_drawdown = max(max_drawdown, (peak - bankroll) / peak)
        curve.append(bankroll)

    return {
        "bets": n_bets,
        "hit_rate": wins / n_bets if n_bets else 0.0,
        "final_bankroll": bankroll,
        "roi": (bankroll - START_BANKROLL) / START_BANKROLL,
        "max_drawdown": max_drawdown,
        "curve": curve,
    }


def run_backtest(profile: StrategyProfile = VALUE) -> dict:
    set_seed()
    mappings = Mappings.load(MAPPINGS_FILE)
    rows = _load_rows(mappings)

    n = len(rows)
    train_rows = rows[: int(n * TRAIN_FRAC)]
    val_rows = rows[int(n * TRAIN_FRAC) : int(n * (TRAIN_FRAC + VAL_FRAC))]
    test_rows = rows[int(n * (TRAIN_FRAC + VAL_FRAC)) :]
    print(f"Backtest dataset: {n} matches ({len(train_rows)} train / "
          f"{len(val_rows)} val / {len(test_rows)} test, chronological).")

    net = SuperSoccerNet(len(mappings.teams), len(mappings.leagues))
    opt = optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state, since_best = float("inf"), None, 0
    for _epoch in range(EPOCHS):
        net.train()
        opt.zero_grad()
        out, labels = _forward(net, train_rows)
        loss = loss_fn(out, labels)
        loss.backward()
        opt.step()

        net.eval()
        with torch.no_grad():
            val_out, val_labels = _forward(net, val_rows)
            val_loss = loss_fn(val_out, val_labels).item()
        if val_loss < best_val - 1e-4:
            best_val, best_state, since_best = val_loss, {
                k: v.clone() for k, v in net.state_dict().items()
            }, 0
        else:
            since_best += 1
        if since_best >= PATIENCE:
            break

    net.load_state_dict(best_state)
    net.eval()

    with torch.no_grad():
        val_out, val_labels = _forward(net, val_rows)
        temperature = fit_temperature(val_out, val_labels)
        test_out, test_labels = _forward(net, test_rows)
        probs = torch.softmax(test_out / temperature, dim=1).numpy()

    labels_np = test_labels.numpy()
    market_probs = np.array([implied_probabilities(r["odds"]) for r in test_rows])

    model_ll = log_loss_of(probs, labels_np)
    market_ll = log_loss_of(market_probs, labels_np)
    sim = simulate_bankroll(probs, test_rows, profile)

    print(f"\n--- Probability quality (test, n={len(test_rows)}) ---")
    print(f"model log-loss:  {model_ll:.4f} (T={temperature:.2f})")
    print(f"market log-loss: {market_ll:.4f}  <- the bar to beat")
    print(f"model {'beats' if model_ll < market_ll else 'does NOT beat'} the market baseline.")

    print(f"\n--- Bankroll simulation (profile: {profile.name}) ---")
    print(f"bets: {sim['bets']} | hit rate: {sim['hit_rate']:.1%} | "
          f"ROI: {sim['roi']:+.1%} | max drawdown: {sim['max_drawdown']:.1%}")
    if sim["bets"] < 100:
        print(f"[CAVEAT] Only {sim['bets']} bets — far too few for statistical "
              "significance; treat as a pipeline demonstration, not an edge claim.")

    _save_plots(sim["curve"])
    return {"model_log_loss": model_ll, "market_log_loss": market_ll,
            "temperature": temperature, "sim": sim}


def _save_plots(curve: list[float]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    REPORTS_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(curve)
    plt.xlabel("Bet #")
    plt.ylabel("Bankroll ($)")
    plt.title("Backtest equity curve (test slice)")
    plt.tight_layout()
    out = REPORTS_DIR / "backtest_equity_curve.png"
    plt.savefig(out)
    plt.close()
    print(f"Equity curve saved -> {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_backtest()
