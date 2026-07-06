"""Generate every statistic and figure cited in docs/RESEARCH_NOTE.md.

Run from the repo root:  python analysis/make_research_note.py
Outputs figures to docs/figures/ and prints a JSON blob of all numbers.

Nothing here is hand-entered: the note's tables are produced by this script
from (a) a fresh seeded retrain on data/soccer_data_full.csv, (b) the Poisson
baseline on the identical chronological split, (c) the committed live ledger
data/bet_history.json, and (d) the historical-odds backtest slice.
"""

# ruff: noqa: E402  -- sys.path must point at the repo root before quantbet imports

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantbet.backtest import (
    EPOCHS as BT_EPOCHS,
)
from quantbet.backtest import (
    LR as BT_LR,
)
from quantbet.backtest import (
    PATIENCE as BT_PATIENCE,
)
from quantbet.backtest import (
    TRAIN_FRAC as BT_TRAIN_FRAC,
)
from quantbet.backtest import (
    VAL_FRAC as BT_VAL_FRAC,
)
from quantbet.backtest import (
    _forward,
    _load_rows,
    implied_probabilities,
)
from quantbet.baselines import PoissonBaseline
from quantbet.config import HISTORY_FILE
from quantbet.model import Mappings, SuperSoccerNet, load_model
from quantbet.train import (
    TRAIN_FRAC,
    VAL_FRAC,
    _tensors,
    build_vocabs,
    collect_logits,
    fit_temperature,
    load_training_frame,
    set_seed,
    train,
)

FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TMP_MODEL = ROOT / "docs" / "_note_model.pth"
TMP_MAPPINGS = ROOT / "docs" / "_note_mappings.json"

B = 10_000  # bootstrap resamples
RNG = np.random.default_rng(0)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- utilities
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def norm_sf(z: float) -> float:
    """P(Z > z) for standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def per_match_logloss(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return -np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-12))


def per_match_brier(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    onehot = np.eye(3)[labels]
    return ((probs - onehot) ** 2).sum(axis=1)


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """Bootstrap CI for mean(a - b) over paired per-match losses (a=model)."""
    d = a - b
    n = len(d)
    idx = RNG.integers(0, n, size=(B, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean_diff": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "p_boot_two_sided": float(2 * min((means < 0).mean(), (means > 0).mean())),
        "n": n,
    }


# ------------------------------------------------- Part A: retrain v2 model
print("=== [A] Retraining v2 model (seed 42) ===")
train_metrics = train(model_path=TMP_MODEL, mappings_path=TMP_MAPPINGS)

set_seed()
df = load_training_frame()
team_vocab, league_vocab = build_vocabs(df)
df["h_idx"] = df["home_team"].map(team_vocab)
df["a_idx"] = df["away_team"].map(team_vocab)
df["l_idx"] = df["league_id"].astype(int).map(league_vocab)

n = len(df)
train_end = int(n * TRAIN_FRAC)
val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
train_df, test_df = df.iloc[:train_end], df.iloc[val_end:]

mappings = Mappings.load(TMP_MAPPINGS)
test_loader = DataLoader(
    _tensors(test_df, mappings.scaler_mean, mappings.scaler_scale), batch_size=512
)
net = load_model(TMP_MODEL, mappings)
logits, labels_t = collect_logits(net, test_loader)
deep_probs = torch.softmax(logits / mappings.temperature, dim=1).numpy()
labels = labels_t.numpy()

# ------------------------------------------ Part B: baselines on same slice
print("\n=== [B] Poisson + class-prior baselines on the identical holdout ===")
set_seed()
poisson = PoissonBaseline(sorted(set(df["home_team"]) | set(df["away_team"])))
poisson.fit(df.iloc[:val_end])
pois_probs = np.array(
    [poisson.predict(r.home_team, r.away_team) for r in test_df.itertuples()]
)
prior = train_df["result"].value_counts(normalize=True).sort_index().values
prior_probs = np.tile(prior, (len(test_df), 1))

losses = {
    "deep": per_match_logloss(deep_probs, labels),
    "poisson": per_match_logloss(pois_probs, labels),
    "prior": per_match_logloss(prior_probs, labels),
}
briers = {
    "deep": per_match_brier(deep_probs, labels),
    "poisson": per_match_brier(pois_probs, labels),
}

offline = {
    "n_test": int(len(labels)),
    "logloss": {k: float(v.mean()) for k, v in losses.items()},
    "brier": {k: float(v.mean()) for k, v in briers.items()},
    "accuracy": {
        "deep": float((deep_probs.argmax(1) == labels).mean()),
        "poisson": float((pois_probs.argmax(1) == labels).mean()),
    },
    "diff_logloss_deep_minus_poisson": paired_bootstrap_diff(losses["deep"], losses["poisson"]),
    "diff_logloss_deep_minus_prior": paired_bootstrap_diff(losses["deep"], losses["prior"]),
    "diff_logloss_poisson_minus_prior": paired_bootstrap_diff(losses["poisson"], losses["prior"]),
    "diff_brier_deep_minus_poisson": paired_bootstrap_diff(briers["deep"], briers["poisson"]),
    "temperature": train_metrics["temperature"],
}

# Figure: bootstrap distribution of the deep-vs-Poisson log-loss difference
d = losses["deep"] - losses["poisson"]
idx = RNG.integers(0, len(d), size=(B, len(d)))
means = d[idx].mean(axis=1)
lo, hi = np.percentile(means, [2.5, 97.5])
plt.figure(figsize=(7, 4))
plt.hist(means, bins=60, color="#4878b0", alpha=0.85)
plt.axvline(0, color="black", lw=1.5, label="no difference")
plt.axvline(lo, color="crimson", ls="--", lw=1.2, label="95% CI")
plt.axvline(hi, color="crimson", ls="--", lw=1.2)
plt.axvline(d.mean(), color="#2a2a2a", ls=":", lw=1.5, label=f"observed ({d.mean():+.4f})")
plt.xlabel("mean log-loss difference (deep − Poisson), 849-match holdout")
plt.ylabel("bootstrap frequency")
plt.title("Is the deep model's log-loss edge over the Poisson baseline real?")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "bootstrap_deep_vs_poisson.png", dpi=150)
plt.close()

# Figure: pooled 3-class reliability of the calibrated v2 model on the holdout
flat_p = deep_probs.ravel()
flat_y = np.eye(3)[labels].ravel()
bins = np.linspace(0, 1, 11)
which = np.clip(np.digitize(flat_p, bins) - 1, 0, 9)
centers, freqs, los_, his_, ns_ = [], [], [], [], []
for b_ in range(10):
    m = which == b_
    if m.sum() < 5:
        continue
    k = int(flat_y[m].sum())
    centers.append(float(flat_p[m].mean()))
    freqs.append(k / m.sum())
    w = wilson(k, int(m.sum()))
    los_.append(freqs[-1] - w[0])
    his_.append(w[1] - freqs[-1])
    ns_.append(int(m.sum()))
plt.figure(figsize=(5.5, 5.5))
plt.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="perfect calibration")
plt.errorbar(centers, freqs, yerr=[los_, his_], fmt="o-", color="#4878b0",
             capsize=3, label="v2 calibrated (holdout)")
for x, y, m_ in zip(centers, freqs, ns_, strict=True):
    plt.annotate(f"n={m_}", (x, y), textcoords="offset points", xytext=(6, -10), fontsize=7)
plt.xlabel("predicted outcome probability")
plt.ylabel("observed frequency")
plt.title("v2 model reliability after temperature scaling\n(849-match chronological holdout, all 3 outcomes pooled)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "v2_holdout_reliability.png", dpi=150)
plt.close()

# --------------------------------------- Part C: live ledger (v1 forward test)
print("\n=== [C] Live ledger statistical autopsy ===")
bets = json.load(open(HISTORY_FILE))
settled = [b for b in bets if b.get("status") in ("won", "lost")]
probs_l = np.array([float(b["model_prob"]) for b in settled])
odds_l = np.array([float(b["odds_taken"]) for b in settled])
stakes = np.array([float(b["stake"]) for b in settled])
profits = np.array([float(b["profit"]) for b in settled])
won = np.array([b["status"] == "won" for b in settled])

wins = int(won.sum())
roi = profits.sum() / stakes.sum()

# ROI bootstrap CI (resampling bets, ratio-of-sums)
idx = RNG.integers(0, len(settled), size=(B, len(settled)))
roi_boot = profits[idx].sum(axis=1) / stakes[idx].sum(axis=1)
roi_ci = np.percentile(roi_boot, [2.5, 97.5])

# Self-consistency test: were results plausible under the model's OWN claims?
exp_self = probs_l.sum()
var_self = (probs_l * (1 - probs_l)).sum()
z_self = (wins - exp_self) / math.sqrt(var_self)

# Zero-edge test: plausible under break-even probabilities 1/odds?
p_be = 1.0 / odds_l
exp_be = p_be.sum()
var_be = (p_be * (1 - p_be)).sum()
z_be = (wins - exp_be) / math.sqrt(var_be)

buckets = []
for lo_b in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
    m = (probs_l >= lo_b) & (probs_l < lo_b + 0.1)
    if m.sum() == 0:
        continue
    k = int(won[m].sum())
    w = wilson(k, int(m.sum()))
    buckets.append({
        "range": f"{lo_b:.0%}-{lo_b + 0.1:.0%}",
        "n": int(m.sum()),
        "claimed": float(probs_l[m].mean()),
        "realized": k / m.sum(),
        "wilson95": [round(w[0], 3), round(w[1], 3)],
        "claim_inside_ci": bool(w[0] <= probs_l[m].mean() <= w[1]),
    })

live = {
    "settled": len(settled),
    "wins": wins,
    "hit_rate": wins / len(settled),
    "avg_odds": float(odds_l.mean()),
    "avg_claimed_prob": float(probs_l.mean()),
    "roi": float(roi),
    "roi_ci95": [float(roi_ci[0]), float(roi_ci[1])],
    "expected_wins_under_model_claims": float(exp_self),
    "z_vs_own_claims": float(z_self),
    "p_vs_own_claims": float(2 * norm_sf(abs(z_self))),
    "expected_wins_under_break_even": float(exp_be),
    "z_vs_break_even": float(z_be),
    "p_vs_break_even": float(2 * norm_sf(abs(z_be))),
    "buckets": buckets,
}

# Figure: v1 live reliability with Wilson CIs
plt.figure(figsize=(5.5, 5.5))
plt.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="perfect calibration")
cx = [b_["claimed"] for b_ in buckets]
cy = [b_["realized"] for b_ in buckets]
yerr_lo = [b_["realized"] - b_["wilson95"][0] for b_ in buckets]
yerr_hi = [b_["wilson95"][1] - b_["realized"] for b_ in buckets]
plt.errorbar(cx, cy, yerr=[yerr_lo, yerr_hi], fmt="o", color="crimson",
             capsize=4, label="v1 live bets (95% Wilson CI)")
for b_ in buckets:
    plt.annotate(f"n={b_['n']}", (b_["claimed"], b_["realized"]),
                 textcoords="offset points", xytext=(7, -4), fontsize=8)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("model's claimed win probability at bet time")
plt.ylabel("realized win rate")
plt.title(f"v1 forward test: claimed vs realized\n({len(settled)} settled paper bets, Feb–May 2026)")
plt.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig(FIG_DIR / "v1_live_reliability.png", dpi=150)
plt.close()

# Figure: cumulative live PnL
order = np.argsort([b["timestamp"] for b in settled])
plt.figure(figsize=(7, 4))
plt.plot(np.cumsum(profits[order]), color="crimson")
plt.axhline(0, color="grey", lw=1, ls="--")
plt.xlabel("settled bet #")
plt.ylabel("cumulative paper PnL ($)")
plt.title("v1 forward test: cumulative PnL over 162 settled bets")
plt.tight_layout()
plt.savefig(FIG_DIR / "v1_cumulative_pnl.png", dpi=150)
plt.close()

# ------------------------- Part D: model vs market on the 29-match slice
print("\n=== [D] Model-vs-market comparison on the odds dataset ===")
set_seed()
rows = _load_rows(mappings)
nrows = len(rows)
tr = rows[: int(nrows * BT_TRAIN_FRAC)]
va = rows[int(nrows * BT_TRAIN_FRAC): int(nrows * (BT_TRAIN_FRAC + BT_VAL_FRAC))]
te = rows[int(nrows * (BT_TRAIN_FRAC + BT_VAL_FRAC)):]

bt_net = SuperSoccerNet(len(mappings.teams), len(mappings.leagues))
opt = optim.Adam(bt_net.parameters(), lr=BT_LR, weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss()
best_val, best_state, since = float("inf"), None, 0
for _ in range(BT_EPOCHS):
    bt_net.train()
    opt.zero_grad()
    out, y = _forward(bt_net, tr)
    loss_fn(out, y).backward()
    opt.step()
    bt_net.eval()
    with torch.no_grad():
        v_out, v_y = _forward(bt_net, va)
        v_loss = loss_fn(v_out, v_y).item()
    if v_loss < best_val - 1e-4:
        best_val, best_state, since = v_loss, {k: v.clone() for k, v in bt_net.state_dict().items()}, 0
    else:
        since += 1
    if since >= BT_PATIENCE:
        break
bt_net.load_state_dict(best_state)
bt_net.eval()
with torch.no_grad():
    v_out, v_y = _forward(bt_net, va)
    bt_T = fit_temperature(v_out, v_y)
    t_out, t_y = _forward(bt_net, te)
    bt_probs = torch.softmax(t_out / bt_T, dim=1).numpy()
bt_labels = t_y.numpy()
mkt_probs = np.array([implied_probabilities(r["odds"]) for r in te])

bt_model_ll = per_match_logloss(bt_probs, bt_labels)
bt_mkt_ll = per_match_logloss(mkt_probs, bt_labels)
market = {
    "n_test": int(len(te)),
    "model_logloss": float(bt_model_ll.mean()),
    "market_logloss": float(bt_mkt_ll.mean()),
    "diff_model_minus_market": paired_bootstrap_diff(bt_model_ll, bt_mkt_ll),
}

# ---------------------------------------------------------------- summary
results = {"offline": offline, "live_v1": live, "market": market}
out_path = ROOT / "docs" / "research_note_stats.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print("\n=== ALL RESULTS ===")
print(json.dumps(results, indent=2))
print(f"\nSaved stats -> {out_path}\nFigures -> {FIG_DIR}")
TMP_MODEL.unlink(missing_ok=True)
TMP_MAPPINGS.unlink(missing_ok=True)