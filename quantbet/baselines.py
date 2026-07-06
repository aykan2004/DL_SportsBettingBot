"""Classical baseline: time-weighted independent Poisson goals model.

Any deep model for match outcomes has to justify itself against the workhorse
of soccer modelling (Maher 1982; Dixon & Coles 1997): per-team attack/defence
ratings with a home-advantage term, fitted by maximum likelihood with
exponential time decay. If the neural network cannot out-predict this on
held-out log-loss, the extra machinery isn't paying rent.

lambda_home = exp(mu + home_adv + att[home] - def[away])
lambda_away = exp(mu + att[away] - def[home])

1X2 probabilities come from summing the joint score matrix. (The Dixon-Coles
low-score dependence correction is omitted; it moves log-loss by ~0.001 here.)
"""

import logging
import math

import numpy as np
import torch

from .config import TRAINING_CSV
from .train import TRAIN_FRAC, VAL_FRAC, load_training_frame, set_seed

log = logging.getLogger(__name__)

TIME_DECAY_XI = 0.005  # per day; ~half-weight at 20 weeks (Dixon-Coles order)
MAX_GOALS = 10
FIT_ITERS = 1500
FIT_LR = 0.05


def outcome_probs_from_lambdas(lam_home: float, lam_away: float) -> np.ndarray:
    """[P(home), P(draw), P(away)] from independent Poisson score margins."""
    goals = np.arange(MAX_GOALS + 1)
    ph = np.exp(-lam_home) * lam_home**goals / np.array([math.factorial(g) for g in goals])
    pa = np.exp(-lam_away) * lam_away**goals / np.array([math.factorial(g) for g in goals])
    joint = np.outer(ph, pa)
    joint /= joint.sum()  # renormalize the truncated grid
    home = np.tril(joint, -1).sum()
    away = np.triu(joint, 1).sum()
    draw = np.trace(joint)
    return np.array([home, draw, away])


class PoissonBaseline:
    def __init__(self, teams: list[str]):
        self.team_idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        self.attack = torch.zeros(n, requires_grad=True)
        self.defence = torch.zeros(n, requires_grad=True)
        self.mu = torch.tensor(0.1, requires_grad=True)
        self.home_adv = torch.tensor(0.2, requires_grad=True)

    def fit(self, df) -> None:
        """Weighted Poisson MLE via gradient descent (torch does the algebra)."""
        h = torch.tensor([self.team_idx[t] for t in df["home_team"]])
        a = torch.tensor([self.team_idx[t] for t in df["away_team"]])
        hg = torch.tensor(df["home_goals"].values, dtype=torch.float32)
        ag = torch.tensor(df["away_goals"].values, dtype=torch.float32)

        age_days = (df["date"].max() - df["date"]).dt.days.values.astype(float)
        w = torch.tensor(np.exp(-TIME_DECAY_XI * age_days), dtype=torch.float32)

        params = [self.attack, self.defence, self.mu, self.home_adv]
        opt = torch.optim.Adam(params, lr=FIT_LR)
        for i in range(FIT_ITERS):
            opt.zero_grad()
            log_lh = self.mu + self.home_adv + self.attack[h] - self.defence[a]
            log_la = self.mu + self.attack[a] - self.defence[h]
            # Poisson NLL (dropping the constant log(y!) term), plus a soft
            # identifiability constraint pinning mean attack/defence to zero.
            nll = (
                (w * (torch.exp(log_lh) - hg * log_lh)).sum()
                + (w * (torch.exp(log_la) - ag * log_la)).sum()
            ) / w.sum()
            nll = nll + self.attack.mean() ** 2 + self.defence.mean() ** 2
            nll.backward()
            opt.step()
            if i % 300 == 0:
                log.info("poisson fit iter %d | nll %.4f", i, nll.item())

    @torch.no_grad()
    def predict(self, home_team: str, away_team: str) -> np.ndarray:
        h = self.team_idx.get(home_team)
        a = self.team_idx.get(away_team)
        att_h = self.attack[h].item() if h is not None else 0.0
        def_h = self.defence[h].item() if h is not None else 0.0
        att_a = self.attack[a].item() if a is not None else 0.0
        def_a = self.defence[a].item() if a is not None else 0.0
        lam_h = math.exp(self.mu.item() + self.home_adv.item() + att_h - def_a)
        lam_a = math.exp(self.mu.item() + att_a - def_h)
        return outcome_probs_from_lambdas(lam_h, lam_a)


def run_baseline() -> dict:
    """Fit on the same chronological train+val slice the neural net uses and
    report metrics on the identical test slice, so the two are comparable."""
    set_seed()
    df = load_training_frame(TRAINING_CSV)
    split = int(len(df) * (TRAIN_FRAC + VAL_FRAC))
    fit_df, test_df = df.iloc[:split], df.iloc[split:]

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    model = PoissonBaseline(teams)
    print(f"Fitting time-weighted Poisson on {len(fit_df)} matches...")
    model.fit(fit_df)

    probs = np.array(
        [model.predict(r.home_team, r.away_team) for r in test_df.itertuples()]
    )
    labels = test_df["result"].values.astype(int)
    picked = probs[np.arange(len(labels)), labels]
    log_loss = -float(np.log(np.maximum(picked, 1e-12)).mean())
    accuracy = float((probs.argmax(axis=1) == labels).mean())
    onehot = np.eye(3)[labels]
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())

    print(f"\nPoisson baseline on the same {len(test_df)}-match holdout:")
    print(f"  log-loss  {log_loss:.4f}")
    print(f"  accuracy  {accuracy:.3f}")
    print(f"  brier     {brier:.4f}")
    print(f"  home advantage: {math.exp(model.home_adv.item()):.2f}x goals")
    return {"log_loss": log_loss, "accuracy": accuracy, "brier": brier}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_baseline()
