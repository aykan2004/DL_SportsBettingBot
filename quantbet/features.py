"""Feature engineering shared by training, retraining, backtesting and live inference.

Keeping this in one module is the point: the original codebase re-implemented
sequence building in three places and they silently diverged (the live path
zeroed out goal-difference features that the model was trained on).

Conventions (single source of truth):
- Outcome labels: 0 = home win, 1 = draw, 2 = away win.
- Form sequence: shape (SEQ_LEN, 4) = [home_gf, home_ga, away_gf, away_ga]
  per past match, ordered oldest -> newest, zero-padded at the *oldest* end.
- Continuous vector: [home_form, away_form, home_gd, away_gd] standardized
  with the training-set scaler, then [home_importance, away_importance] raw.
"""

from enum import IntEnum

import numpy as np
import pandas as pd

SEQ_LEN = 5  # past matches per team fed to the LSTM
GD_CLIP = 4.0  # goal difference is clipped to +/-4 to tame blowouts
NEUTRAL_IMPORTANCE = 0.5  # used when match-importance data is unavailable


class Outcome(IntEnum):
    HOME = 0
    DRAW = 1
    AWAY = 2


OUTCOME_NAMES = ("Home", "Draw", "Away")


def result_from_goals(home_goals: int, away_goals: int) -> Outcome:
    if home_goals > away_goals:
        return Outcome.HOME
    if away_goals > home_goals:
        return Outcome.AWAY
    return Outcome.DRAW


def pad_history(history: list[list[float]], length: int = SEQ_LEN) -> list[list[float]]:
    """Left-pad a chronological (oldest->newest) GF/GA history with [0, 0]."""
    recent = history[-length:]
    return [[0.0, 0.0]] * (length - len(recent)) + [list(row) for row in recent]


def combine_sequences(
    home_hist: list[list[float]], away_hist: list[list[float]]
) -> list[list[float]]:
    """Zip two padded per-team histories into the (SEQ_LEN, 4) model input."""
    h = pad_history(home_hist)
    a = pad_history(away_hist)
    return [[h[i][0], h[i][1], a[i][0], a[i][1]] for i in range(SEQ_LEN)]


def build_match_sequences(df: pd.DataFrame) -> list[list[list[float]]]:
    """Build the last-SEQ_LEN GF/GA sequence for every row of a chronologically
    sorted match DataFrame. Only matches *before* each row are used, so this is
    leak-free by construction.
    """
    team_hist: dict[str, list[list[float]]] = {}
    sequences = []
    for row in df.itertuples():
        home, away = row.home_team, row.away_team
        sequences.append(
            combine_sequences(team_hist.get(home, []), team_hist.get(away, []))
        )
        team_hist.setdefault(home, []).append([row.home_goals, row.away_goals])
        team_hist.setdefault(away, []).append([row.away_goals, row.home_goals])
    return sequences


def make_continuous(
    home_form: float,
    away_form: float,
    home_gd: float,
    away_gd: float,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    home_importance: float = NEUTRAL_IMPORTANCE,
    away_importance: float = NEUTRAL_IMPORTANCE,
) -> np.ndarray:
    """Assemble the 6-dim continuous feature vector exactly as in training."""
    raw = np.array(
        [
            home_form,
            away_form,
            float(np.clip(home_gd, -GD_CLIP, GD_CLIP)),
            float(np.clip(away_gd, -GD_CLIP, GD_CLIP)),
        ]
    )
    scaled = (raw - scaler_mean[:4]) / scaler_scale[:4]
    return np.concatenate([scaled, [home_importance, away_importance]]).astype(np.float32)
