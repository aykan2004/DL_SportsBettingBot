"""Model architecture and checkpoint helpers."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .features import SEQ_LEN


class SuperSoccerNet(nn.Module):
    """Hybrid net for 1X2 match-outcome probabilities.

    Three input branches:
    - Entity embeddings for the home team, away team and competition, which let
      the network learn latent team strength beyond the handcrafted features.
    - An LSTM over each side's last SEQ_LEN matches (GF/GA for both teams per
      step), capturing momentum that a summed form score throws away.
    - A 6-dim continuous vector (standardized form/GD + match importance).

    The branches are concatenated (16+16+4+16+6 = 58) and passed through a
    small MLP with BatchNorm and Dropout. Output is 3 raw logits; calibrated
    probabilities come from softmax(logits / temperature) — see train.py.
    """

    def __init__(self, num_teams: int, num_leagues: int):
        super().__init__()
        self.team_embed = nn.Embedding(num_teams, 16)
        self.league_embed = nn.Embedding(num_leagues, 4)
        self.lstm = nn.LSTM(input_size=4, hidden_size=16, batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(58, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 3),
        )

    def forward(self, h_idx, a_idx, l_idx, seq_data, cont_data):
        h_emb = self.team_embed(h_idx)
        a_emb = self.team_embed(a_idx)
        l_emb = self.league_embed(l_idx)

        # seq_data: (batch, SEQ_LEN, 4); keep the final hidden state.
        _, (h_n, _) = self.lstm(seq_data)
        lstm_h = h_n.squeeze(0)

        combined = torch.cat([h_emb, a_emb, l_emb, lstm_h, cont_data], dim=1)
        return self.fc_layers(combined)


class Mappings:
    """Everything inference needs besides the weights: vocabularies, the
    training-set scaler, and the calibration temperature."""

    def __init__(
        self,
        teams: dict[str, int],
        leagues: dict[int, int],
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        temperature: float = 1.0,
    ):
        self.teams = teams
        self.leagues = leagues
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale
        self.temperature = temperature

    @classmethod
    def load(cls, path: Path) -> "Mappings":
        with open(path) as f:
            m = json.load(f)
        return cls(
            teams=m["teams"],
            leagues={int(k): v for k, v in m["leagues"].items()},
            scaler_mean=np.array(m["scaler_mean"]),
            scaler_scale=np.array(m["scaler_scale"]),
            temperature=float(m.get("temperature", 1.0)),
        )

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "teams": self.teams,
                    "leagues": {str(k): v for k, v in self.leagues.items()},
                    "scaler_mean": self.scaler_mean.tolist(),
                    "scaler_scale": self.scaler_scale.tolist(),
                    "temperature": self.temperature,
                },
                f,
                indent=4,
            )


def load_model(model_path: Path, mappings: Mappings) -> SuperSoccerNet:
    """Load weights strictly — a shape mismatch means the checkpoint and
    mappings are out of sync, and silently continuing with random layers
    (the old strict=False behaviour) produces garbage predictions."""
    net = SuperSoccerNet(len(mappings.teams), len(mappings.leagues))
    net.load_state_dict(torch.load(model_path, weights_only=True))
    net.eval()
    return net


@torch.no_grad()
def predict_proba(
    net: SuperSoccerNet,
    mappings: Mappings,
    h_idx: int,
    a_idx: int,
    l_idx: int,
    seq: list[list[float]],
    cont: np.ndarray,
) -> np.ndarray:
    """Temperature-scaled outcome probabilities for a single fixture."""
    assert len(seq) == SEQ_LEN
    net.eval()
    logits = net(
        torch.tensor([h_idx]),
        torch.tensor([a_idx]),
        torch.tensor([l_idx]),
        torch.tensor([seq], dtype=torch.float32),
        torch.tensor([cont], dtype=torch.float32),
    )
    return torch.softmax(logits / mappings.temperature, dim=1).numpy()[0]
