"""Integration test for the inference glue path — fixture in, candidates out.

This is exactly where v1's train/serve skew bug lived (goal-difference
features silently zeroed), so the wiring between the API layer, the feature
builder and the model gets its own offline test with the network mocked out.
"""

import numpy as np
import pytest
import torch

import quantbet.slip as slip
from quantbet.model import Mappings, SuperSoccerNet

FIXTURE = {
    "teams": {
        "home": {"id": 42, "name": "Arsenal"},
        "away": {"id": 49, "name": "Chelsea"},
    },
    "fixture": {"date": "2026-07-05T15:00:00+00:00"},
}


@pytest.fixture
def mappings():
    return Mappings(
        teams={"Arsenal": 0, "Chelsea": 1},
        leagues={39: 0},
        scaler_mean=np.zeros(4),
        scaler_scale=np.ones(4),
        temperature=1.0,
    )


@pytest.fixture
def net():
    torch.manual_seed(0)
    return SuperSoccerNet(num_teams=2, num_leagues=1)


def _mock_stats(_team_id):
    return [7.5, 0.6], [[1, 0], [2, 1], [0, 0], [3, 1], [1, 1]]


class TestPriceFixture:
    def test_produces_priced_candidates_with_snapshot(self, monkeypatch, mappings, net):
        monkeypatch.setattr(slip, "get_team_recent_stats", _mock_stats)
        monkeypatch.setattr(slip, "get_match_odds", lambda *a: [2.1, 3.4, 3.6])

        candidates = slip._price_fixture(net, mappings, FIXTURE, league_id=39)

        assert len(candidates) == 3
        assert {c["bet"] for c in candidates} == {"Home", "Draw", "Away"}
        assert sum(c["prob"] for c in candidates) == pytest.approx(1.0, abs=1e-5)
        for c in candidates:
            assert c["ev"] == pytest.approx(c["prob"] * c["odds"] - 1)
            # The retrainer depends on this snapshot existing.
            assert len(c["features"]["cont"]) == 6
            assert len(c["features"]["seq"]) == 5
        # Regression guard: scaled GD features must flow through, not zeros.
        cont = candidates[0]["features"]["cont"]
        assert cont[2] != 0.0 and cont[3] != 0.0

    def test_skips_fixture_when_form_unavailable(self, monkeypatch, mappings, net):
        monkeypatch.setattr(slip, "get_team_recent_stats", lambda _id: None)
        assert slip._price_fixture(net, mappings, FIXTURE, league_id=39) == []

    def test_skips_unmapped_teams(self, mappings, net):
        unknown = {
            "teams": {
                "home": {"id": 1, "name": "Nowhere FC"},
                "away": {"id": 2, "name": "Chelsea"},
            },
            "fixture": {"date": "2026-07-05T15:00:00+00:00"},
        }
        assert slip._price_fixture(net, mappings, unknown, league_id=39) == []

    def test_suspended_market_lines_excluded(self, monkeypatch, mappings, net):
        monkeypatch.setattr(slip, "get_team_recent_stats", _mock_stats)
        monkeypatch.setattr(slip, "get_match_odds", lambda *a: [1.01, 3.4, 3.6])
        candidates = slip._price_fixture(net, mappings, FIXTURE, league_id=39)
        assert {c["bet"] for c in candidates} == {"Draw", "Away"}
