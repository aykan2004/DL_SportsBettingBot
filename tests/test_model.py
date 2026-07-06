import numpy as np
import pytest
import torch

from quantbet.features import SEQ_LEN
from quantbet.model import Mappings, SuperSoccerNet, predict_proba


@pytest.fixture
def tiny_net():
    torch.manual_seed(0)
    return SuperSoccerNet(num_teams=10, num_leagues=3)


class TestForward:
    def test_output_shape(self, tiny_net):
        batch = 4
        out = tiny_net(
            torch.zeros(batch, dtype=torch.long),
            torch.ones(batch, dtype=torch.long),
            torch.zeros(batch, dtype=torch.long),
            torch.zeros(batch, SEQ_LEN, 4),
            torch.zeros(batch, 6),
        )
        assert out.shape == (batch, 3)


class TestPredictProba:
    def test_probabilities_sum_to_one(self, tiny_net):
        mappings = Mappings(
            teams={f"t{i}": i for i in range(10)},
            leagues={39: 0},
            scaler_mean=np.zeros(4),
            scaler_scale=np.ones(4),
            temperature=1.0,
        )
        probs = predict_proba(
            tiny_net, mappings, 0, 1, 0,
            seq=[[0, 0, 0, 0]] * SEQ_LEN,
            cont=np.zeros(6, dtype=np.float32),
        )
        assert probs.shape == (3,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_higher_temperature_flattens_distribution(self, tiny_net):
        base = Mappings({"a": 0}, {39: 0}, np.zeros(4), np.ones(4), temperature=1.0)
        hot = Mappings({"a": 0}, {39: 0}, np.zeros(4), np.ones(4), temperature=5.0)
        args = (0, 0, 0, [[1, 0, 2, 1]] * SEQ_LEN, np.ones(6, dtype=np.float32))
        p_base = predict_proba(tiny_net, base, *args)
        p_hot = predict_proba(tiny_net, hot, *args)
        assert p_hot.max() < p_base.max() or p_base.max() == pytest.approx(1 / 3)


class TestMappingsRoundtrip:
    def test_save_load(self, tmp_path):
        m = Mappings(
            teams={"Arsenal": 0},
            leagues={39: 0, 2: 1},
            scaler_mean=np.array([1.0, 2.0, 3.0, 4.0]),
            scaler_scale=np.array([1.0, 1.0, 2.0, 2.0]),
            temperature=1.37,
        )
        path = tmp_path / "mappings.json"
        m.save(path)
        loaded = Mappings.load(path)
        assert loaded.teams == m.teams
        assert loaded.leagues == m.leagues  # int keys survive the roundtrip
        assert loaded.temperature == pytest.approx(1.37)
        np.testing.assert_allclose(loaded.scaler_mean, m.scaler_mean)
