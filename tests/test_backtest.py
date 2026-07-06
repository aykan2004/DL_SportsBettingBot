import numpy as np
import pytest

from quantbet.backtest import implied_probabilities, log_loss_of, simulate_bankroll
from quantbet.strategy import StrategyProfile


class TestImpliedProbabilities:
    def test_overround_is_removed(self):
        probs = implied_probabilities([2.0, 3.0, 6.0])  # book sums to 1.0 exactly
        assert probs.sum() == pytest.approx(1.0)
        assert probs[0] == pytest.approx(0.5)

    def test_typical_book_with_margin(self):
        probs = implied_probabilities([1.9, 3.4, 4.2])
        assert probs.sum() == pytest.approx(1.0)
        assert probs[0] > 1 / 1.9 / 1.2  # sane scale after normalization


class TestLogLoss:
    def test_perfect_prediction(self):
        probs = np.array([[1.0, 0.0, 0.0]])
        assert log_loss_of(probs, np.array([0])) == pytest.approx(0.0, abs=1e-9)

    def test_uniform_prediction(self):
        probs = np.full((10, 3), 1 / 3)
        assert log_loss_of(probs, np.zeros(10, dtype=int)) == pytest.approx(np.log(3))


ALWAYS_BET = StrategyProfile(
    name="test", min_prob=0.0, min_ev=0.0, min_odds=1.0, max_odds=100.0,
    kelly_multiplier=1.0, max_stake_pct=0.05, sort_key="ev",
)


class TestSimulateBankroll:
    def test_winning_bet_grows_bankroll(self):
        probs = np.array([[0.9, 0.05, 0.05]])
        rows = [{"odds": [2.0, 3.0, 3.0], "label": 0}]
        sim = simulate_bankroll(probs, rows, ALWAYS_BET)
        assert sim["bets"] == 1
        assert sim["final_bankroll"] > 1000.0
        assert sim["hit_rate"] == 1.0

    def test_no_edge_no_bets(self):
        probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
        rows = [{"odds": [1.5, 1.5, 1.5], "label": 0}]  # all -EV
        sim = simulate_bankroll(probs, rows, ALWAYS_BET)
        assert sim["bets"] == 0
        assert sim["final_bankroll"] == 1000.0

    def test_drawdown_tracked_on_losses(self):
        probs = np.array([[0.9, 0.05, 0.05]])
        rows = [{"odds": [2.0, 3.0, 3.0], "label": 2}]  # bet loses
        sim = simulate_bankroll(probs, rows, ALWAYS_BET)
        assert sim["final_bankroll"] < 1000.0
        assert sim["max_drawdown"] > 0.0
