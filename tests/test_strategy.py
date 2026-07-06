import pytest

from quantbet.strategy import (
    SAFE,
    VALUE,
    StrategyProfile,
    dedupe_by_match,
    expected_value,
    kelly_fraction,
)


class TestExpectedValue:
    def test_fair_odds_have_zero_ev(self):
        assert expected_value(0.5, 2.0) == pytest.approx(0.0)

    def test_positive_edge(self):
        assert expected_value(0.6, 2.0) == pytest.approx(0.2)

    def test_negative_edge(self):
        assert expected_value(0.4, 2.0) == pytest.approx(-0.2)


class TestKelly:
    def test_known_value(self):
        # p=0.6 at evens: f* = (1*0.6 - 0.4) / 1 = 0.2
        assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)

    def test_no_edge_means_no_bet(self):
        assert kelly_fraction(0.5, 2.0) == pytest.approx(0.0)

    def test_negative_edge_clamped_to_zero(self):
        assert kelly_fraction(0.3, 2.0) == 0.0

    def test_degenerate_odds(self):
        assert kelly_fraction(0.9, 1.0) == 0.0
        assert kelly_fraction(0.9, 0.5) == 0.0


class TestProfiles:
    def test_safe_accepts_strong_favourite_with_edge(self):
        assert SAFE.accepts(prob=0.60, decimal_odds=2.0)

    def test_safe_rejects_low_probability(self):
        assert not SAFE.accepts(prob=0.40, decimal_odds=3.0)

    def test_safe_rejects_odds_outside_band(self):
        assert not SAFE.accepts(prob=0.90, decimal_odds=1.10)
        assert not SAFE.accepts(prob=0.55, decimal_odds=4.0)

    def test_safe_rejects_insufficient_ev(self):
        # p=0.51 @ 2.0 -> EV = 0.02 < 0.05
        assert not SAFE.accepts(prob=0.51, decimal_odds=2.0)

    def test_value_requires_bigger_edge(self):
        assert not VALUE.accepts(prob=0.30, decimal_odds=3.5)  # EV 0.05
        assert VALUE.accepts(prob=0.35, decimal_odds=3.5)  # EV 0.225

    def test_stake_is_capped(self):
        profile = StrategyProfile(
            name="x", min_prob=0, min_ev=0, min_odds=1, max_odds=100,
            kelly_multiplier=1.0, max_stake_pct=0.05, sort_key="ev",
        )
        # Full Kelly here would be 0.2; the cap must win.
        assert profile.stake_fraction(0.6, 2.0) == pytest.approx(0.05)

    def test_fractional_kelly_applied(self):
        assert SAFE.stake_fraction(0.55, 2.0) == pytest.approx(0.1 * 0.5)


class TestDedupe:
    def test_keeps_highest_ev_line_per_match(self):
        bets = [
            {"match": "A vs B", "ev": 0.05},
            {"match": "A vs B", "ev": 0.12},
            {"match": "C vs D", "ev": 0.03},
        ]
        result = {b["match"]: b["ev"] for b in dedupe_by_match(bets)}
        assert result == {"A vs B": 0.12, "C vs D": 0.03}
