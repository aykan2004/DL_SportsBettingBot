import numpy as np
import pytest

from quantbet.baselines import PoissonBaseline, outcome_probs_from_lambdas


class TestOutcomeProbs:
    def test_probabilities_sum_to_one(self):
        probs = outcome_probs_from_lambdas(1.5, 1.1)
        assert probs.sum() == pytest.approx(1.0)
        assert (probs > 0).all()

    def test_symmetric_lambdas_give_symmetric_probs(self):
        probs = outcome_probs_from_lambdas(1.3, 1.3)
        assert probs[0] == pytest.approx(probs[2])

    def test_stronger_home_attack_favours_home(self):
        strong = outcome_probs_from_lambdas(2.5, 0.8)
        weak = outcome_probs_from_lambdas(0.8, 2.5)
        assert strong[0] > 0.6
        assert weak[2] > 0.6

    def test_low_scoring_games_have_more_draws(self):
        low = outcome_probs_from_lambdas(0.8, 0.8)
        high = outcome_probs_from_lambdas(3.0, 3.0)
        assert low[1] > high[1]


class TestPoissonBaseline:
    def test_unknown_teams_get_league_average(self):
        model = PoissonBaseline(["A", "B"])
        probs = model.predict("Unknown1", "Unknown2")
        assert probs.shape == (3,)
        assert probs.sum() == pytest.approx(1.0)
        # With average ratings, home advantage should tilt toward home.
        assert probs[0] > probs[2]

    def test_fit_learns_dominant_team(self):
        import pandas as pd

        rng = np.random.default_rng(0)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
        # Team A batters team B in every meeting.
        df = pd.DataFrame(
            {
                "date": dates,
                "home_team": ["A", "B"] * (n // 2),
                "away_team": ["B", "A"] * (n // 2),
                "home_goals": rng.poisson([3.0, 0.3] * (n // 2)),
                "away_goals": rng.poisson([0.3, 3.0] * (n // 2)),
            }
        )
        model = PoissonBaseline(["A", "B"])
        model.fit(df)
        probs = model.predict("A", "B")
        assert probs[0] > 0.7  # A at home vs B must be a heavy favourite
