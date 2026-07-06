import pytest

from quantbet.settlement import grade_bet


class TestGradeBet:
    def test_straight_win_pays_odds_minus_one(self):
        status, profit = grade_bet("Home", "Home", stake=10.0, odds=2.5)
        assert status == "won"
        assert profit == pytest.approx(15.0)

    def test_straight_loss_forfeits_stake(self):
        status, profit = grade_bet("Home", "Away", stake=10.0, odds=2.5)
        assert status == "lost"
        assert profit == -10.0

    def test_draw_pick_wins_on_draw(self):
        status, profit = grade_bet("Draw", "Draw", stake=5.0, odds=3.4)
        assert status == "won"
        assert profit == pytest.approx(12.0)

    def test_void_match_refunds(self):
        status, profit = grade_bet("Home", "Void", stake=10.0, odds=2.0)
        assert status == "refunded"
        assert profit == 0.0

    def test_dnb_draw_refunds(self):
        status, profit = grade_bet("Home (DNB)", "Draw", stake=10.0, odds=1.8)
        assert status == "refunded"
        assert profit == 0.0

    def test_dnb_win(self):
        status, profit = grade_bet("Away (DNB)", "Away", stake=10.0, odds=1.9)
        assert status == "won"
        assert profit == pytest.approx(9.0)

    def test_dnb_loss(self):
        status, profit = grade_bet("Away (DNB)", "Home", stake=10.0, odds=1.9)
        assert status == "lost"
        assert profit == -10.0
