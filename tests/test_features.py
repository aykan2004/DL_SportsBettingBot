import numpy as np
import pandas as pd
import pytest

from quantbet.features import (
    SEQ_LEN,
    Outcome,
    build_match_sequences,
    combine_sequences,
    make_continuous,
    pad_history,
    result_from_goals,
)


class TestLabels:
    def test_canonical_encoding(self):
        assert result_from_goals(2, 1) == Outcome.HOME == 0
        assert result_from_goals(1, 1) == Outcome.DRAW == 1
        assert result_from_goals(0, 3) == Outcome.AWAY == 2


class TestPadHistory:
    def test_padding_lands_on_oldest_slots(self):
        padded = pad_history([[2, 1], [3, 0]])
        assert padded == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [2, 1], [3, 0]]

    def test_truncates_to_most_recent(self):
        hist = [[i, 0] for i in range(8)]
        assert pad_history(hist) == [[3, 0], [4, 0], [5, 0], [6, 0], [7, 0]]


class TestCombineSequences:
    def test_shape_and_interleaving(self):
        seq = combine_sequences([[1, 0]], [[2, 2]])
        assert len(seq) == SEQ_LEN and all(len(step) == 4 for step in seq)
        # Newest step holds [h_gf, h_ga, a_gf, a_ga].
        assert seq[-1] == [1, 0, 2, 2]


class TestBuildMatchSequences:
    def test_no_lookahead(self):
        """A match's sequence must only contain matches played before it."""
        df = pd.DataFrame(
            {
                "home_team": ["A", "A"],
                "away_team": ["B", "B"],
                "home_goals": [2, 1],
                "away_goals": [0, 1],
            }
        )
        seqs = build_match_sequences(df)
        assert seqs[0] == [[0.0, 0.0, 0.0, 0.0]] * SEQ_LEN  # nothing before match 1
        # Match 2 sees match 1: A scored 2 conceded 0; B scored 0 conceded 2.
        assert seqs[1][-1] == [2, 0, 0, 2]

    def test_home_away_perspective(self):
        """GF/GA must follow the team, not the venue."""
        df = pd.DataFrame(
            {
                "home_team": ["A", "B"],
                "away_team": ["B", "A"],
                "home_goals": [0, 0],
                "away_goals": [3, 0],
            }
        )
        seqs = build_match_sequences(df)
        # In match 2, B (now home) scored 3 / conceded 0 in its previous away
        # game, while A scored 0 / conceded 3.
        assert seqs[1][-1] == [3, 0, 0, 3]


class TestMakeContinuous:
    def test_scaling_and_gd_clipping(self):
        mean = np.array([1.0, 1.0, 0.0, 0.0])
        scale = np.array([2.0, 2.0, 1.0, 1.0])
        cont = make_continuous(3.0, 1.0, 9.0, -9.0, mean, scale, 0.7, 0.3)
        assert cont == pytest.approx([1.0, 0.0, 4.0, -4.0, 0.7, 0.3])
        assert cont.dtype == np.float32

    def test_gd_features_are_not_zeroed(self):
        """Regression test for the live-inference bug: goal difference was
        silently replaced with zeros, so the model ran on features it was
        never trained with."""
        mean = np.zeros(4)
        scale = np.ones(4)
        cont = make_continuous(0.0, 0.0, 2.0, -1.0, mean, scale)
        assert cont[2] == pytest.approx(2.0)
        assert cont[3] == pytest.approx(-1.0)
