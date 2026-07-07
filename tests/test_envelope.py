"""Envelope-rejection negative test (design doc §6.1, §12 M0).

The "asserted in code, not just prose" scope boundary needs a test that the assertion
actually *fires*: a mis-declared adapter must be rejected both by ``assert_v1_envelope``
and by constructing an ``MCTS`` over it. (M1.5's zero-``core/``-diff check proves the
positive abstraction claim — a different property.)
"""

from __future__ import annotations

import pytest

from core import MCTS, EnvelopeError, assert_v1_envelope
from games.connect4 import Connect4
from games.othello import Othello
from games.tictactoe import TicTacToe
from tests.fixtures.bad_adapters import ImperfectInfoGame, StochasticGame, ThreePlayerGame

BAD = [ThreePlayerGame(), StochasticGame(), ImperfectInfoGame()]
BAD_IDS = ["3-player", "stochastic", "imperfect-info"]


@pytest.mark.parametrize("bad", BAD, ids=BAD_IDS)
def test_assert_envelope_rejects_out_of_envelope_adapters(bad):
    with pytest.raises(EnvelopeError):
        assert_v1_envelope(bad)


@pytest.mark.parametrize("bad", BAD, ids=BAD_IDS)
def test_mcts_construction_rejects_out_of_envelope_adapters(bad):
    # The envelope check must fire before any search happens.
    with pytest.raises(EnvelopeError):
        MCTS(bad)


@pytest.mark.parametrize("game", [TicTacToe(), Connect4(), Othello()], ids=["ttt", "c4", "othello"])
def test_envelope_accepts_valid_games(game):
    assert_v1_envelope(game)  # must not raise
    MCTS(game)  # must not raise
