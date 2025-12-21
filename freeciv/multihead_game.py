from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    from freeciv_alpha_zero.Game import Game  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    try:
        from ..Game import Game
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Unable to locate Game definition for Freeciv package") from exc

from .multihead_state import MultiheadState
from .config import MapConfig
from .providers import BaseProvider, RandomMapProvider

Player = int


@dataclass
class MultiheadBoard:
    state: MultiheadState
    perspective: Player

    def encode(self, perspective: Player = 1):
        return self.state.encode(self.perspective)


class MultiheadGame(Game):
    """
    Multi-unit move/attack + research environment for multi-head training.
    """

    def __init__(
        self,
        cfg: MapConfig | None = None,
        provider: BaseProvider | None = None,
        max_units: int = 6,
        max_cities: int = 3,
    ):
        self.cfg = cfg or MapConfig()
        self.provider = provider or RandomMapProvider(self.cfg.map_w, self.cfg.map_h)
        self.max_units = max_units
        self.max_cities = max_cities
        # Precompute action size to avoid repeated state construction in MCTS.
        tmp_state = MultiheadState(
            self.cfg,
            self.provider,
            max_units=self.max_units,
            max_cities=self.max_cities,
        )
        self._action_size = tmp_state.ACTION_SIZE
        # Expose head sizes for multi-head policy networks.
        self.policy_head_sizes = (tmp_state.MOVE_SIZE, tmp_state.ATTACK_SIZE, tmp_state.ECON_SIZE)

    def getInitBoard(self) -> MultiheadState:
        return MultiheadState(
            self.cfg,
            self.provider,
            max_units=self.max_units,
            max_cities=self.max_cities,
        )

    def getBoardSize(self) -> Tuple[int, int, int]:
        state = self.getInitBoard()
        board = state.encode(1)
        return board.shape

    def getActionSize(self) -> int:
        return self._action_size

    def getNextState(self, board: MultiheadState, player: Player, action: int):
        state, actual_player = self._unwrap(board, player)
        state = state.duplicate()
        state.step(actual_player, action)
        next_player = -actual_player
        return state, next_player

    def getValidMoves(self, board, player):
        state, actual_player = self._unwrap(board, player)
        return state.valid_moves(actual_player)

    def getGameEnded(self, board, player):
        state, actual_player = self._unwrap(board, player)
        # If an explicit winner is set, honor it.
        if state.winner == 1:
            score = 1.0
        elif state.winner == -1:
            score = -1.0
        elif state.winner == 0 or state.turn >= self.cfg.max_turns:
            # Use heuristic tiebreakers on max-turn or mutual destruction.
            hs = state.heuristic_score(actual_player)
            if hs > 0:
                score = 1.0
            elif hs < 0:
                score = -1.0
            else:
                score = self.cfg.draw_value
        else:
            # Game still ongoing
            return 0
        return score if actual_player == 1 else -score

    def getCanonicalForm(self, board, player):
        if isinstance(board, MultiheadBoard) and board.perspective == player:
            return board
        return MultiheadBoard(board, player)

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board) -> str:
        if isinstance(board, MultiheadBoard):
            return f"{board.perspective}:{board.state.string()}"
        return f"p1:{board.string()}"

    def _unwrap(self, board, player) -> Tuple[MultiheadState, Player]:
        if isinstance(board, MultiheadBoard):
            return board.state, board.perspective
        return board, player
