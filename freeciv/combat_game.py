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

from .combat_state import CombatState
from .config import MapConfig
from .providers import BaseProvider, RandomMapProvider

Player = int


@dataclass
class CombatBoard:
    state: CombatState
    perspective: Player

    def encode(self, perspective: Player = 1):
        # Ignore requested perspective; this board already encodes for its own perspective.
        return self.state.encode(self.perspective)


class CombatGame(Game):
    """
    Simple combat-focused game wrapper for self-play.
    Multiple units per side, movement + attack only.
    """

    def __init__(self, cfg: MapConfig | None = None, provider: BaseProvider | None = None):
        self.cfg = cfg or MapConfig()
        self.provider = provider or RandomMapProvider(self.cfg.map_w, self.cfg.map_h)

    def getInitBoard(self) -> CombatState:
        return CombatState(self.cfg, self.provider)

    def getBoardSize(self) -> Tuple[int, int, int]:
        state = self.getInitBoard()
        board = state.encode(1)
        return board.shape

    def getActionSize(self) -> int:
        return CombatState.ACTION_SIZE + 1

    def getNextState(self, board: CombatState, player: Player, action: int):
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
        if state.winner is None and state.turn < self.cfg.max_turns:
            return 0
        # winner=1 => score 1, winner=-1 => -1, draw -> draw_value
        if state.winner == 1:
            score = 1.0
        elif state.winner == -1:
            score = -1.0
        else:
            score = self.cfg.draw_value
        return score if actual_player == 1 else -score

    def getCanonicalForm(self, board, player):
        if isinstance(board, CombatBoard) and board.perspective == player:
            return board
        return CombatBoard(board, player)

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board) -> str:
        if isinstance(board, CombatBoard):
            return f"{board.perspective}:{board.state.string()}"
        # If passed a raw CombatState, include a default perspective marker.
        return f"p1:{board.string()}"

    def _unwrap(self, board, player) -> Tuple[CombatState, Player]:
        if isinstance(board, CombatBoard):
            return board.state, board.perspective
        return board, player
