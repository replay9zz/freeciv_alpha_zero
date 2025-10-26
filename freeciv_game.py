from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Ensure alpha-zero-general is importable when consumers run this module directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__('sys').path:
    __import__('sys').path.append(str(ROOT))

from alpha_zero_general.Game import Game  # type: ignore

from freeciv_alpha_zero.config import MapConfig
from freeciv_alpha_zero.providers import BaseProvider, RandomMapProvider
from freeciv_alpha_zero.state import FreecivBoardState, Player


@dataclass
class CanonicalBoard:
    state: FreecivBoardState
    perspective: Player


class FreecivGame(Game):
    def __init__(self, cfg: MapConfig | None = None, provider: BaseProvider | None = None):
        self.cfg = cfg or MapConfig()
        self.provider = provider or RandomMapProvider(self.cfg.map_w, self.cfg.map_h)

    # ----- Game interface -----
    def getInitBoard(self) -> FreecivBoardState:
        return FreecivBoardState(self.cfg, self.provider)

    def getBoardSize(self) -> Tuple[int, int, int]:
        state = self.getInitBoard()
        board = state.encode(1)
        return board.shape  # (channels, H, W)

    def getActionSize(self) -> int:
        return FreecivBoardState.ACTION_SIZE + 1

    def getNextState(self, board: FreecivBoardState, player: Player, action: int):
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
        score = state.final_score()
        return score if actual_player == 1 else -score

    def getCanonicalForm(self, board, player):
        if isinstance(board, CanonicalBoard) and board.perspective == player:
            return board
        return CanonicalBoard(board, player)

    def getSymmetries(self, board, pi):
        # Hex grid has rotational symmetry, but we keep it simple for now.
        return [(board, pi)]

    def stringRepresentation(self, board) -> str:
        if isinstance(board, CanonicalBoard):
            return f"{board.perspective}:{board.state.string()}"
        return f"raw:{board.string()}"

    # ----- helpers -----
    def _unwrap(self, board, player) -> Tuple[FreecivBoardState, Player]:
        if isinstance(board, CanonicalBoard):
            return board.state, board.perspective
        return board, player
