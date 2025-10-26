from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from freeciv_rl.freeciv_movement import FreecivMovement

from freeciv_alpha_zero.config import MapConfig
from freeciv_alpha_zero.providers import BaseProvider, GroundTruth

Player = int  # 1 or -1
Coord = Tuple[int, int]


@dataclass
class FreecivBoardState:
    cfg: MapConfig
    provider: BaseProvider
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    gt: GroundTruth | None = None
    units: Dict[Player, Coord] = field(default_factory=dict)
    visited: Dict[Player, np.ndarray] = field(default_factory=dict)
    revealed: Dict[Player, np.ndarray] = field(default_factory=dict)
    turn: int = 0
    winner: Optional[Player] = None
    terminal_reason: Optional[str] = None

    ACTION_SIZE = 6
    PASS_ACTION = ACTION_SIZE

    def __post_init__(self) -> None:
        self.movement = FreecivMovement(self.cfg.map_w, self.cfg.map_h)
        if self.gt is None:
            self.reset()

    def reset(self) -> None:
        self.gt = self.provider.resample()
        self.turn = 0
        self.winner = None
        self.terminal_reason = None
        self.units = {}
        self.visited = {}
        self.revealed = {}

        spawn_a = self._find_spawn((0, 0))
        spawn_b = self._find_spawn((self.cfg.map_w - 1, self.cfg.map_h - 1))
        self.units[1] = spawn_a
        self.units[-1] = spawn_b

        for p in (1, -1):
            self.visited[p] = np.zeros((self.cfg.map_h, self.cfg.map_w), dtype=bool)
            self.revealed[p] = np.zeros((self.cfg.map_h, self.cfg.map_w), dtype=bool)
            x, y = self.units[p]
            self.visited[p][y, x] = True
            self._reveal(p)

    # ---------- helpers ----------
    def _find_spawn(self, start_hint: Coord) -> Coord:
        assert self.gt is not None
        sx, sy = start_hint
        sx = int(np.clip(sx, 0, self.cfg.map_w - 1))
        sy = int(np.clip(sy, 0, self.cfg.map_h - 1))
        if self.gt.au_map[sy, sx] == 'A':
            return sx, sy
        # fallback: breadth-first search for closest 'A'
        frontier = [(sx, sy)]
        seen = { (sx, sy) }
        while frontier:
            nx, ny = frontier.pop(0)
            if 0 <= nx < self.cfg.map_w and 0 <= ny < self.cfg.map_h:
                if self.gt.au_map[ny, nx] == 'A':
                    return nx, ny
                for adj in self.movement.get_native_neighbors(nx, ny):
                    if adj in seen or adj[0] is None:
                        continue
                    seen.add(adj)
                    frontier.append(adj)
        raise RuntimeError("Map has no available tiles for spawn")

    def _player_idx(self, player: Player) -> int:
        return 0 if player == 1 else 1

    def _reveal(self, player: Player) -> None:
        x, y = self.units[player]
        radius = max(1, self.cfg.fog_radius)
        frontier = [(x, y, 0)]
        seen = set()
        while frontier:
            nx, ny, dist = frontier.pop(0)
            if (nx, ny) in seen or nx is None:
                continue
            seen.add((nx, ny))
            if not (0 <= nx < self.cfg.map_w and 0 <= ny < self.cfg.map_h):
                continue
            self.revealed[player][ny, nx] = True
            if dist >= radius:
                continue
            for adj in self.movement.get_native_neighbors(nx, ny):
                if adj[0] is None:
                    continue
                frontier.append((adj[0], adj[1], dist + 1))

    def duplicate(self) -> "FreecivBoardState":
        new = FreecivBoardState.__new__(FreecivBoardState)
        new.cfg = self.cfg
        new.provider = self.provider
        new.rng = self.rng
        new.movement = self.movement
        new.gt = self.gt.copy() if self.gt else None
        new.units = {p: (xy[0], xy[1]) for p, xy in self.units.items()}
        new.visited = {p: grid.copy() for p, grid in self.visited.items()}
        new.revealed = {p: grid.copy() for p, grid in self.revealed.items()}
        new.turn = self.turn
        new.winner = self.winner
        new.terminal_reason = self.terminal_reason
        return new

    # ---------- game mechanics ----------
    def valid_moves(self, player: Player) -> np.ndarray:
        moves = np.zeros(self.ACTION_SIZE + 1, dtype=np.int8)
        if self.winner is not None:
            moves[-1] = 1
            return moves
        x, y = self.units[player]
        neighbors = self.movement.get_native_neighbors(x, y)
        for idx, (nx, ny) in enumerate(neighbors):
            if nx is None:
                continue
            if self.gt.au_map[ny, nx] == 'A':
                moves[idx] = 1
        moves[-1] = 1  # allow pass
        return moves

    def step(self, player: Player, action: int) -> None:
        if self.winner is not None:
            return
        if action == self.PASS_ACTION:
            self.turn += 1
            if self.turn >= self.cfg.max_turns:
                self._resolve_terminal(reason="max_turns")
            return

        valid = self.valid_moves(player)
        if action < 0 or action >= self.ACTION_SIZE or valid[action] == 0:
            # invalid -> treat as pass with penalty by forcing stay
            self.turn += 1
            if self.turn >= self.cfg.max_turns:
                self._resolve_terminal(reason="max_turns")
            return

        x, y = self.units[player]
        nx, ny = self.movement.get_native_neighbors(x, y)[action]
        if nx is None:
            self.turn += 1
            return
        if self.gt.au_map[ny, nx] != 'A':
            self.turn += 1
            return

        self.units[player] = (nx, ny)
        self.visited[player][ny, nx] = True
        self._reveal(player)
        opponent = -player
        if self.units[player] == self.units[opponent]:
            self._resolve_terminal(winner=player, reason="capture")
            return
        if self.gt.enemy_map[ny, nx]:
            self._resolve_terminal(winner=opponent, reason="enemy_trap")
            return

        self.turn += 1
        if self.turn >= self.cfg.max_turns:
            self._resolve_terminal(reason="max_turns")

    def _resolve_terminal(self, winner: Optional[Player] = None, reason: str = "unknown") -> None:
        self.winner = winner
        self.terminal_reason = reason

    def final_score(self) -> float:
        if self.winner == 1:
            return 1.0
        if self.winner == -1:
            return -1.0
        # territory comparison
        diff = float(self.visited[1].sum() - self.visited[-1].sum())
        total = max(1.0, float(self.cfg.map_w * self.cfg.map_h))
        normalized = diff / total
        if abs(normalized) < 1e-6:
            return self.cfg.draw_value
        return float(np.clip(normalized, -1.0, 1.0))

    # ---------- encodings ----------
    def encode(self, perspective: Player) -> np.ndarray:
        assert self.gt is not None
        me = perspective
        opp = -perspective
        channels = []
        channels.append((self.gt.au_map == 'A').astype(np.float32))
        channels.append((self.gt.au_map == 'U').astype(np.float32))
        unit_me = np.zeros_like(channels[0])
        unit_opp = np.zeros_like(channels[0])
        mx, my = self.units[me]
        unit_me[my, mx] = 1.0
        ox, oy = self.units[opp]
        unit_opp[oy, ox] = 1.0
        channels.append(unit_me)
        channels.append(unit_opp)
        channels.append(self.visited[me].astype(np.float32))
        channels.append(self.visited[opp].astype(np.float32))
        channels.append(self.gt.enemy_map.astype(np.float32))
        stacked = np.stack(channels, axis=0)
        return stacked

    def string(self) -> str:
        parts = [f"turn={self.turn}"]
        for p in (1, -1):
            x, y = self.units[p]
            parts.append(f"p{p}:{x},{y}")
        if self.winner:
            parts.append(f"winner={self.winner}")
        return '|'.join(parts)
