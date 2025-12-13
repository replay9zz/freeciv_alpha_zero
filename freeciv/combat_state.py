from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from freeciv_rl.freeciv_movement import FreecivMovement

from .config import MapConfig
from .providers import BaseProvider, GroundTruth

Player = int  # 1 or -1
Coord = Tuple[int, int]


@dataclass
class CombatUnit:
    x: int
    y: int
    hp: int
    atk: int
    df: int
    alive: bool = True


@dataclass
class CombatState:
    cfg: MapConfig
    provider: BaseProvider
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    gt: GroundTruth | None = None
    movement: FreecivMovement | None = None
    units: Dict[Player, List[CombatUnit]] = field(default_factory=lambda: {1: [], -1: []})
    turn: int = 0
    winner: Optional[Player] = None
    terminal_reason: Optional[str] = None
    ACTION_SIZE: int = 6 * 3  # 3 units * 6 directions + pass (added later)
    PASS_ACTION: int = ACTION_SIZE

    def __post_init__(self) -> None:
        self.movement = FreecivMovement(self.cfg.map_w, self.cfg.map_h)
        if self.gt is None:
            self.reset()

    def reset(self) -> None:
        self.gt = self.provider.resample()
        self.turn = 0
        self.winner = None
        self.terminal_reason = None
        self.units = {1: [], -1: []}
        self._spawn_units()

    def _spawn_units(self) -> None:
        assert self.gt is not None
        # Spawn three units per side on opposite corners/edges.
        starts = [(0, 0), (0, self.cfg.map_h - 2), (1, self.cfg.map_h - 1)]
        opp_starts = [(self.cfg.map_w - 1, self.cfg.map_h - 1), (self.cfg.map_w - 1, 1), (self.cfg.map_w - 2, 0)]
        stats = [
            {"hp": 10, "atk": 2, "df": 1},  # warrior-like
            {"hp": 8, "atk": 3, "df": 1},   # archer-like (strong attack)
            {"hp": 12, "atk": 1, "df": 2},  # phalanx-like (strong defense)
        ]
        for idx, (sx, sy) in enumerate(starts):
            nx, ny = self._find_spawn_near(sx, sy)
            st = stats[min(idx, len(stats) - 1)]
            self.units[1].append(CombatUnit(nx, ny, st["hp"], st["atk"], st["df"]))
        for idx, (sx, sy) in enumerate(opp_starts):
            nx, ny = self._find_spawn_near(sx, sy)
            st = stats[min(idx, len(stats) - 1)]
            self.units[-1].append(CombatUnit(nx, ny, st["hp"], st["atk"], st["df"]))

    def _find_spawn_near(self, sx: int, sy: int) -> Coord:
        assert self.gt is not None
        sx = int(np.clip(sx, 0, self.cfg.map_w - 1))
        sy = int(np.clip(sy, 0, self.cfg.map_h - 1))
        if self.gt.au_map[sy, sx] == 'A':
            return sx, sy
        frontier = [(sx, sy)]
        seen = { (sx, sy) }
        while frontier:
            x, y = frontier.pop(0)
            if 0 <= x < self.cfg.map_w and 0 <= y < self.cfg.map_h and self.gt.au_map[y, x] == 'A':
                return x, y
            for nx, ny in self.movement.get_native_neighbors(x, y):
                if nx is None:
                    continue
                if (nx, ny) in seen:
                    continue
                seen.add((nx, ny))
                frontier.append((nx, ny))
        return sx, sy

    def duplicate(self) -> "CombatState":
        new = CombatState.__new__(CombatState)
        new.cfg = self.cfg
        new.provider = self.provider
        new.rng = self.rng
        new.movement = self.movement
        new.gt = self.gt.copy() if self.gt else None
        new.units = {
            p: [CombatUnit(u.x, u.y, u.hp, u.atk, u.df, u.alive) for u in lst]
            for p, lst in self.units.items()
        }
        new.turn = self.turn
        new.winner = self.winner
        new.terminal_reason = self.terminal_reason
        new.ACTION_SIZE = self.ACTION_SIZE
        new.PASS_ACTION = self.PASS_ACTION
        return new

    def valid_moves(self, player: Player) -> np.ndarray:
        moves = np.zeros(self.ACTION_SIZE + 1, dtype=np.int8)
        if self.winner is not None:
            moves[-1] = 1
            return moves
        for idx, unit in enumerate(self.units[player]):
            if not unit.alive:
                continue
            base = idx * 6
            neighbors = self.movement.get_native_neighbors(unit.x, unit.y)
            for dir_idx, (nx, ny) in enumerate(neighbors):
                if nx is None or ny is None:
                    continue
                if self.gt and 0 <= ny < self.cfg.map_h and 0 <= nx < self.cfg.map_w and self.gt.au_map[ny, nx] == 'A':
                    moves[base + dir_idx] = 1
        moves[-1] = 1
        return moves

    def step(self, player: Player, action: int) -> None:
        if self.winner is not None:
            return
        if action == self.PASS_ACTION or action < 0 or action > self.ACTION_SIZE:
            self._next_turn()
            return
        unit_idx = action // 6
        dir_idx = action % 6
        if unit_idx >= len(self.units[player]):
            self._next_turn()
            return
        unit = self.units[player][unit_idx]
        if not unit.alive:
            self._next_turn()
            return
        neighbors = self.movement.get_native_neighbors(unit.x, unit.y)
        nx, ny = neighbors[dir_idx]
        if nx is None or ny is None or not self.gt or self.gt.au_map[ny, nx] != 'A':
            self._next_turn()
            return

        # Check for enemy on target tile -> attack, else move
        enemy = self._unit_at(nx, ny, -player)
        if enemy:
            self._attack(unit, enemy)
        else:
            unit.x, unit.y = nx, ny
        self._resolve_terminal()
        self._next_turn()

    def _attack(self, attacker: CombatUnit, defender: CombatUnit) -> None:
        # Simple deterministic damage: attacker deals atk to defender hp, defender counter-attacks with df.
        defender.hp -= attacker.atk
        if defender.hp <= 0:
            defender.alive = False
        else:
            attacker.hp -= max(1, defender.df)
            if attacker.hp <= 0:
                attacker.alive = False

    def _unit_at(self, x: int, y: int, player: Player) -> Optional[CombatUnit]:
        for u in self.units[player]:
            if u.alive and u.x == x and u.y == y:
                return u
        return None

    def _resolve_terminal(self) -> None:
        alive_me = any(u.alive for u in self.units[1])
        alive_opp = any(u.alive for u in self.units[-1])
        if alive_me and not alive_opp:
            self.winner = 1
            self.terminal_reason = "eliminate_opp"
        elif alive_opp and not alive_me:
            self.winner = -1
            self.terminal_reason = "eliminate_me"
        elif not alive_me and not alive_opp:
            self.winner = 0
            self.terminal_reason = "mutual_destruction"

    def _next_turn(self) -> None:
        self.turn += 1
        if self.turn >= self.cfg.max_turns and self.winner is None:
            self.winner = 0
            self.terminal_reason = "max_turns"

    # ---------- encodings ----------
    def encode(self, perspective: Player) -> np.ndarray:
        assert self.gt is not None
        channels = []
        channels.append((self.gt.au_map == 'A').astype(np.float32))
        channels.append((self.gt.au_map == 'U').astype(np.float32))
        unit_me = np.zeros_like(channels[0])
        unit_opp = np.zeros_like(channels[0])
        hp_me = np.zeros_like(channels[0])
        hp_opp = np.zeros_like(channels[0])
        for u in self.units[perspective]:
            if not u.alive:
                continue
            unit_me[u.y, u.x] = 1.0
            hp_me[u.y, u.x] = u.hp / 20.0
        for u in self.units[-perspective]:
            if not u.alive:
                continue
            unit_opp[u.y, u.x] = 1.0
            hp_opp[u.y, u.x] = u.hp / 20.0
        channels.append(unit_me)
        channels.append(unit_opp)
        channels.append(hp_me)
        channels.append(hp_opp)
        turn_plane = np.full_like(channels[0], min(self.turn / max(1, self.cfg.max_turns), 1.0))
        channels.append(turn_plane)
        return np.stack(channels, axis=0)

    def string(self) -> str:
        parts = [f"turn={self.turn}"]
        for p in (1, -1):
            for idx, u in enumerate(self.units[p]):
                if u.alive:
                    parts.append(f"p{p}u{idx}:{u.x},{u.y},hp{u.hp}")
        if self.winner is not None:
            parts.append(f"winner={self.winner}")
        return '|'.join(parts)
