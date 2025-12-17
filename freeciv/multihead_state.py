from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from freeciv_rl.freeciv_movement import FreecivMovement

from .config import MapConfig
from .providers import BaseProvider, GroundTruth
from .research_policy import RESEARCH_TECHS, TARGET_TECH_NAME

Player = int  # 1 or -1
Coord = Tuple[int, int]


@dataclass
class MHUnit:
    x: int
    y: int
    hp: int
    atk: int
    df: int
    alive: bool = True
    can_build_city: bool = False


@dataclass
class MultiheadState:
    """
    Prototype multi-unit state for multi-head training:
    - Multiple units per side (fixed slots, default 4)
    - Move or attack per unit
    - Research actions preserved to keep tech dimensions aligned
    """
    cfg: MapConfig
    provider: BaseProvider
    max_units: int = 4
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    gt: GroundTruth | None = None
    movement: FreecivMovement | None = None
    units: Dict[Player, List[MHUnit]] = field(default_factory=lambda: {1: [], -1: []})
    research_done: Dict[Player, Dict[str, bool]] = field(default_factory=lambda: {1: {}, -1: {}})
    turn: int = 0
    actions_this_turn: int = 0
    max_actions_per_turn: int = 0
    kills: Dict[Player, int] = field(default_factory=lambda: {1: 0, -1: 0})
    winner: Optional[Player] = None
    terminal_reason: Optional[str] = None

    # Action layout:
    # For each unit slot: 6 move + 6 attack = 12 actions
    # After unit actions: research actions | pass
    RESEARCH_TECHS: Tuple[str, ...] = RESEARCH_TECHS
    MOVE_PER_UNIT = 6
    ATTACK_PER_UNIT = 6

    def __post_init__(self) -> None:
        self.movement = FreecivMovement(self.cfg.map_w, self.cfg.map_h)
        # Head sizes
        self.MOVE_SIZE = self.max_units * self.MOVE_PER_UNIT
        self.ATTACK_SIZE = self.max_units * self.ATTACK_PER_UNIT
        # Econ head contains research actions, build-city actions (per unit slot), and a pass/turn-end action.
        self.ECON_RESEARCH_OFFSET = 0
        self.ECON_BUILD_CITY_OFFSET = len(self.RESEARCH_TECHS)
        self.ECON_PASS_OFFSET = len(self.RESEARCH_TECHS) + self.max_units
        self.ECON_SIZE = len(self.RESEARCH_TECHS) + self.max_units + 1
        self.ACTION_SIZE = self.MOVE_SIZE + self.ATTACK_SIZE + self.ECON_SIZE
        self.PASS_ACTION = self.ACTION_SIZE - 1  # last index in econ head
        # Allow multiple actions within the same logical turn; cap to avoid stalling.
        self.max_actions_per_turn = max(1, self.max_units * 2)
        if self.gt is None:
            self.reset()
        if not self.research_done.get(1):
            self.research_done = self._init_research_status()

    def _init_research_status(self) -> Dict[Player, Dict[str, bool]]:
        base = {tech: False for tech in self.RESEARCH_TECHS}
        return {1: dict(base), -1: dict(base)}

    def reset(self) -> None:
        self.gt = self.provider.resample()
        self.turn = 0
        self.actions_this_turn = 0
        self.winner = None
        self.terminal_reason = None
        self.units = {1: [], -1: []}
        self.research_done = self._init_research_status()
        self.kills = {1: 0, -1: 0}
        self._spawn_units()

    def _spawn_units(self) -> None:
        assert self.gt is not None
        # Simple mirrored spawns.
        starts_me = [(0, 0), (0, self.cfg.map_h - 2), (1, self.cfg.map_h - 1), (1, 1)]
        starts_opp = [(self.cfg.map_w - 1, self.cfg.map_h - 1), (self.cfg.map_w - 1, 1), (self.cfg.map_w - 2, 0), (self.cfg.map_w - 2, self.cfg.map_h - 2)]
        stats = [
            {"hp": 10, "atk": 2, "df": 1},  # warrior-ish
            {"hp": 8, "atk": 3, "df": 1},   # archer-ish
            {"hp": 12, "atk": 1, "df": 2},  # phalanx-ish
            {"hp": 9, "atk": 2, "df": 2},   # balanced
        ]
        for idx in range(self.max_units):
            sx, sy = starts_me[idx % len(starts_me)]
            ox, oy = starts_opp[idx % len(starts_opp)]
            st = stats[min(idx, len(stats) - 1)]
            mx, my = self._find_spawn_near(sx, sy)
            ox2, oy2 = self._find_spawn_near(ox, oy)
            self.units[1].append(MHUnit(mx, my, st["hp"], st["atk"], st["df"]))
            self.units[-1].append(MHUnit(ox2, oy2, st["hp"], st["atk"], st["df"]))

    def _find_spawn_near(self, sx: int, sy: int) -> Coord:
        assert self.gt is not None
        sx = int(np.clip(sx, 0, self.cfg.map_w - 1))
        sy = int(np.clip(sy, 0, self.cfg.map_h - 1))
        if self.gt.au_map[sy, sx] == 'A':
            return sx, sy
        frontier = [(sx, sy)]
        seen = {(sx, sy)}
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

    def duplicate(self) -> "MultiheadState":
        new = MultiheadState.__new__(MultiheadState)
        new.cfg = self.cfg
        new.provider = self.provider
        new.max_units = self.max_units
        new.rng = self.rng
        new.movement = self.movement
        new.gt = self.gt.copy() if self.gt else None
        new.units = {
            p: [MHUnit(u.x, u.y, u.hp, u.atk, u.df, u.alive, u.can_build_city) for u in lst]
            for p, lst in self.units.items()
        }
        new.research_done = {p: dict(flags) for p, flags in self.research_done.items()}
        new.turn = self.turn
        new.actions_this_turn = self.actions_this_turn
        new.max_actions_per_turn = self.max_actions_per_turn
        new.kills = dict(self.kills)
        new.winner = self.winner
        new.terminal_reason = self.terminal_reason
        new.RESEARCH_TECHS = self.RESEARCH_TECHS
        new.MOVE_PER_UNIT = self.MOVE_PER_UNIT
        new.ATTACK_PER_UNIT = self.ATTACK_PER_UNIT
        new.MOVE_SIZE = self.MOVE_SIZE
        new.ATTACK_SIZE = self.ATTACK_SIZE
        new.ECON_SIZE = self.ECON_SIZE
        new.ECON_RESEARCH_OFFSET = self.ECON_RESEARCH_OFFSET
        new.ECON_BUILD_CITY_OFFSET = self.ECON_BUILD_CITY_OFFSET
        new.ECON_PASS_OFFSET = self.ECON_PASS_OFFSET
        new.ACTION_SIZE = self.ACTION_SIZE
        new.PASS_ACTION = self.PASS_ACTION
        return new

    def valid_moves(self, player: Player) -> np.ndarray:
        moves = np.zeros(self.ACTION_SIZE, dtype=np.int8)
        if self.winner is not None:
            moves[self.PASS_ACTION] = 1
            return moves
        # move head
        for idx in range(self.max_units):
            move_base = idx * self.MOVE_PER_UNIT
            atk_base = self.MOVE_SIZE + idx * self.ATTACK_PER_UNIT
            u = self.units[player][idx] if idx < len(self.units[player]) else None
            if u is None or not u.alive:
                continue
            neighbors = self.movement.get_native_neighbors(u.x, u.y)
            for dir_idx, (nx, ny) in enumerate(neighbors):
                if nx is None or ny is None:
                    continue
                if self.gt and 0 <= ny < self.cfg.map_h and 0 <= nx < self.cfg.map_w and self.gt.au_map[ny, nx] == 'A':
                    # move only if not blocked by friendly
                    if self._unit_at(nx, ny, player) is None:
                        moves[move_base + dir_idx] = 1
                    # attack only if an enemy occupies the target
                    if self._unit_at(nx, ny, -player) is not None:
                        moves[atk_base + dir_idx] = 1
        # research actions (one-time per tech)
        offset = self.MOVE_SIZE + self.ATTACK_SIZE
        for idx, tech in enumerate(self.RESEARCH_TECHS):
            if not self.research_done[player].get(tech, False):
                moves[offset + idx] = 1
        # build city actions (per unit slot)
        build_offset = offset + self.ECON_BUILD_CITY_OFFSET
        for idx in range(self.max_units):
            u = self.units[player][idx] if idx < len(self.units[player]) else None
            if u is None or not u.alive or not u.can_build_city:
                continue
            moves[build_offset + idx] = 1
        # pass always valid
        moves[self.PASS_ACTION] = 1
        return moves

    def step(self, player: Player, action: int) -> None:
        if self.winner is not None:
            return
        if action == self.PASS_ACTION or action < 0 or action >= self.ACTION_SIZE:
            self._advance_turn()
            return

        if action < self.MOVE_SIZE:
            unit_idx = action // self.MOVE_PER_UNIT
            dir_idx = action % self.MOVE_PER_UNIT
            self._handle_unit_action(player, unit_idx, dir_idx, is_attack=False)
        elif action < self.MOVE_SIZE + self.ATTACK_SIZE:
            rel = action - self.MOVE_SIZE
            unit_idx = rel // self.ATTACK_PER_UNIT
            dir_idx = rel % self.ATTACK_PER_UNIT
            self._handle_unit_action(player, unit_idx, dir_idx, is_attack=True)
        else:
            econ_idx = action - (self.MOVE_SIZE + self.ATTACK_SIZE)
            # research
            if 0 <= econ_idx < len(self.RESEARCH_TECHS):
                tech = self.RESEARCH_TECHS[econ_idx]
                if not self.research_done[player].get(tech, False):
                    self.research_done[player][tech] = True
                # small bonus hooks can be added later
            # build city (per unit slot)
            elif self.ECON_BUILD_CITY_OFFSET <= econ_idx < self.ECON_PASS_OFFSET:
                unit_idx = econ_idx - self.ECON_BUILD_CITY_OFFSET
                if unit_idx < len(self.units[player]):
                    u = self.units[player][unit_idx]
                    if u.alive and u.can_build_city:
                        # For now, treat building a city as consuming the settler unit.
                        u.alive = False
        self._resolve_terminal()
        # Stay in the same turn unless we exceed the per-turn action cap.
        self.actions_this_turn += 1
        if self.actions_this_turn >= self.max_actions_per_turn:
            self._advance_turn()

    def _handle_unit_action(self, player: Player, unit_idx: int, dir_idx: int, is_attack: bool) -> None:
        if unit_idx >= len(self.units[player]):
            return
        u = self.units[player][unit_idx]
        if not u.alive:
            return
        neighbors = self.movement.get_native_neighbors(u.x, u.y)
        nx, ny = neighbors[dir_idx]
        if nx is None or ny is None or not self.gt or self.gt.au_map[ny, nx] != 'A':
            return
        if is_attack:
            enemy = self._unit_at(nx, ny, -player)
            if enemy:
                self._attack(u, enemy)
        else:
            # Move if no friendly blocking
            if self._unit_at(nx, ny, player) is None:
                u.x, u.y = nx, ny

    def _attack(self, attacker: MHUnit, defender: MHUnit) -> None:
        defender.hp -= attacker.atk
        if defender.hp <= 0:
            defender.alive = False
            # Credit the kill to the attacker side.
            for side, lst in self.units.items():
                if defender in lst:
                    self.kills[1 if side == 1 else -1] += 0  # defender side; attacker credited below
                    break
            self.kills[1 if attacker in self.units[1] else -1] += 1
        else:
            attacker.hp -= max(1, defender.df)
            if attacker.hp <= 0:
                attacker.alive = False

    def _unit_at(self, x: int, y: int, player: Player) -> Optional[MHUnit]:
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

    def _advance_turn(self) -> None:
        self.turn += 1
        self.actions_this_turn = 0
        if self.turn >= self.cfg.max_turns and self.winner is None:
            self.winner = 0
            self.terminal_reason = "max_turns"

    def _alive_count(self, player: Player) -> int:
        return sum(1 for u in self.units[player] if u.alive)

    def _hp_sum(self, player: Player) -> int:
        return sum(u.hp for u in self.units[player] if u.alive)

    def heuristic_score(self, player: Player) -> int:
        """
        Heuristic tiebreak when no explicit winner:
        1) kill diff
        2) alive unit count diff
        3) hp sum diff
        returns +1 if player leads, -1 if behind, 0 if equal.
        """
        opp = -player
        kd = self.kills[player] - self.kills[opp]
        if kd != 0:
            return 1 if kd > 0 else -1
        ad = self._alive_count(player) - self._alive_count(opp)
        if ad != 0:
            return 1 if ad > 0 else -1
        hd = self._hp_sum(player) - self._hp_sum(opp)
        if hd != 0:
            return 1 if hd > 0 else -1
        return 0

    # ---------- encodings ----------
    def encode(self, perspective: Player) -> np.ndarray:
        assert self.gt is not None
        me = perspective
        opp = -perspective
        channels: List[np.ndarray] = []
        channels.append((self.gt.au_map == 'A').astype(np.float32))
        channels.append((self.gt.au_map == 'U').astype(np.float32))
        unit_me = np.zeros_like(channels[0])
        unit_opp = np.zeros_like(channels[0])
        hp_me = np.zeros_like(channels[0])
        hp_opp = np.zeros_like(channels[0])
        for u in self.units[me]:
            if not u.alive:
                continue
            unit_me[u.y, u.x] = 1.0
            hp_me[u.y, u.x] = u.hp / 20.0
        for u in self.units[opp]:
            if not u.alive:
                continue
            unit_opp[u.y, u.x] = 1.0
            hp_opp[u.y, u.x] = u.hp / 20.0
        channels.append(unit_me)
        channels.append(unit_opp)
        channels.append(hp_me)
        channels.append(hp_opp)
        # research planes
        for tech in self.RESEARCH_TECHS:
            tme = np.full_like(unit_me, 1.0 if self.research_done[me].get(tech, False) else 0.0)
            topp = np.full_like(unit_me, 1.0 if self.research_done[opp].get(tech, False) else 0.0)
            channels.append(tme)
            channels.append(topp)
        turn_plane = np.full_like(unit_me, min(self.turn / max(1, self.cfg.max_turns), 1.0))
        channels.append(turn_plane)
        return np.stack(channels, axis=0)

    def string(self) -> str:
        parts = [f"turn={self.turn}"]
        parts.append(f"acts={self.actions_this_turn}")
        for p in (1, -1):
            for idx, u in enumerate(self.units[p]):
                if u.alive:
                    parts.append(f"p{p}u{idx}:{u.x},{u.y},hp{u.hp}")
            # Include research status to disambiguate states with identical unit positions.
            res_bits = ''.join('1' if self.research_done[p].get(tech, False) else '0' for tech in self.RESEARCH_TECHS)
            parts.append(f"r{p}:{res_bits}")
        if self.winner is not None:
            parts.append(f"winner={self.winner}")
        parts.append(f"kills:{self.kills.get(1,0)}/{self.kills.get(-1,0)}")
        return '|'.join(parts)
