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
    scores: Dict[Player, float] = field(default_factory=lambda: {1: 0.0, -1: 0.0})
    prev_positions: Dict[Player, Optional[Coord]] = field(default_factory=lambda: {1: None, -1: None})
    cities: Dict[Player, Optional[Coord]] = field(default_factory=lambda: {1: None, -1: None})
    research_complete: Dict[Player, bool] = field(default_factory=lambda: {1: False, -1: False})
    research_done: Dict[Player, Dict[str, bool]] = field(default_factory=lambda: {1: {}, -1: {}})

    SETTLER_MOVE_COUNT = 6
    BUILD_CITY_ACTION = SETTLER_MOVE_COUNT
    TARGET_TECH_NAME = TARGET_TECH_NAME
    RESEARCH_TECHS: Tuple[str, ...] = RESEARCH_TECHS
    RESEARCH_ACTION_BASE = BUILD_CITY_ACTION + 1
    RESEARCH_ACTION_COUNT = len(RESEARCH_TECHS)
    ACTION_SIZE = RESEARCH_ACTION_BASE + RESEARCH_ACTION_COUNT
    PASS_ACTION = ACTION_SIZE

    def __post_init__(self) -> None:
        self.movement = FreecivMovement(self.cfg.map_w, self.cfg.map_h)
        if self.gt is None:
            self.reset()
        if not self.research_done.get(1):
            self.research_done = self._init_research_status()

    def reset(self) -> None:
        self.gt = self.provider.resample()
        self.turn = 0
        self.winner = None
        self.terminal_reason = None
        self.units = {}
        self.visited = {}
        self.revealed = {}
        self.scores = {1: 0.0, -1: 0.0}
        self.prev_positions = {1: None, -1: None}
        self.cities = {1: None, -1: None}
        self.research_complete = {1: False, -1: False}
        self.research_done = self._init_research_status()

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
            self.prev_positions[p] = None

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

    def _init_research_status(self) -> Dict[Player, Dict[str, bool]]:
        base = {tech: False for tech in self.RESEARCH_TECHS}
        return {1: dict(base), -1: dict(base)}

    def _player_idx(self, player: Player) -> int:
        return 0 if player == 1 else 1

    def _reveal(self, player: Player, origin: Optional[Coord] = None) -> None:
        if origin is None:
            origin = self.units[player]
        x, y = origin
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
        new.scores = {p: score for p, score in self.scores.items()}
        new.prev_positions = {p: pos for p, pos in self.prev_positions.items()}
        new.cities = {p: (coord if coord is None else (coord[0], coord[1]))
                      for p, coord in self.cities.items()}
        new.research_complete = {p: flag for p, flag in self.research_complete.items()}
        new.research_done = {p: {tech: status for tech, status in self.research_done.get(p, {}).items()}
                             for p in self.research_done}
        return new

    # ---------- game mechanics ----------
    def valid_moves(self, player: Player) -> np.ndarray:
        moves = np.zeros(self.ACTION_SIZE + 1, dtype=np.int8)
        if self.winner is not None:
            moves[-1] = 1
            return moves
        has_city = self.cities[player] is not None
        # Settler moves
        settler_pos = self.units.get(player)
        if settler_pos is not None:
            neighbors = self.movement.get_native_neighbors(*settler_pos)
            for idx, (nx, ny) in enumerate(neighbors):
                if nx is None:
                    continue
                if self.gt.au_map[ny, nx] == 'A':
                    moves[idx] = 1

        # City build action (only once per player)
        if self.cities[player] is None:
            moves[self.BUILD_CITY_ACTION] = 1

        # Research action (one-time per tech) gated on having a city
        if has_city:
            for idx, tech in enumerate(self.RESEARCH_TECHS):
                if self.research_done[player].get(tech, False):
                    continue
                moves[self.RESEARCH_ACTION_BASE + idx] = 1

        moves[-1] = 1  # allow pass
        return moves

    def step(self, player: Player, action: int) -> None:
        if self.winner is not None:
            return

        prev_pos = self.units[player]

        if action == self.PASS_ACTION:
            self.scores[player] += self.cfg.backtrack_penalty
            self.prev_positions[player] = prev_pos
            self.turn += 1
            if self.turn >= self.cfg.max_turns:
                self._resolve_terminal(reason="max_turns")
            return

        valid = self.valid_moves(player)
        if action < 0 or action > self.ACTION_SIZE or valid[action] == 0:
            # invalid -> treat as pass with penalty by forcing stay
            self.turn += 1
            if self.turn >= self.cfg.max_turns:
                self._resolve_terminal(reason="max_turns")
            return

        acted = False
        if 0 <= action < self.SETTLER_MOVE_COUNT:
            acted = True
            if not self._move_actor(player, action):
                return
        elif action == self.BUILD_CITY_ACTION:
            acted = True
            self._handle_build_city(player)
            if self.winner is not None:
                return
        elif self.RESEARCH_ACTION_BASE <= action < self.RESEARCH_ACTION_BASE + self.RESEARCH_ACTION_COUNT:
            acted = True
            tech_idx = action - self.RESEARCH_ACTION_BASE
            tech_name = self.RESEARCH_TECHS[tech_idx]
            if not self.research_done[player].get(tech_name, False):
                self.research_done[player][tech_name] = True
                if tech_name == self.TARGET_TECH_NAME:
                    self.research_complete[player] = True
                reward = self.cfg.research_reward_map.get(tech_name, self.cfg.research_reward)
                self.scores[player] += reward

        if not acted:
            # treat as pass if somehow no branch matched, though valid() should prevent this
            self.scores[player] += self.cfg.backtrack_penalty

        self.turn += 1
        if self.turn >= self.cfg.max_turns:
            self._resolve_terminal(reason="max_turns")

    def _move_actor(self, player: Player, dir_idx: int) -> bool:
        if self.winner is not None:
            return False

        position = self.units[player]
        if position is None:
            self.scores[player] += self.cfg.wall_penalty
            return False

        neighbors = self.movement.get_native_neighbors(*position)
        nx, ny = neighbors[dir_idx]
        if nx is None or ny is None:
            self.scores[player] += self.cfg.wall_penalty
            return False
        if self.gt.au_map[ny, nx] != 'A':
            self.scores[player] += self.cfg.wall_penalty
            return False

        was_visited = bool(self.visited[player][ny, nx])
        prev_reveal = self.revealed[player].sum()
        self.units[player] = (nx, ny)
        self.visited[player][ny, nx] = True
        self._reveal(player, origin=(nx, ny))
        newly_revealed = self.revealed[player].sum() - prev_reveal
        self._apply_move_rewards(player, was_visited, newly_revealed, position, (nx, ny))

        if self.winner is not None:
            return False

        if self.gt.enemy_map[ny, nx]:
            opponent = -player
            self.scores[opponent] += self.cfg.elimination_bonus
            self.scores[player] -= self.cfg.elimination_bonus
            self._resolve_terminal(winner=opponent, reason="enemy_trap")
            return False

        self._check_collisions(player)
        return self.winner is None

    def _apply_move_rewards(
        self,
        player: Player,
        was_visited: bool,
        newly_revealed: float,
        prev_pos: Coord,
        new_pos: Coord,
    ) -> None:
        # Small reward just for making a move to discourage idling.
        self.scores[player] += getattr(self.cfg, "move_reward", 0.0)

        if newly_revealed > 0:
            self.scores[player] += self.cfg.frontier_bonus * newly_revealed

        if not was_visited:
            self.scores[player] += self.cfg.visit_reward
        else:
            self.scores[player] += self.cfg.backtrack_penalty

        tracker = self.prev_positions
        if tracker[player] is not None and new_pos == tracker[player]:
            self.scores[player] += self.cfg.backtrack_penalty
        tracker[player] = prev_pos

    def _handle_build_city(self, player: Player) -> None:
        if self.cities[player] is not None:
            return
        location = self.units.get(player)
        if location is None:
            return
        self.cities[player] = location
        self.scores[player] += self.cfg.build_city_reward
        self._check_collisions(player)

    def _check_collisions(self, player: Player) -> None:
        if self.winner is not None:
            return
        opponent = -player
        player_positions: List[Tuple[str, Coord]] = []
        opponent_positions: List[Tuple[str, Coord]] = []

        if self.units.get(player) is not None:
            player_positions.append(("settler", self.units[player]))

        if self.units.get(opponent) is not None:
            opponent_positions.append(("settler", self.units[opponent]))

        for pkind, pcoord in player_positions:
            for okind, ocoord in opponent_positions:
                if pcoord == ocoord:
                    self.scores[player] += self.cfg.elimination_bonus
                    self.scores[opponent] -= self.cfg.elimination_bonus
                    self.units[opponent] = ocoord
                    self._resolve_terminal(winner=player, reason=f"{pkind}_capture")
                    return

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
            normalized = self.cfg.draw_value

        score_delta = self.scores[1] - self.scores[-1]
        score_scale = (
            self.cfg.map_w * self.cfg.map_h * max(self.cfg.visit_reward + abs(self.cfg.frontier_bonus), 1e-3)
            + self.cfg.max_turns * abs(self.cfg.backtrack_penalty)
            + max(self.cfg.elimination_bonus, 0.0)
        )
        score_component = np.clip(score_delta / max(1.0, score_scale), -1.0, 1.0)

        return float(np.clip(normalized + score_component, -1.0, 1.0))

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
        city_me = np.zeros_like(unit_me)
        city_opp = np.zeros_like(unit_me)
        if self.cities[me] is not None:
            cx, cy = self.cities[me]
            city_me[cy, cx] = 1.0
        if self.cities[opp] is not None:
            cx, cy = self.cities[opp]
            city_opp[cy, cx] = 1.0
        channels.append(city_me)
        channels.append(city_opp)
        channels.append(self.visited[me].astype(np.float32))
        channels.append(self.visited[opp].astype(np.float32))
        channels.append(self.gt.enemy_map.astype(np.float32))
        research_me = np.full_like(unit_me, 1.0 if self.research_complete.get(me, False) else 0.0)
        research_opp = np.full_like(unit_me, 1.0 if self.research_complete.get(opp, False) else 0.0)
        channels.append(research_me)
        channels.append(research_opp)
        for tech in self.RESEARCH_TECHS:
            tech_me = np.full_like(unit_me, 1.0 if self.research_done.get(me, {}).get(tech, False) else 0.0)
            tech_opp = np.full_like(unit_me, 1.0 if self.research_done.get(opp, {}).get(tech, False) else 0.0)
            channels.append(tech_me)
            channels.append(tech_opp)
        stacked = np.stack(channels, axis=0)
        return stacked

    def string(self) -> str:
        parts = [f"turn={self.turn}"]
        for p in (1, -1):
            x, y = self.units[p]
            parts.append(f"p{p}:{x},{y}")
            if self.cities[p] is not None:
                cx, cy = self.cities[p]
                parts.append(f"city{p}:{cx},{cy}")
            parts.append(f"research{p}:{self.research_complete[p]}")
            done_labels = ",".join(
                tech for tech, done in self.research_done.get(p, {}).items() if done
            )
            if done_labels:
                parts.append(f"techs{p}:{done_labels}")
        if self.winner:
            parts.append(f"winner={self.winner}")
        return '|'.join(parts)
