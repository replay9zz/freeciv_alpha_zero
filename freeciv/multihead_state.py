from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from freeciv_rl.freeciv_movement import FreecivMovement

from .config import MapConfig
from .providers import BaseProvider, GroundTruth
from .research_policy import (
    RESEARCH_TECHS,
    TARGET_TECH_NAME,
    TECH_COSTS,
    TECH_PREREQS,
    TECH_COST_STYLE,
    BASE_TECH_COST,
    MIN_TECH_COST,
    TECH_COST_FACTOR,
    build_tech_costs,
)
from .ruleset_loader import load_civ2civ3_ruleset

Player = int  # 1 or -1
Coord = Tuple[int, int]


@dataclass
class UnitSpec:
    name: str
    atk: int
    df: int
    hp: int
    firepower: int
    moves: int
    cost: int
    can_build_city: bool = False
    obsolete_by: Optional[str] = None


@dataclass
class BuildingSpec:
    name: str
    cost: int
    req_techs: List[str] = field(default_factory=list)
    req_buildings: List[str] = field(default_factory=list)
    genus: str = ""
    flags: List[str] = field(default_factory=list)


@dataclass
class City:
    x: int
    y: int
    size: int = 1
    food_storage: float = 0.0
    production_kind: Optional[str] = None  # "unit" or "building"
    production_target: Optional[str] = None
    production_progress: float = 0.0
    has_city_walls: bool = False
    buildings: set[str] = field(default_factory=set)


@dataclass
class MHUnit:
    x: int
    y: int
    hp: int
    atk: int
    df: int
    firepower: int
    unit_type: str
    alive: bool = True
    can_build_city: bool = False
    home_city: Optional[int] = None
    last_move_turn: int = -1


_RULESET = load_civ2civ3_ruleset()

PRODUCTION_UNIT_NAMES: Tuple[str, ...] = tuple(rule.name for rule in _RULESET.units)
PRODUCTION_BUILDING_NAMES: Tuple[str, ...] = tuple(
    rule.name for rule in _RULESET.buildings
)
PRODUCTION_ITEM_NAMES: Tuple[Tuple[str, str], ...] = tuple(
    [("unit", name) for name in PRODUCTION_UNIT_NAMES]
    + [("building", name) for name in PRODUCTION_BUILDING_NAMES]
)

UNIT_SPECS: Dict[str, UnitSpec] = {}
UNIT_TECHS: Dict[str, List[str]] = {}
UNIT_OBSOLETE_BY: Dict[str, Optional[str]] = {}
for rule in _RULESET.units:
    can_build = "Cities" in rule.flags
    UNIT_SPECS[rule.name] = UnitSpec(
        name=rule.name,
        atk=rule.attack,
        df=rule.defense,
        hp=rule.hp,
        firepower=rule.firepower,
        moves=rule.moves,
        cost=rule.cost,
        can_build_city=can_build,
        obsolete_by=rule.obsolete_by,
    )
    UNIT_TECHS[rule.name] = list(rule.req_techs)
    UNIT_OBSOLETE_BY[rule.name] = rule.obsolete_by

BUILDING_SPECS: Dict[str, BuildingSpec] = {}
BUILDING_TECHS: Dict[str, List[str]] = {}
BUILDING_REQ_BUILDINGS: Dict[str, List[str]] = {}
GREAT_WONDER_NAMES: set[str] = set()
for rule in _RULESET.buildings:
    BUILDING_SPECS[rule.name] = BuildingSpec(
        name=rule.name,
        cost=rule.cost,
        req_techs=list(rule.req_techs),
        req_buildings=list(rule.req_buildings),
        genus=rule.genus,
        flags=list(rule.flags),
    )
    BUILDING_TECHS[rule.name] = list(rule.req_techs)
    BUILDING_REQ_BUILDINGS[rule.name] = list(rule.req_buildings)
    if rule.genus == "GreatWonder":
        GREAT_WONDER_NAMES.add(rule.name)


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
    max_units: int = 6
    max_cities: int = 3
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    gt: GroundTruth | None = None
    movement: FreecivMovement | None = None
    units: Dict[Player, List[MHUnit]] = field(default_factory=lambda: {1: [], -1: []})
    cities: Dict[Player, List[City]] = field(default_factory=lambda: {1: [], -1: []})
    research_done: Dict[Player, Dict[str, bool]] = field(default_factory=lambda: {1: {}, -1: {}})
    research_target: Dict[Player, Optional[str]] = field(
        default_factory=lambda: {1: None, -1: None}
    )
    research_progress: Dict[Player, float] = field(
        default_factory=lambda: {1: 0.0, -1: 0.0}
    )
    visited: Dict[Player, np.ndarray] = field(default_factory=lambda: {1: None, -1: None})
    turn: int = 0
    actions_this_turn: int = 0
    max_actions_per_turn: int = 0
    acted_unit_slots: Dict[Player, set[int]] = field(
        default_factory=lambda: {1: set(), -1: set()}
    )
    kills: Dict[Player, int] = field(default_factory=lambda: {1: 0, -1: 0})
    units_built: Dict[Player, int] = field(default_factory=lambda: {1: 0, -1: 0})
    future_techs: Dict[Player, int] = field(default_factory=lambda: {1: 0, -1: 0})
    scores: Dict[Player, float] = field(default_factory=lambda: {1: 0.0, -1: 0.0})
    winner: Optional[Player] = None
    terminal_reason: Optional[str] = None
    tech_costs: Dict[str, float] = field(default_factory=lambda: dict(TECH_COSTS))

    # Action layout:
    # For each unit slot: 6 move + 6 attack = 12 actions
    # After unit actions: research actions | pass
    RESEARCH_TECHS: Tuple[str, ...] = RESEARCH_TECHS
    PRODUCTION_UNIT_NAMES: Tuple[str, ...] = PRODUCTION_UNIT_NAMES
    PRODUCTION_BUILDING_NAMES: Tuple[str, ...] = PRODUCTION_BUILDING_NAMES
    PRODUCTION_ITEM_NAMES: Tuple[Tuple[str, str], ...] = PRODUCTION_ITEM_NAMES
    MOVE_PER_UNIT = 6
    ATTACK_PER_UNIT = 6

    def __post_init__(self) -> None:
        self.movement = FreecivMovement(self.cfg.map_w, self.cfg.map_h)
        # Head sizes
        self.MOVE_SIZE = self.max_units * self.MOVE_PER_UNIT
        self.ATTACK_SIZE = self.max_units * self.ATTACK_PER_UNIT
        # Econ head contains research actions, build-city actions (per unit slot),
        # production actions (per city slot), and a pass/turn-end action.
        self.ECON_RESEARCH_OFFSET = 0
        self.ECON_BUILD_CITY_OFFSET = len(self.RESEARCH_TECHS)
        self.ECON_PRODUCTION_OFFSET = self.ECON_BUILD_CITY_OFFSET + self.max_units
        self.PRODUCTION_ITEM_COUNT = len(PRODUCTION_ITEM_NAMES)
        self.PRODUCTION_UNIT_COUNT = len(PRODUCTION_UNIT_NAMES)
        self.ECON_PASS_OFFSET = (
            self.ECON_PRODUCTION_OFFSET + self.max_cities * self.PRODUCTION_ITEM_COUNT
        )
        self.ECON_SIZE = self.ECON_PASS_OFFSET + 1
        self.ACTION_SIZE = self.MOVE_SIZE + self.ATTACK_SIZE + self.ECON_SIZE
        self.PASS_ACTION = self.ACTION_SIZE - 1  # last index in econ head
        # Allow multiple actions within the same logical turn; cap to avoid stalling.
        self.max_actions_per_turn = max(1, self.max_units * 2)
        if self.gt is None:
            self.reset()
        if not self.research_done.get(1):
            self.research_done = self._init_research_status()
        self._refresh_tech_costs()

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
        self.cities = {1: [], -1: []}
        self.research_done = self._init_research_status()
        self.research_target = {1: None, -1: None}
        self.research_progress = {1: 0.0, -1: 0.0}
        self.visited = {
            1: np.zeros((self.cfg.map_h, self.cfg.map_w), dtype=bool),
            -1: np.zeros((self.cfg.map_h, self.cfg.map_w), dtype=bool),
        }
        self.kills = {1: 0, -1: 0}
        self.units_built = {1: 0, -1: 0}
        self.future_techs = {1: 0, -1: 0}
        self.scores = {1: 0.0, -1: 0.0}
        self.acted_unit_slots = {1: set(), -1: set()}
        self._refresh_tech_costs()
        self._spawn_units()

    def _refresh_tech_costs(self) -> None:
        style = getattr(self.cfg, "tech_cost_style", TECH_COST_STYLE)
        base_cost = getattr(self.cfg, "base_tech_cost", BASE_TECH_COST)
        min_cost = getattr(self.cfg, "min_tech_cost", MIN_TECH_COST)
        cost_factor = getattr(self.cfg, "tech_cost_factor", TECH_COST_FACTOR)
        self.tech_costs = build_tech_costs(
            TECH_PREREQS,
            style=style,
            base_cost=base_cost,
            min_cost=min_cost,
            cost_factor=cost_factor,
        )

    def _spawn_units(self) -> None:
        assert self.gt is not None
        starts_me = [(0, 0)]
        starts_opp = [(self.cfg.map_w - 1, self.cfg.map_h - 1)]
        spawn_units = ("Settlers", "Workers", "Explorer", "Diplomat")
        missing = [name for name in spawn_units if name not in UNIT_SPECS]
        if missing:
            raise RuntimeError(
                "Missing unit specs for start units: " + ", ".join(missing)
            )
        offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
        mx, my = starts_me[0]
        ox, oy = starts_opp[0]
        for idx, name in enumerate(spawn_units):
            dx, dy = offsets[idx] if idx < len(offsets) else (0, 0)
            spec = UNIT_SPECS[name]
            ux, uy = self._find_spawn_near(mx + dx, my + dy)
            self.units[1].append(
                MHUnit(
                    ux,
                    uy,
                    spec.hp,
                    spec.atk,
                    spec.df,
                    spec.firepower,
                    spec.name,
                    True,
                    spec.can_build_city,
                )
            )
            vx, vy = self._find_spawn_near(ox - dx, oy - dy)
            self.units[-1].append(
                MHUnit(
                    vx,
                    vy,
                    spec.hp,
                    spec.atk,
                    spec.df,
                    spec.firepower,
                    spec.name,
                    True,
                    spec.can_build_city,
                )
            )
        self._ensure_unit_slots()
        for player in (1, -1):
            for u in self.units[player]:
                if u.alive:
                    self.visited[player][u.y, u.x] = True

    def _ensure_unit_slots(self) -> None:
        for player in (1, -1):
            while len(self.units[player]) < self.max_units:
                self.units[player].append(
                    MHUnit(
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        "None",
                        False,
                        False,
                        None,
                    )
                )

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

    def _city_spacing_ok(self, player: Player, x: int, y: int) -> bool:
        min_dist = getattr(self.cfg, "city_min_distance", 0)
        if min_dist <= 0:
            return True
        if not self.cities.get(player):
            return True
        frontier = deque()
        frontier.append((x, y, 0))
        seen = {(x, y)}
        while frontier:
            cx, cy, dist = frontier.popleft()
            if dist >= min_dist:
                continue
            if self._city_at(cx, cy, player) is not None:
                return False
            for nx, ny in self.movement.get_native_neighbors(cx, cy):
                if nx is None or ny is None:
                    continue
                if (nx, ny) in seen:
                    continue
                if self.gt and self.gt.au_map[ny, nx] != "A":
                    continue
                seen.add((nx, ny))
                frontier.append((nx, ny, dist + 1))
        return True

    def duplicate(self) -> "MultiheadState":
        new = MultiheadState.__new__(MultiheadState)
        new.cfg = self.cfg
        new.provider = self.provider
        new.max_units = self.max_units
        new.max_cities = self.max_cities
        new.rng = self.rng
        new.movement = self.movement
        new.gt = self.gt.copy() if self.gt else None
        new.units = {
            p: [
                MHUnit(
                    u.x,
                    u.y,
                    u.hp,
                    u.atk,
                    u.df,
                    u.firepower,
                    u.unit_type,
                    u.alive,
                    u.can_build_city,
                    u.home_city,
                    u.last_move_turn,
                )
                for u in lst
            ]
            for p, lst in self.units.items()
        }
        new.cities = {
            p: [
                City(
                    x=c.x,
                    y=c.y,
                    size=c.size,
                    food_storage=c.food_storage,
                    production_kind=c.production_kind,
                    production_target=c.production_target,
                    production_progress=c.production_progress,
                    has_city_walls=c.has_city_walls,
                    buildings=set(c.buildings),
                )
                for c in lst
            ]
            for p, lst in self.cities.items()
        }
        new.research_done = {p: dict(flags) for p, flags in self.research_done.items()}
        new.research_target = {
            p: target for p, target in self.research_target.items()
        }
        new.research_progress = {
            p: float(val) for p, val in self.research_progress.items()
        }
        new.visited = {
            1: self.visited[1].copy(),
            -1: self.visited[-1].copy(),
        }
        new.turn = self.turn
        new.actions_this_turn = self.actions_this_turn
        new.max_actions_per_turn = self.max_actions_per_turn
        new.acted_unit_slots = {
            1: set(self.acted_unit_slots.get(1, set())),
            -1: set(self.acted_unit_slots.get(-1, set())),
        }
        new.kills = dict(self.kills)
        new.units_built = dict(self.units_built)
        new.future_techs = dict(self.future_techs)
        new.scores = dict(self.scores)
        new.winner = self.winner
        new.terminal_reason = self.terminal_reason
        new.tech_costs = dict(self.tech_costs)
        new.RESEARCH_TECHS = self.RESEARCH_TECHS
        new.PRODUCTION_UNIT_NAMES = self.PRODUCTION_UNIT_NAMES
        new.PRODUCTION_BUILDING_NAMES = self.PRODUCTION_BUILDING_NAMES
        new.PRODUCTION_ITEM_NAMES = self.PRODUCTION_ITEM_NAMES
        new.MOVE_PER_UNIT = self.MOVE_PER_UNIT
        new.ATTACK_PER_UNIT = self.ATTACK_PER_UNIT
        new.MOVE_SIZE = self.MOVE_SIZE
        new.ATTACK_SIZE = self.ATTACK_SIZE
        new.ECON_SIZE = self.ECON_SIZE
        new.ECON_RESEARCH_OFFSET = self.ECON_RESEARCH_OFFSET
        new.ECON_BUILD_CITY_OFFSET = self.ECON_BUILD_CITY_OFFSET
        new.ECON_PRODUCTION_OFFSET = self.ECON_PRODUCTION_OFFSET
        new.ECON_PASS_OFFSET = self.ECON_PASS_OFFSET
        new.PRODUCTION_ITEM_COUNT = self.PRODUCTION_ITEM_COUNT
        new.PRODUCTION_UNIT_COUNT = self.PRODUCTION_UNIT_COUNT
        new.ACTION_SIZE = self.ACTION_SIZE
        new.PASS_ACTION = self.PASS_ACTION
        return new

    def valid_moves(self, player: Player) -> np.ndarray:
        moves = np.zeros(self.ACTION_SIZE, dtype=np.int8)
        if self.winner is not None:
            moves[self.PASS_ACTION] = 1
            return moves
        # move head
        acted_slots = self.acted_unit_slots.get(player, set())
        for idx in range(self.max_units):
            move_base = idx * self.MOVE_PER_UNIT
            atk_base = self.MOVE_SIZE + idx * self.ATTACK_PER_UNIT
            u = self.units[player][idx] if idx < len(self.units[player]) else None
            if u is None or not u.alive or idx in acted_slots:
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
                    if u.atk > 0 and (
                        self._unit_at(nx, ny, -player) is not None
                        or self._city_at(nx, ny, -player) is not None
                    ):
                        moves[atk_base + dir_idx] = 1
        # research actions (select current target)
        offset = self.MOVE_SIZE + self.ATTACK_SIZE
        current_target = self.research_target.get(player)
        if current_target and self.research_done[player].get(current_target, False):
            current_target = None
        if current_target is None:
            for idx, tech in enumerate(self.RESEARCH_TECHS):
                if self.research_done[player].get(tech, False):
                    continue
                prereqs = TECH_PREREQS.get(tech, [])
                if any(
                    not self.research_done[player].get(req, False) for req in prereqs
                ):
                    continue
                moves[offset + idx] = 1
        # build city actions (per unit slot)
        build_offset = offset + self.ECON_BUILD_CITY_OFFSET
        if len(self.cities[player]) < self.max_cities:
            for idx in range(self.max_units):
                u = self.units[player][idx] if idx < len(self.units[player]) else None
                if u is None or not u.alive or not u.can_build_city or idx in acted_slots:
                    continue
                if self._city_at(u.x, u.y, player) is not None:
                    continue
                if not self._city_spacing_ok(player, u.x, u.y):
                    continue
                moves[build_offset + idx] = 1
        # production actions (per city slot)
        prod_offset = offset + self.ECON_PRODUCTION_OFFSET
        for city_idx in range(min(len(self.cities[player]), self.max_cities)):
            city = self.cities[player][city_idx]
            for item_idx, (kind, name) in enumerate(PRODUCTION_ITEM_NAMES):
                if kind == "unit":
                    if name == "Settlers" and city.size <= 1:
                        continue
                    if self._city_unit_count(player, city_idx) >= self.cfg.city_unit_cap:
                        continue
                    if not self._unit_unlocked(player, name):
                        continue
                    if self._unit_obsolete(player, name):
                        continue
                else:
                    if not self._building_unlocked(player, city, name):
                        continue
                moves[
                    prod_offset + city_idx * self.PRODUCTION_ITEM_COUNT + item_idx
                ] = 1
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
            self.acted_unit_slots.setdefault(player, set()).add(unit_idx)
        elif action < self.MOVE_SIZE + self.ATTACK_SIZE:
            rel = action - self.MOVE_SIZE
            unit_idx = rel // self.ATTACK_PER_UNIT
            dir_idx = rel % self.ATTACK_PER_UNIT
            self._handle_unit_action(player, unit_idx, dir_idx, is_attack=True)
            self.acted_unit_slots.setdefault(player, set()).add(unit_idx)
        else:
            econ_idx = action - (self.MOVE_SIZE + self.ATTACK_SIZE)
            # research
            if 0 <= econ_idx < len(self.RESEARCH_TECHS):
                tech = self.RESEARCH_TECHS[econ_idx]
                if (
                    not self.research_done[player].get(tech, False)
                    and self.research_target.get(player) != tech
                ):
                    prereqs = TECH_PREREQS.get(tech, [])
                    if all(
                        self.research_done[player].get(req, False) for req in prereqs
                    ):
                        self.research_target[player] = tech
                # small bonus hooks can be added later
            # build city (per unit slot)
            elif self.ECON_BUILD_CITY_OFFSET <= econ_idx < self.ECON_PRODUCTION_OFFSET:
                unit_idx = econ_idx - self.ECON_BUILD_CITY_OFFSET
                if unit_idx < len(self.units[player]) and len(self.cities[player]) < self.max_cities:
                    u = self.units[player][unit_idx]
                    if (
                        u.alive
                        and u.can_build_city
                        and self._city_at(u.x, u.y, player) is None
                        and self._city_spacing_ok(player, u.x, u.y)
                    ):
                        u.alive = False
                        self._add_city(player, u.x, u.y)
                        self.scores[player] += self.cfg.build_city_reward
                        self.acted_unit_slots.setdefault(player, set()).add(unit_idx)
            # production selection
            elif self.ECON_PRODUCTION_OFFSET <= econ_idx < self.ECON_PASS_OFFSET:
                rel = econ_idx - self.ECON_PRODUCTION_OFFSET
                city_slot = rel // self.PRODUCTION_ITEM_COUNT
                item_idx = rel % self.PRODUCTION_ITEM_COUNT
                if city_slot < len(self.cities[player]):
                    city = self.cities[player][city_slot]
                    kind, name = PRODUCTION_ITEM_NAMES[item_idx]
                    if kind == "unit":
                        if self._unit_unlocked(player, name):
                            if (
                                city.production_kind != "unit"
                                or city.production_target != name
                            ):
                                city.production_kind = "unit"
                                city.production_target = name
                                city.production_progress = 0.0
                    else:
                        if self._building_unlocked(player, city, name):
                            if (
                                city.production_kind != "building"
                                or city.production_target != name
                            ):
                                city.production_kind = "building"
                                city.production_target = name
                                city.production_progress = 0.0
        self._resolve_terminal()
        # Stay in the same turn unless we exceed the per-turn action cap.
        self.actions_this_turn += 1
        if self.actions_this_turn >= self.max_actions_per_turn:
            self._advance_turn()

    def _handle_unit_action(
        self, player: Player, unit_idx: int, dir_idx: int, is_attack: bool
    ) -> None:
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
            if u.atk <= 0:
                return
            attack_reward = getattr(self.cfg, "attack_reward", 0.0)
            enemy = self._unit_at(nx, ny, -player)
            if enemy:
                if attack_reward:
                    self.scores[player] += attack_reward
                    attack_reward = 0.0
                self._attack(player, u, enemy, -player)
                if enemy.alive:
                    return
                if not u.alive:
                    return
                city_idx = self._city_at_index(nx, ny, -player)
                if city_idx is not None:
                    if attack_reward:
                        self.scores[player] += attack_reward
                        attack_reward = 0.0
                    self._attack_city(player, u, -player, city_idx)
            else:
                city_idx = self._city_at_index(nx, ny, -player)
                if city_idx is not None:
                    if attack_reward:
                        self.scores[player] += attack_reward
                        attack_reward = 0.0
                    self._attack_city(player, u, -player, city_idx)
        else:
            # Move if no friendly blocking
            if self._unit_at(nx, ny, player) is None:
                u.x, u.y = nx, ny
                u.last_move_turn = self.turn
                if not self.visited[player][ny, nx]:
                    self.visited[player][ny, nx] = True
                    self.scores[player] += self.cfg.move_reward

    def _attack(
        self,
        player: Player,
        attacker: MHUnit,
        defender: MHUnit,
        defender_player: Player,
    ) -> None:
        if attacker.atk <= 0:
            return
        atk = max(1, attacker.atk)
        df = max(1, defender.df)
        city = self._city_at(defender.x, defender.y, defender_player)
        if city is not None:
            df *= max(0.1, getattr(self.cfg, "city_defense_multiplier", 1.0))
            if city.has_city_walls:
                df *= max(
                    0.1, getattr(self.cfg, "city_walls_defense_multiplier", 1.0)
                )
        fatigue_mult = getattr(self.cfg, "move_fatigue_defense_multiplier", 1.0)
        if (
            fatigue_mult != 1.0
            and self.turn > 0
            and defender.last_move_turn == self.turn - 1
        ):
            df = max(0.1, df * fatigue_mult)
        p_hit = atk / float(atk + df)
        while attacker.hp > 0 and defender.hp > 0:
            if self.rng.random() < p_hit:
                defender.hp -= max(1, attacker.firepower)
            else:
                attacker.hp -= max(1, defender.firepower)

        if defender.hp <= 0:
            defender.alive = False
            self.kills[player] += 1
            self.scores[player] += self.cfg.elimination_bonus
            self.scores[-player] -= self.cfg.elimination_bonus
        if attacker.hp <= 0:
            attacker.alive = False

    def _attack_city(
        self,
        player: Player,
        attacker: MHUnit,
        defender: Player,
        city_idx: int,
    ) -> None:
        if attacker.atk <= 0:
            return
        if city_idx < 0 or city_idx >= len(self.cities[defender]):
            return
        city = self.cities[defender][city_idx]
        if self._unit_at(city.x, city.y, defender) is not None:
            return
        self._remove_city(defender, city_idx)
        self.scores[player] += self.cfg.city_capture_reward
        self.scores[-player] -= self.cfg.city_capture_reward

    def _unit_at(self, x: int, y: int, player: Player) -> Optional[MHUnit]:
        for u in self.units[player]:
            if u.alive and u.x == x and u.y == y:
                return u
        return None

    def _city_at(self, x: int, y: int, player: Player) -> Optional[City]:
        for city in self.cities[player]:
            if city.x == x and city.y == y:
                return city
        return None

    def _city_at_index(self, x: int, y: int, player: Player) -> Optional[int]:
        for idx, city in enumerate(self.cities[player]):
            if city.x == x and city.y == y:
                return idx
        return None

    def _unit_unlocked(self, player: Player, unit_name: str) -> bool:
        techs = UNIT_TECHS.get(unit_name, [])
        if not techs:
            return True
        return all(self.research_done[player].get(tech, False) for tech in techs)

    def _unit_is_obsolete_exempt(self, unit_name: str) -> bool:
        label = (unit_name or "").lower()
        return any(tag in label for tag in ("settler", "worker", "engineer", "migrant"))

    def _unit_obsolete(self, player: Player, unit_name: str) -> bool:
        if self._unit_is_obsolete_exempt(unit_name):
            return False
        obsolete_by = UNIT_OBSOLETE_BY.get(unit_name)
        if not obsolete_by:
            return False
        return self._unit_unlocked(player, obsolete_by)

    def _building_is_palace(self, building_name: str) -> bool:
        return "palace" in (building_name or "").lower()

    def _building_unlocked(
        self, player: Player, city: City, building_name: str
    ) -> bool:
        if self._building_is_palace(building_name):
            return False
        if building_name in city.buildings:
            return False
        techs = BUILDING_TECHS.get(building_name, [])
        if techs and not all(
            self.research_done[player].get(tech, False) for tech in techs
        ):
            return False
        req_buildings = BUILDING_REQ_BUILDINGS.get(building_name, [])
        if req_buildings and not all(req in city.buildings for req in req_buildings):
            return False
        return True

    def _city_unit_count(self, player: Player, city_idx: int) -> int:
        return sum(
            1
            for u in self.units[player]
            if u.alive and u.home_city == city_idx
        )

    def _add_city(
        self, player: Player, x: int, y: int, *, has_city_walls: bool = False
    ) -> None:
        if len(self.cities[player]) >= self.max_cities:
            return
        city = City(x=x, y=y, has_city_walls=has_city_walls)
        if has_city_walls:
            city.buildings.add("City Walls")
        self.cities[player].append(city)

    def _remove_city(self, player: Player, city_idx: int) -> None:
        if city_idx < 0 or city_idx >= len(self.cities[player]):
            return
        del self.cities[player][city_idx]
        for u in self.units[player]:
            if not u.alive or u.home_city is None:
                continue
            if u.home_city == city_idx:
                u.home_city = None
            elif u.home_city > city_idx:
                u.home_city -= 1

    def _place_unit(
        self, player: Player, unit: MHUnit, city_idx: Optional[int]
    ) -> bool:
        for slot in self.units[player]:
            if not slot.alive:
                slot.x = unit.x
                slot.y = unit.y
                slot.hp = unit.hp
                slot.atk = unit.atk
                slot.df = unit.df
                slot.firepower = unit.firepower
                slot.unit_type = unit.unit_type
                slot.alive = True
                slot.can_build_city = unit.can_build_city
                slot.home_city = city_idx
                slot.last_move_turn = unit.last_move_turn
                return True
        if len(self.units[player]) < self.max_units:
            unit.home_city = city_idx
            self.units[player].append(unit)
            return True
        return False

    def _spawn_from_city(self, player: Player, city_idx: int, unit_name: str) -> bool:
        if city_idx >= len(self.cities[player]):
            return False
        if self._city_unit_count(player, city_idx) >= self.cfg.city_unit_cap:
            return False
        spec = UNIT_SPECS.get(unit_name)
        if spec is None:
            return False
        city = self.cities[player][city_idx]
        candidates = [(city.x, city.y)] + [
            (nx, ny)
            for nx, ny in self.movement.get_native_neighbors(city.x, city.y)
            if nx is not None and ny is not None
        ]
        for cx, cy in candidates:
            if (
                self._unit_at(cx, cy, player) is None
                and self.gt
                and 0 <= cy < self.cfg.map_h
                and 0 <= cx < self.cfg.map_w
                and self.gt.au_map[cy, cx] == 'A'
            ):
                unit = MHUnit(
                    cx,
                    cy,
                    spec.hp,
                    spec.atk,
                    spec.df,
                    spec.firepower,
                    spec.name,
                    True,
                    spec.can_build_city,
                    city_idx,
                )
                if self._place_unit(player, unit, city_idx):
                    self.units_built[player] = self.units_built.get(player, 0) + 1
                    return True
                return False
        return False

    def _apply_city_economy(self) -> None:
        bulbs_by_player = {1: 0.0, -1: 0.0}
        for player in (1, -1):
            for city_idx, city in enumerate(self.cities[player]):
                size = max(1, city.size)
                total_food = self.cfg.city_food + self.cfg.grass_food * size
                total_shields = (
                    self.cfg.city_shield + self.cfg.grass_shield * size
                ) * getattr(self.cfg, "production_rate_multiplier", 1.0)
                total_trade = self.cfg.city_trade + self.cfg.grass_trade * size
                science_rate = getattr(self.cfg, "tax_science_rate", 0.0)
                bulbs_by_player[player] += (
                    total_trade
                    * science_rate
                    * getattr(self.cfg, "research_rate_multiplier", 1.0)
                )

                food_surplus = total_food - self.cfg.food_consumption * size
                if food_surplus > 0:
                    city.food_storage += food_surplus
                    if city.food_storage >= self.cfg.food_growth:
                        city.food_storage -= self.cfg.food_growth
                        city.size += 1
                else:
                    city.food_storage = max(0.0, city.food_storage + food_surplus)

                city.production_progress += total_shields
                if city.production_target and city.production_kind == "unit":
                    spec = UNIT_SPECS.get(city.production_target)
                    if spec and city.production_progress >= spec.cost:
                        if self._spawn_from_city(
                            player, city_idx, city.production_target
                        ):
                            city.production_progress -= spec.cost
                elif city.production_target and city.production_kind == "building":
                    spec = BUILDING_SPECS.get(city.production_target)
                    if (
                        spec
                        and city.production_progress >= spec.cost
                        and city.production_target not in city.buildings
                    ):
                        city.buildings.add(city.production_target)
                        if city.production_target == "City Walls":
                            city.has_city_walls = True
                        city.production_progress -= spec.cost
                        city.production_kind = None
                        city.production_target = None

                # Research bulbs are added from trade; completion handled below.
        self._apply_research_bulbs(bulbs_by_player)

    def _apply_research_bulbs(self, bulbs_by_player: Dict[Player, float]) -> None:
        for player, bulbs in bulbs_by_player.items():
            if bulbs != 0:
                self.research_progress[player] = (
                    self.research_progress.get(player, 0.0) + bulbs
                )
            target = self.research_target.get(player)
            if not target:
                continue
            if self.research_done[player].get(target, False):
                self.research_target[player] = None
                continue
            cost = TECH_COSTS.get(target, 0.0)
            if self.tech_costs:
                cost = self.tech_costs.get(target, cost)
            if cost <= 0:
                cost = getattr(self.cfg, "base_tech_cost", 10.0)
            if self.research_progress[player] >= cost:
                self.research_progress[player] -= cost
                self.research_done[player][target] = True
                reward = self.cfg.research_reward_map.get(
                    target, self.cfg.research_reward
                )
                self.scores[player] += reward
                self.research_target[player] = None

    def _resolve_terminal(self) -> None:
        alive_me = any(u.alive for u in self.units[1]) or bool(self.cities[1])
        alive_opp = any(u.alive for u in self.units[-1]) or bool(self.cities[-1])
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
        self._apply_city_economy()
        self.turn += 1
        self.actions_this_turn = 0
        self.acted_unit_slots = {1: set(), -1: set()}
        if self.turn >= self.cfg.max_turns and self.winner is None:
            self.winner = 0
            self.terminal_reason = "max_turns"

    def _alive_count(self, player: Player) -> int:
        return sum(1 for u in self.units[player] if u.alive)

    def _hp_sum(self, player: Player) -> int:
        return sum(u.hp for u in self.units[player] if u.alive)

    def civilization_score(self, player: Player) -> float:
        pop = sum(city.size for city in self.cities[player])
        techs = sum(1 for done in self.research_done[player].values() if done)
        future = self.future_techs.get(player, 0)
        wonders = sum(
            1
            for city in self.cities[player]
            for building in city.buildings
            if building in GREAT_WONDER_NAMES
        )
        units_built = self.units_built.get(player, 0)
        kills = self.kills.get(player, 0)
        score = (
            pop * self.cfg.score_population
            + techs * self.cfg.score_tech
            + future * self.cfg.score_future_tech
            + wonders * self.cfg.score_great_wonder
            + units_built * self.cfg.score_units_built
            + kills * self.cfg.score_units_killed
        )
        return float(score)

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
        cd = len(self.cities[player]) - len(self.cities[opp])
        if cd != 0:
            return 1 if cd > 0 else -1
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
        fatigue_me = np.zeros_like(channels[0])
        fatigue_opp = np.zeros_like(channels[0])
        city_me = np.zeros_like(channels[0])
        city_opp = np.zeros_like(channels[0])
        city_size_me = np.zeros_like(channels[0])
        city_size_opp = np.zeros_like(channels[0])
        city_walls_me = np.zeros_like(channels[0])
        city_walls_opp = np.zeros_like(channels[0])
        fatigue_turn = self.turn - 1
        for u in self.units[me]:
            if not u.alive:
                continue
            unit_me[u.y, u.x] = 1.0
            hp_me[u.y, u.x] = u.hp / 20.0
            if self.turn > 0 and u.last_move_turn == fatigue_turn:
                fatigue_me[u.y, u.x] = 1.0
        for u in self.units[opp]:
            if not u.alive:
                continue
            unit_opp[u.y, u.x] = 1.0
            hp_opp[u.y, u.x] = u.hp / 20.0
            if self.turn > 0 and u.last_move_turn == fatigue_turn:
                fatigue_opp[u.y, u.x] = 1.0
        for c in self.cities[me]:
            city_me[c.y, c.x] = 1.0
            city_size_me[c.y, c.x] = min(
                1.0, float(c.size) / max(1.0, self.cfg.city_size_norm)
            )
            if c.has_city_walls:
                city_walls_me[c.y, c.x] = 1.0
        for c in self.cities[opp]:
            city_opp[c.y, c.x] = 1.0
            city_size_opp[c.y, c.x] = min(
                1.0, float(c.size) / max(1.0, self.cfg.city_size_norm)
            )
            if c.has_city_walls:
                city_walls_opp[c.y, c.x] = 1.0
        channels.append(unit_me)
        channels.append(unit_opp)
        channels.append(hp_me)
        channels.append(hp_opp)
        channels.append(fatigue_me)
        channels.append(fatigue_opp)
        channels.append(city_me)
        channels.append(city_opp)
        channels.append(city_size_me)
        channels.append(city_size_opp)
        channels.append(city_walls_me)
        channels.append(city_walls_opp)
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
            for cidx, city in enumerate(self.cities[p]):
                parts.append(f"c{p}{cidx}:{city.x},{city.y},sz{city.size}")
            # Include research status to disambiguate states with identical unit positions.
            res_bits = ''.join('1' if self.research_done[p].get(tech, False) else '0' for tech in self.RESEARCH_TECHS)
            parts.append(f"r{p}:{res_bits}")
        if self.winner is not None:
            parts.append(f"winner={self.winner}")
        parts.append(f"kills:{self.kills.get(1,0)}/{self.kills.get(-1,0)}")
        return '|'.join(parts)
