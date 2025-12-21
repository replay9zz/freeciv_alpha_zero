from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MapConfig:
    # x-axis, horizonatal, not GUI number
    map_w: int = 4
    # y-axis, vertical, not GUI number 
    map_h: int = 16
    max_turns: int = 120
    fog_radius: int = 2
    frontier_bonus: float = 0.12
    visit_reward: float = 0.25
    backtrack_penalty: float = -0.08
    wall_penalty: float = -0.15
    elimination_bonus: float = 3.0
    draw_value: float = -0.05
    build_city_reward: float = 0.15
    research_reward: float = 0.02
    # Optional per-tech rewards; falls back to research_reward when missing.
    research_reward_map: Dict[str, float] = field(default_factory=dict)
    move_reward: float = 0.02
    # City economy (minimal ruleset approximation)
    city_food: int = 2
    city_shield: int = 1
    city_trade: int = 2
    grass_food: int = 2
    grass_shield: int = 1
    grass_trade: int = 1
    food_consumption: int = 2
    food_growth: int = 20
    city_unit_cap: int = 6
    city_size_norm: int = 10


@dataclass
class TrainingConfig:
    episodes: int = 100
    num_iters: int = 5
    num_eps: int = 25
    temp_threshold: int = 15
    update_threshold: float = 0.6
    maxlen_of_queue: int = 200000
    num_mcts_sims: int = 100
    arena_compare: int = 40
    cpuct: float = 1.0
    checkpoint: str = "temp/fcaz/"
    load_model: bool = False
    load_folder_file: tuple[str, str] | None = None
