from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MapConfig:
    map_w: int = 9
    map_h: int = 9
    max_turns: int = 64
    fog_radius: int = 2
    frontier_bonus: float = 0.05
    visit_reward: float = 0.1
    backtrack_penalty: float = -0.02
    wall_penalty: float = -0.15
    elimination_bonus: float = 0.5
    draw_value: float = 1e-4


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
