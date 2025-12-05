from __future__ import annotations

import math
from typing import List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from freeciv_rl.freeciv_movement import FreecivMovement

if TYPE_CHECKING:  # pragma: no cover - type hint convenience
    from freeciv_alpha_zero.freeciv.live_agent import Snapshot


def choose_action(
    snapshot: "Snapshot",
    pi: np.ndarray,
    action_size: int,
    movement: FreecivMovement,
    visited_tiles: Set[Tuple[int, int]],
    previous_pos: Optional[Tuple[int, int]],
    valid_actions: Optional[np.ndarray] = None,
) -> int:
    px, py = snapshot.player_pos
    neighbors = movement.get_native_neighbors(px, py)
    mask = np.ones(action_size, dtype=np.float32)

    for idx, (nx, ny) in enumerate(neighbors):
        if nx is None or ny is None:
            mask[idx] = 0.0
            continue
        status = snapshot.status_lookup.get((nx, ny))
        if not status:
            mask[idx] = 0.0
            continue
        au_char, enemy_flag, enemy_units, friendly_units = status
        # Allow stepping onto any ally or unknown tile; enemy presence is attackable, friendly stacking allowed.
        if au_char in ('A', 'U') or enemy_units or enemy_flag:
            mask[idx] = 1.0
        else:
            mask[idx] = 0.0

    neighbor_indices = list(range(len(neighbors)))
    non_pass_indices = [idx for idx in neighbor_indices if mask[idx] > 0]

    # 1) Prefer avoiding immediate backtracking when another move exists.
    if previous_pos is not None:
        backtrack_indices = [
            idx for idx, (nx, ny) in enumerate(neighbors)
            if (nx, ny) == previous_pos and mask[idx] > 0
        ]
        if len(non_pass_indices) > len(backtrack_indices):
            for idx in backtrack_indices:
                mask[idx] *= 0.2  # demote but still allow backtracking when necessary

    # 2) Prefer unvisited tiles if any remain legal.
    has_unvisited = False
    visited_indices: List[int] = []
    for idx, (nx, ny) in enumerate(neighbors):
        if mask[idx] <= 0:
            continue
        if (nx, ny) not in visited_tiles:
            has_unvisited = True
        else:
            visited_indices.append(idx)
    if has_unvisited:
        for idx in visited_indices:
            mask[idx] *= 0.5  # allow revisiting but prefer unexplored tiles

    # 3) Only prefer pass when no other moves exist.
    if any(mask[idx] > 0 for idx in neighbor_indices):
        mask[-1] *= 0.1

    if valid_actions is not None:
        mask *= valid_actions.astype(np.float32)

    masked = pi * mask
    total = masked.sum()
    if not math.isfinite(total) or total <= 1e-6:
        return action_size - 1

    return int(np.argmax(masked))


def fallback_move_direction(
    snapshot: "Snapshot",
    movement: FreecivMovement,
    previous_pos: Optional[Tuple[int, int]],
    dir_ids: List[int],
) -> Optional[int]:
    """
    Simple greedy fallback: prefer enemy tiles, then unexplored, then allied tiles,
    avoiding immediate backtracking when possible.
    """
    px, py = snapshot.player_pos
    neighbors = movement.get_native_neighbors(px, py)
    candidates: List[Tuple[float, int]] = []
    for idx, (nx, ny) in enumerate(neighbors):
        if idx >= len(dir_ids) or nx is None or ny is None:
            continue
        status = snapshot.status_lookup.get((nx, ny))
        if status is None:
            continue
        _au_char, enemy_flag, enemy_units, _friendly_units = status
        score = 0.0
        if enemy_flag or enemy_units:
            score += 2.0
        if snapshot.enemy_map[ny, nx]:
            score += 1.0
        if snapshot.visited[ny, nx]:
            score -= 0.2
        if previous_pos is not None and (nx, ny) == previous_pos:
            score -= 0.5
        candidates.append((score, idx))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda item: item[0])
    best_idx = candidates[0][1]
    return dir_ids[best_idx]
