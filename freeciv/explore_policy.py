from __future__ import annotations

import math
from typing import List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from freeciv_rl.freeciv_movement import FreecivMovement

if TYPE_CHECKING:  # pragma: no cover - type hint convenience
    from freeciv_alpha_zero.freeciv.live_agent import Snapshot


def best_bfs_step(
    snapshot: "Snapshot",
    movement: FreecivMovement,
    target: Tuple[int, int],
) -> Optional[int]:
    """
    Return the neighbor index (0-5) that is the first step of a shortest path to target.
    Uses only known 'A' tiles as passable.
    """
    start = snapshot.player_pos
    if start == target:
        return None
    neighbors = movement.get_native_neighbors(*start)
    seen: Set[Tuple[int, int]] = {start}
    queue: List[Tuple[int, int, int]] = []
    # Seed frontier with immediate neighbors and remember which dir they came from.
    for idx, (nx, ny) in enumerate(neighbors):
        if nx is None or ny is None:
            continue
        if snapshot.au_map[ny, nx] != 'A':
            continue
        queue.append((nx, ny, idx))
        seen.add((nx, ny))
    while queue:
        cx, cy, first_idx = queue.pop(0)
        if (cx, cy) == target:
            return first_idx
        for adj_idx, (nx, ny) in enumerate(movement.get_native_neighbors(cx, cy)):
            if nx is None or ny is None:
                continue
            if (nx, ny) in seen:
                continue
            if snapshot.au_map[ny, nx] != 'A':
                continue
            seen.add((nx, ny))
            queue.append((nx, ny, first_idx))
    return None


def choose_action(
    snapshot: "Snapshot",
    pi: np.ndarray,
    action_size: int,
    movement: FreecivMovement,
    visited_tiles: Set[Tuple[int, int]],
    previous_pos: Optional[Tuple[int, int]],
    target_coord: Optional[Tuple[int, int]] = None,
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
        au_char, enemy_flag, enemy_units, friendly_units, *_rest = status
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

    # 3) If a target is known, prefer steps that reduce distance to it.
    if target_coord is not None:
        tx, ty = target_coord
        cur_dist = abs(tx - px) + abs(ty - py)
        bfs_idx = best_bfs_step(snapshot, movement, target_coord)
        for idx, (nx, ny) in enumerate(neighbors):
            if mask[idx] <= 0:
                continue
            step_dist = abs(tx - nx) + abs(ty - ny)
            if bfs_idx is not None and idx == bfs_idx:
                mask[idx] *= 5.0  # follow BFS one-step shortest
            elif step_dist < cur_dist:
                mask[idx] *= 3.0  # strong preference for closing distance
            elif step_dist == cur_dist:
                mask[idx] *= 1.0
            else:
                mask[idx] *= 0.3  # discourage moving away

    # 4) Only prefer pass when no other moves exist.
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
    target_coord: Optional[Tuple[int, int]] = None,
) -> Optional[int]:
    """
    Simple greedy fallback: prefer enemy tiles, then unexplored, then allied tiles,
    avoiding immediate backtracking when possible.
    """
    px, py = snapshot.player_pos
    # If target known and reachable by BFS, take that step directly.
    if target_coord is not None:
        bfs_idx = best_bfs_step(snapshot, movement, target_coord)
        if bfs_idx is not None and bfs_idx < len(dir_ids):
            return dir_ids[bfs_idx]

    neighbors = movement.get_native_neighbors(px, py)
    candidates: List[Tuple[float, int]] = []
    for idx, (nx, ny) in enumerate(neighbors):
        if idx >= len(dir_ids) or nx is None or ny is None:
            continue
        status = snapshot.status_lookup.get((nx, ny))
        if status is None:
            continue
        _au_char, enemy_flag, enemy_units, _friendly_units, *_rest = status
        score = 0.0
        if enemy_flag or enemy_units:
            score += 2.0
        if snapshot.enemy_map[ny, nx]:
            score += 1.0
        if snapshot.visited[ny, nx]:
            score -= 0.2
        if previous_pos is not None and (nx, ny) == previous_pos:
            score -= 0.5
        if target_coord is not None:
            tx, ty = target_coord
            cur_dist = abs(tx - px) + abs(ty - py)
            step_dist = abs(tx - nx) + abs(ty - ny)
            score += (cur_dist - step_dist) * 0.5  # reward closing distance
        candidates.append((score, idx))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda item: item[0])
    best_idx = candidates[0][1]
    return dir_ids[best_idx]
