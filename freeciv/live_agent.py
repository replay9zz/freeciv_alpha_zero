from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

try:
    from freeciv_rl.freeciv_luaremote import LuaRemoteClient  # type: ignore
    from freeciv_rl.freeciv_movement import FreecivMovement  # type: ignore
    from freeciv_rl.lua_helper import (  # type: ignore
        list_visible_tiles_call,
        parse_position_result,
        parse_vision_tiles,
        simple_find_unit_pos,
    )
except Exception:
    import sys

    _repo_root = Path(__file__).resolve().parents[2]
    _fallback_rl = _repo_root / "freeciv_rl"
    if _fallback_rl.exists():
        sys.path.insert(0, str(_fallback_rl))
    try:
        from freeciv_rl.freeciv_luaremote import LuaRemoteClient  # type: ignore
        from freeciv_rl.freeciv_movement import FreecivMovement  # type: ignore
        from freeciv_rl.lua_helper import (  # type: ignore
            list_visible_tiles_call,
            parse_position_result,
            parse_vision_tiles,
            simple_find_unit_pos,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The live agent requires the freeciv_rl helpers. Ensure the freeciv_rl package is available."
        ) from exc

from freeciv_alpha_zero.freeciv.config import MapConfig
from freeciv_alpha_zero.freeciv.game import CanonicalBoard, FreecivGame
from freeciv_alpha_zero.freeciv.nnet import NNetWrapper
from freeciv_alpha_zero.freeciv.providers import GroundTruth
from freeciv_alpha_zero.freeciv.state import FreecivBoardState


@dataclass
class Snapshot:
    au_map: np.ndarray
    enemy_map: np.ndarray
    visited: np.ndarray
    revealed: np.ndarray
    player_pos: Tuple[int, int]
    enemy_pos: Tuple[int, int]
    status_lookup: Dict[Tuple[int, int], Tuple[str, bool, bool, bool]]


def chunked(seq: Iterable[Tuple[int, int]], size: int) -> Iterable[List[Tuple[int, int]]]:
    bucket: List[Tuple[int, int]] = []
    for item in seq:
        bucket.append(item)
        if len(bucket) >= size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def build_state(cfg: MapConfig, snapshot: Snapshot) -> FreecivBoardState:
    state = FreecivBoardState.__new__(FreecivBoardState)  # type: ignore[misc]
    state.cfg = cfg
    state.provider = None
    state.rng = np.random.default_rng()
    state.movement = FreecivMovement(cfg.map_w, cfg.map_h)
    state.gt = GroundTruth(snapshot.au_map.copy(), snapshot.enemy_map.copy())
    state.units = {
        1: snapshot.player_pos,
        -1: snapshot.enemy_pos,
    }
    state.visited = {
        1: snapshot.visited.copy(),
        -1: np.zeros_like(snapshot.visited, dtype=bool),
    }
    state.revealed = {
        1: snapshot.revealed.copy(),
        -1: np.zeros_like(snapshot.revealed, dtype=bool),
    }
    state.turn = 0
    state.winner = None
    state.terminal_reason = None
    return state


def gather_snapshot(
    client: LuaRemoteClient,
    movement: FreecivMovement,
    cfg: MapConfig,
    unit_id: int,
    owner_id: Optional[int],
    known_tiles: Dict[Tuple[int, int], str],
    known_enemy: Dict[Tuple[int, int], bool],
    visited_tiles: Set[Tuple[int, int]],
) -> Tuple[Snapshot, Optional[int]]:
    pos_result = client.eval(simple_find_unit_pos(unit_id))
    pos_info = parse_position_result(pos_result)
    if pos_info is None:
        raise RuntimeError("Controlled unit was not found in the current Freeciv session.")

    player_pos = (pos_info[0], pos_info[1])
    if owner_id is None and pos_info[2] is not None and pos_info[2] >= 0:
        owner_id = int(pos_info[2])

    visible_tiles: Set[Tuple[int, int]] = set()
    status_lookup: Dict[Tuple[int, int], Tuple[str, bool, bool, bool]] = {}

    if owner_id is not None:
        try:
            tiles_result = client.eval(list_visible_tiles_call(owner_id, unit_id))
            visible_tiles = set(parse_vision_tiles(tiles_result))
        except Exception:
            visible_tiles = set()

    visible_tiles.add(player_pos)

    coords_to_query: Set[Tuple[int, int]] = set(visible_tiles)
    coords_to_query.update(
        coord for coord in movement.get_native_neighbors(*player_pos)
        if coord[0] is not None and coord[1] is not None
    )

    for batch in chunked(coords_to_query, 48):
        try:
            batch_status = client.neighbors_status(unit_id, batch)
        except Exception:
            continue
        for entry in batch_status:
            if len(entry) < 7:
                continue
            nx, ny, au_char, enemy_flag, _terrain, enemy_units, friendly_units = entry
            status_lookup[(nx, ny)] = (
                au_char,
                bool(enemy_flag),
                bool(enemy_units),
                bool(friendly_units),
            )

    for coord, status in status_lookup.items():
        au_char, enemy_flag, enemy_units, friendly_units = status
        if au_char:
            known_tiles[coord] = au_char
        if enemy_flag or enemy_units:
            known_enemy[coord] = True

    known_tiles[player_pos] = 'A'

    au_grid = np.full((cfg.map_h, cfg.map_w), 'U', dtype='<U1')
    enemy_grid = np.zeros((cfg.map_h, cfg.map_w), dtype=bool)
    visited_grid = np.zeros((cfg.map_h, cfg.map_w), dtype=bool)
    revealed_grid = np.zeros((cfg.map_h, cfg.map_w), dtype=bool)

    for (nx, ny), au_char in known_tiles.items():
        if 0 <= ny < cfg.map_h and 0 <= nx < cfg.map_w:
            au_grid[ny, nx] = au_char
            revealed_grid[ny, nx] = True
    for (nx, ny), enemy_flag in known_enemy.items():
        if enemy_flag and 0 <= ny < cfg.map_h and 0 <= nx < cfg.map_w:
            enemy_grid[ny, nx] = True
    for (nx, ny) in visited_tiles:
        if 0 <= ny < cfg.map_h and 0 <= nx < cfg.map_w:
            visited_grid[ny, nx] = True

    px, py = player_pos
    if 0 <= py < cfg.map_h and 0 <= px < cfg.map_w:
        au_grid[py, px] = 'A'
        revealed_grid[py, px] = True
        visited_grid[py, px] = True

    enemy_pos = player_pos
    for coord, flag in known_enemy.items():
        if flag:
            enemy_pos = coord
            break

    snapshot = Snapshot(
        au_map=au_grid,
        enemy_map=enemy_grid,
        visited=visited_grid,
        revealed=revealed_grid,
        player_pos=player_pos,
        enemy_pos=enemy_pos,
        status_lookup=status_lookup,
    )
    return snapshot, owner_id


def choose_action(
    snapshot: Snapshot,
    pi: np.ndarray,
    action_size: int,
    movement: FreecivMovement,
    visited_tiles: Set[Tuple[int, int]],
    previous_pos: Optional[Tuple[int, int]],
) -> int:
    px, py = snapshot.player_pos
    neighbors = movement.get_native_neighbors(px, py)
    mask = np.zeros(action_size, dtype=np.float32)

    for idx, (nx, ny) in enumerate(neighbors):
        if nx is None or ny is None:
            continue
        status = snapshot.status_lookup.get((nx, ny))
        if not status:
            continue
        au_char, enemy_flag, enemy_units, friendly_units = status
        if au_char == 'A' and not friendly_units and not enemy_units and not enemy_flag:
            mask[idx] = 1.0

    mask[-1] = 1.0

    neighbor_indices = list(range(action_size - 1))
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

    masked = pi * mask
    total = masked.sum()
    if not math.isfinite(total) or total <= 1e-6:
        return action_size - 1

    return int(np.argmax(masked))


def parse_dir_ids(raw: str) -> List[int]:
    parts = [s.strip() for s in raw.split(',') if s.strip()]
    if len(parts) != 6:
        raise ValueError("Expected 6 comma-separated direction ids for [N,NE,SE,S,SW,NW]")
    return [int(p) for p in parts]


def load_network(checkpoint: Path, map_cfg: MapConfig) -> NNetWrapper:
    game = FreecivGame(map_cfg)
    nnet = NNetWrapper(game)
    folder = str(checkpoint.parent) if checkpoint.parent != Path("") else "."
    try:
        nnet.load_checkpoint(folder, checkpoint.name)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load checkpoint into network with map size "
            f"{map_cfg.map_w}x{map_cfg.map_h}. Ensure --map-width/--map-height match the training "
            "configuration used when creating the checkpoint."
        ) from exc
    return nnet


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a trained Freeciv AlphaZero model via LuaRemote.")
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=4444)
    ap.add_argument('--timeout', type=float, default=2.5)
    ap.add_argument('--unit-id', type=int, required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--map-width', type=int, default=9)
    ap.add_argument('--map-height', type=int, default=9)
    ap.add_argument('--max-turns', type=int, default=64)
    ap.add_argument('--dir-ids', default='0,1,4,7,6,3')
    ap.add_argument('--sleep', type=float, default=0.1)
    ap.add_argument('--max-steps', type=int, default=200)
    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint file {checkpoint_path} not found.")

    dir_ids = parse_dir_ids(args.dir_ids)
    map_cfg = MapConfig(map_w=args.map_width, map_h=args.map_height, max_turns=args.max_turns)
    nnet = load_network(checkpoint_path, map_cfg)

    client = LuaRemoteClient(args.host, args.port, timeout=args.timeout)
    client.connect()

    owner_id: Optional[int] = None
    pos_result = client.eval(simple_find_unit_pos(args.unit_id))
    pos_info = parse_position_result(pos_result)
    if pos_info and pos_info[2] is not None and pos_info[2] >= 0:
        owner_id = int(pos_info[2])

    movement = FreecivMovement(map_width=map_cfg.map_w, map_height=map_cfg.map_h)
    known_tiles: Dict[Tuple[int, int], str] = {}
    known_enemy: Dict[Tuple[int, int], bool] = {}
    visited_tiles: Set[Tuple[int, int]] = set()
    previous_pos: Optional[Tuple[int, int]] = None

    steps = 0
    while steps < args.max_steps:
        snapshot, owner_id = gather_snapshot(
            client=client,
            movement=movement,
            cfg=map_cfg,
            unit_id=args.unit_id,
            owner_id=owner_id,
            known_tiles=known_tiles,
            known_enemy=known_enemy,
            visited_tiles=visited_tiles,
        )
        visited_tiles.add(snapshot.player_pos)

        board_state = build_state(map_cfg, snapshot)
        canonical = CanonicalBoard(board_state, 1)
        pi, _value = nnet.predict(canonical)
        action = choose_action(
            snapshot,
            pi,
            board_state.ACTION_SIZE + 1,
            movement,
            visited_tiles,
            previous_pos,
        )

        if action == board_state.PASS_ACTION:
            client.end_turn()
            print(f"[step {steps}] pass (no valid moves)")
            previous_pos = None
        else:
            dir_id = dir_ids[action]
            success = client.move_dir_id(args.unit_id, dir_id)
            client.end_turn()
            print(f"[step {steps}] move dir_id={dir_id} success={success}")
            previous_pos = snapshot.player_pos

        time.sleep(args.sleep)
        steps += 1

    print(f"Completed {steps} steps; exiting.")


if __name__ == "__main__":
    main()
