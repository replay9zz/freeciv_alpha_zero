from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

try:
    from freeciv_rl.freeciv_luaremote import LuaRemoteClient  # type: ignore
    from freeciv_rl.freeciv_movement import FreecivMovement  # type: ignore
    from freeciv_rl.lua_helper import (  # type: ignore
        list_all_units,
        list_all_cities,
        list_all_unit_types,
        get_player_research,
        player_knows_tech,
        set_player_research,
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
            list_all_units,
            list_all_cities,
            list_all_unit_types,
            get_player_research,
            player_knows_tech,
            set_player_research,
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
from freeciv_alpha_zero.freeciv.train import load_tech_unlocks, unit_value
from freeciv_alpha_zero.freeciv.game import CanonicalBoard, FreecivGame
from freeciv_alpha_zero.freeciv.nnet import NNetWrapper
from freeciv_alpha_zero.freeciv.providers import GroundTruth
from freeciv_alpha_zero.freeciv.state import FreecivBoardState
from freeciv_alpha_zero.freeciv.research_policy import TARGET_TECH_NAME
from freeciv_alpha_zero.freeciv.explore_policy import (
    choose_action,
    fallback_move_direction,
)


def format_unit_label(unit_id: int, unit_types: Dict[int, str]) -> str:
    label = unit_types.get(unit_id)
    if label:
        if label.lower() == "settlers":
            label = "Settler"
        return f"{unit_id}({label})"
    return str(unit_id)


@dataclass
class Snapshot:
    au_map: np.ndarray
    enemy_map: np.ndarray
    visited: np.ndarray
    revealed: np.ndarray
    player_pos: Tuple[int, int]
    enemy_pos: Tuple[int, int]
    # (A/U, enemy_flag, enemy_units, friendly_units, has_walls?)
    status_lookup: Dict[Tuple[int, int], Tuple[str, bool, bool, bool, bool]]
    research_name: Optional[str] = None
    research_done: bool = False
    research_flags: Dict[str, bool] = field(default_factory=dict)


def chunked(seq: Iterable[Tuple[int, int]], size: int) -> Iterable[List[Tuple[int, int]]]:
    bucket: List[Tuple[int, int]] = []
    for item in seq:
        bucket.append(item)
        if len(bucket) >= size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def simple_knows_tech(client: LuaRemoteClient, player_id: int, tech_name: str) -> bool:
    """
    Minimal knows_tech check by name to avoid helper mis-detections.
    """
    lua = (
        "return (function() "
        f"local pl = find.player and find.player({player_id}); "
        f"local t = find.tech_type and find.tech_type('{tech_name}'); "
        "if pl and t and pl.knows_tech and pl:knows_tech(t) then return '__YES__' end; "
        "return '__NO__' "
        "end)()"
    )
    try:
        res = client.eval(lua)
        ret = res.last_return() if res else None
        return isinstance(ret, str) and '__YES__' in ret
    except Exception:
        return False


def discover_controlled_units(
    client: LuaRemoteClient,
    player_hint: Optional[int],
) -> Tuple[List[int], Optional[int]]:
    try:
        units = list_all_units(client)
    except Exception as exc:  # pragma: no cover - remote failure
        raise RuntimeError("Failed to enumerate units via LuaRemote.") from exc

    player_id = player_hint
    if player_id is None:
        for _uid, _x, _y, unit_owner in units:
            if unit_owner >= 0:
                player_id = unit_owner
                break

    if player_id is None:
        return [], None

    controlled = [uid for uid, _x, _y, unit_owner in units if unit_owner == player_id]
    return controlled, player_id


def discover_player_cities(
    client: LuaRemoteClient,
    player_id: Optional[int],
) -> List[Tuple[int, int, int]]:
    if player_id is None:
        return []
    try:
        cities = list_all_cities(client)
    except Exception:
        return []
    owned: List[Tuple[int, int, int]] = []
    for cid, cx, cy, owner, _name in cities:
        if owner == player_id:
            owned.append((cid, cx, cy))
    return owned


def load_unit_values(path: str) -> Dict[str, Tuple[str, float]]:
    """
    Load unit values from tech unlocks file.
    Returns mapping unit_name -> (required_tech or None, value_score)
    """
    unlocks = load_tech_unlocks(path)
    out: Dict[str, Tuple[str, float]] = {}
    for item in unlocks:
        tech = item.get("tech")
        for ent in item.get("unlocks", []):
            if ent.get("kind") != "unit":
                continue
            name = ent.get("name")
            if not name:
                continue
            val = unit_value(ent)
            out[name] = (tech, val)
    return out


def pick_production_target(research_flags: Dict[str, bool], unit_values: Dict[str, Tuple[str, float]]) -> str:
    """
    Choose the highest-value unlocked unit given known techs.
    Falls back to Warriors if nothing else is unlocked.
    """
    best_name = "Warriors"
    best_val = -1.0
    for name, (req_tech, val) in unit_values.items():
        if req_tech and not research_flags.get(req_tech, False):
            continue
        if val > best_val:
            best_name = name
            best_val = val
    return best_name


def queue_city_production(
    client: LuaRemoteClient,
    city_id: int,
    research_flags: Dict[str, bool],
    unit_values: Optional[Dict[str, Tuple[str, float]]] = None,
) -> bool:
    """
    Set production to the best unlocked unit (prefers stronger tech-gated units).
    """
    if not unit_values:
        target_name = pick_production_target(research_flags, {})
        print(f"[production] unit values unavailable; defaulting to {target_name}")
    else:
        target_name = pick_production_target(research_flags, unit_values)
        print(f"[production] choose {target_name} (unlocked={research_flags.get(unit_values.get(target_name, ('',0))[0], True)})")
    return client.set_city_production(city_id, "UnitType", target_name)


def set_research_to_target(
    client: LuaRemoteClient,
    player_id: Optional[int],
    research_flags: Optional[Dict[str, bool]] = None,
    tech_name: Optional[str] = None,
) -> bool:
    if player_id is None:
        return False
    # Pick a tech considering prereqs for the main goal.
    if tech_name is None:
        flags = research_flags or {}
        if not flags.get(TARGET_TECH_NAME, False):
            # Prioritize Warrior Code -> Bronze Working -> Iron Working.
            if not flags.get("Warrior Code", False):
                tech_name = "Warrior Code"
            elif not flags.get("Bronze Working", False):
                tech_name = "Bronze Working"
            else:
                tech_name = TARGET_TECH_NAME
        else:
            tech_name = TARGET_TECH_NAME
    try:
        ok = set_player_research(client, player_id, tech_name)
        # Log what actually got set after the request.
        actual = None
        try:
            actual = get_player_research(client, player_id)
        except Exception:
            actual = None
        print(f"[research] request={tech_name} actual={actual} success={ok}")
        return ok
    except Exception:
        return False


def is_target_researched(
    client: LuaRemoteClient,
    player_id: Optional[int],
    tech_name: str = TARGET_TECH_NAME,
) -> bool:
    """
    Check completion by invoking knows_tech and logging the result.
    """
    if player_id is None:
        return False
    try:
        return player_knows_tech(client, player_id, tech_name)
    except Exception:
        return False


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
    state.scores = {1: 0.0, -1: 0.0}
    state.prev_positions = {1: None, -1: None}
    state.cities = {1: None, -1: None}
    state.research_done = {
        1: {tech: snapshot.research_flags.get(tech, False) for tech in FreecivBoardState.RESEARCH_TECHS},
        -1: {tech: False for tech in FreecivBoardState.RESEARCH_TECHS},
    }
    state.research_complete = {
        1: state.research_done[1].get(state.TARGET_TECH_NAME, snapshot.research_done),
        -1: False,
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
    player_id: Optional[int],
    known_tiles: Dict[Tuple[int, int], str],
    known_enemy: Dict[Tuple[int, int], bool],
    visited_tiles: Set[Tuple[int, int]],
) -> Tuple[Snapshot, Optional[int]]:
    pos_result = client.eval(simple_find_unit_pos(unit_id))
    pos_info = parse_position_result(pos_result)
    if pos_info is None:
        raise RuntimeError("Controlled unit was not found in the current Freeciv session.")

    player_pos = (pos_info[0], pos_info[1])
    if player_id is None and pos_info[2] is not None and pos_info[2] >= 0:
        player_id = int(pos_info[2])
    research_name: Optional[str] = None
    research_done = False
    research_flags: Dict[str, bool] = {tech: False for tech in FreecivBoardState.RESEARCH_TECHS}
    if player_id is not None:
        try:
            # Use a direct call to avoid mis-detection
            research_name = client.eval(
                "return (function() "
                f"local pl = find.player and find.player({player_id}); "
                "if not pl or not pl.researching then return '__NORESEARCH__' end; "
                "local ok, tech = pcall(function() return pl:researching() end); "
                "if not ok or not tech then return '__NORESEARCH__' end; "
                "local ok2, name = pcall(function() return tech:rule_name() end); "
                "if ok2 and name and name ~= '' then return '__TECH__ '..name end; "
                "return '__NORESEARCH__' "
                "end)()"
            ).last_return()
        except Exception:
            research_name = None
        try:
            research_done = is_target_researched(client, player_id)
        except Exception:
            research_done = False
        for tech in FreecivBoardState.RESEARCH_TECHS:
            try:
                research_flags[tech] = simple_knows_tech(client, player_id, tech)
            except Exception:
                continue
        research_done = research_flags.get(TARGET_TECH_NAME, research_done)

    visible_tiles: Set[Tuple[int, int]] = set()
    status_lookup: Dict[Tuple[int, int], Tuple[str, bool, bool, bool, bool]] = {}

    if player_id is not None:
        try:
            tiles_result = client.eval(list_visible_tiles_call(player_id, unit_id))
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
            nx, ny, au_char, enemy_flag, _terrain, enemy_units, friendly_units = entry[:7]
            has_walls = False
            if len(entry) > 7:
                has_walls = bool(entry[7])
            status_lookup[(nx, ny)] = (
                au_char,
                bool(enemy_flag),
                bool(enemy_units),
                bool(friendly_units),
                has_walls,
            )

    for coord, status in status_lookup.items():
        if not status:
            continue
        # status may include has_walls at the end; we only care about the first four here.
        au_char, enemy_flag, enemy_units, friendly_units = status[:4]
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
        research_name=research_name,
        research_done=research_done,
        research_flags=research_flags,
    )
    return snapshot, player_id


def parse_dir_ids(raw: str) -> List[int]:
    parts = [s.strip() for s in raw.split(',') if s.strip()]
    if len(parts) != 6:
        raise ValueError("Expected 6 comma-separated direction ids for [N,NE,SE,S,SW,NW]")
    return [int(p) for p in parts]


def parse_tech_weights(raw_values: List[str]) -> Dict[str, float]:
    """
    Parse tech weight strings like ["Iron Working=2.0", "Masonry=0.5,Alphabet=1.2"].
    """
    weights: Dict[str, float] = {}
    for entry in raw_values:
        for chunk in entry.split(','):
            if not chunk.strip():
                continue
            if '=' not in chunk:
                print(f"[tech-weight] skipping invalid entry (expected name=weight): '{chunk}'")
                continue
            name, sval = chunk.split('=', 1)
            try:
                weights[name.strip()] = float(sval.strip())
            except ValueError:
                print(f"[tech-weight] skipping non-numeric weight for '{name}': '{sval}'")
    return weights


def apply_tech_weights(pi: np.ndarray, board_state: FreecivBoardState, weights: Dict[str, float]) -> np.ndarray:
    """
    Apply optional multiplicative weights to research actions in the policy vector.
    """
    if not weights:
        return pi
    weighted = pi.copy()
    base = board_state.RESEARCH_ACTION_BASE
    for idx, tech_name in enumerate(board_state.RESEARCH_TECHS):
        weight = weights.get(tech_name)
        if weight is None:
            continue
        action_idx = base + idx
        if action_idx < len(weighted):
            weighted[action_idx] *= weight
    return weighted


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
    ap.add_argument('--unit-id', type=int, help='Control a single unit id (disables auto-discovery).')
    ap.add_argument('--player-id', type=int, help='Restrict auto-discovery to a specific player id.')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--map-width', type=int, default=9)
    ap.add_argument('--map-height', type=int, default=9)
    ap.add_argument('--max-turns', type=int, default=64)
    # Default hex dir ids align with the mapping used in freeciv_rl.run_model_agent:
    # [N, NE, SE, S, SW, NW] -> [0, 1, 4, 7, 6, 3]
    ap.add_argument('--dir-ids', default='0,1,4,7,6,3')
    ap.add_argument('--sleep', type=float, default=0.1)
    ap.add_argument('--max-steps', type=int, default=200)
    ap.add_argument(
        '--tech-weight',
        action='append',
        default=[],
        help="Apply weight multipliers to tech research actions (format: Name=weight,Name2=weight2). "
             "May be specified multiple times.",
    )
    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint file {checkpoint_path} not found.")

    dir_ids = parse_dir_ids(args.dir_ids)
    tech_weights = parse_tech_weights(args.tech_weight)
    map_cfg = MapConfig(map_w=args.map_width, map_h=args.map_height, max_turns=args.max_turns)
    nnet = load_network(checkpoint_path, map_cfg)

    client = LuaRemoteClient(args.host, args.port, timeout=args.timeout)
    client.connect()

    unit_type_labels: Dict[int, str] = {}
    try:
        unit_type_labels = list_all_unit_types(client)
    except Exception:
        unit_type_labels = {}

    player_id: Optional[int] = args.player_id
    controlled_units: List[int]
    if args.unit_id is not None:
        controlled_units = [args.unit_id]
        pos_result = client.eval(simple_find_unit_pos(args.unit_id))
        pos_info = parse_position_result(pos_result)
        if pos_info and pos_info[2] is not None and pos_info[2] >= 0:
            player_id = int(pos_info[2])
    else:
        controlled_units, player_id = discover_controlled_units(client, player_id)
        if not controlled_units:
            raise SystemExit(
                "No controllable units were discovered. Provide --unit-id or --player-id to limit the search."
            )
        print(
            f"Discovered {len(controlled_units)} unit(s) for player {player_id if player_id is not None else 'unknown'}."
        )

    movement = FreecivMovement(map_width=map_cfg.map_w, map_height=map_cfg.map_h)
    known_tiles: Dict[Tuple[int, int], str] = {}
    known_enemy: Dict[Tuple[int, int], bool] = {}
    visited_tiles: Set[Tuple[int, int]] = set()
    previous_pos: Dict[int, Optional[Tuple[int, int]]] = {uid: None for uid in controlled_units}
    owned_cities = discover_player_cities(client, player_id)
    queued_city_production: Set[int] = set()
    research_confirmed_complete = False
    last_research_flags: Dict[str, bool] = {}

    # Preload unlock values for production selection
    unlock_path = Path(__file__).resolve().parent / "data" / "tech_unlocks.yaml"
    try:
        unit_values = load_unit_values(unlock_path)
        print(f"[config] loaded {len(unit_values)} unit values from {unlock_path}")
    except Exception as exc:
        unit_values = {}
        print(f"[config] failed to load unit values from {unlock_path}: {exc}")

    steps = 0
    turns = 0
    while steps < args.max_steps:
        try:
            unit_type_labels = list_all_unit_types(client)
        except Exception:
            pass
        if not controlled_units:
            if player_id is not None:
                refreshed_units, player_id = discover_controlled_units(client, player_id)
                if refreshed_units:
                    controlled_units.extend(refreshed_units)
                    for uid in refreshed_units:
                        previous_pos[uid] = None
            if not controlled_units:
                if owned_cities:
                    # No active units; just end turn and let existing research/production progress.
                    client.end_turn()
                    turns += 1
                    steps += 1
                    owned_cities = discover_player_cities(client, player_id)
                    time.sleep(args.sleep)
                    continue
                break
        acted_this_turn = False
        for unit_id in list(controlled_units):
            if steps >= args.max_steps:
                break
            unit_desc = format_unit_label(unit_id, unit_type_labels)
            label_lower = unit_type_labels.get(unit_id, "").lower()
            is_settler = "settler" in label_lower
            try:
                snapshot, player_id = gather_snapshot(
                    client=client,
                    movement=movement,
                    cfg=map_cfg,
                    unit_id=unit_id,
                    player_id=player_id,
                    known_tiles=known_tiles,
                    known_enemy=known_enemy,
                    visited_tiles=visited_tiles,
                )
                last_research_flags = snapshot.research_flags
                print(f"[research-status] current={snapshot.research_name} flags={snapshot.research_flags}")
            except RuntimeError as exc:
                print(f"[step {steps}] unit={unit_desc} unavailable: {exc}")
                controlled_units.remove(unit_id)
                previous_pos.pop(unit_id, None)
                continue
            visited_tiles.add(snapshot.player_pos)

            # Found a city immediately if none exists.
            if not owned_cities and is_settler:
                city_name = f"AutoCity{len(owned_cities) + 1}"
                built = client.found_city(unit_id, city_name)
                action_desc = f"founded city '{city_name}'"
                if not built:
                    built = client.build_city(unit_id)
                    action_desc = "built a city"
                if built:
                    print(f"[step {steps}] unit={unit_desc} {action_desc}")
                    owned_cities = discover_player_cities(client, player_id)
                    controlled_units.remove(unit_id)
                    previous_pos.pop(unit_id, None)
                    time.sleep(args.sleep)
                    steps += 1
                    acted_this_turn = True
                    continue
            else:
                # Research completion status based on snapshot flags only.
                research_confirmed_complete = bool(snapshot.research_done)

                # Queue unit once research is done.
                if research_confirmed_complete:
                    for cid, _cx, _cy in owned_cities:
                        if cid in queued_city_production:
                            continue
                        queued = queue_city_production(client, cid, snapshot.research_flags, unit_values)
                        print(f"[step {steps}] queued production in city {cid} success={queued}")
                        if queued:
                            queued_city_production.add(cid)

            # Attempt to attack immediately if an enemy unit is adjacent.
            px, py = snapshot.player_pos
            enemy_targets: List[Tuple[int, int, int]] = []
            for idx, (nx, ny) in enumerate(movement.get_native_neighbors(px, py)):
                if nx is None or ny is None:
                    continue
                status = snapshot.status_lookup.get((nx, ny))
                if not status:
                    continue
                _au_char, _enemy_flag, enemy_units, friendly_units, *_rest = status
                if enemy_units and not friendly_units:
                    enemy_targets.append((idx, nx, ny))

            board_state = build_state(map_cfg, snapshot)
            canonical = CanonicalBoard(board_state, 1)
            pi, _value = nnet.predict(canonical)
            pi = apply_tech_weights(pi, board_state, tech_weights)
            valid_actions = board_state.valid_moves(1)
            action = choose_action(
                snapshot,
                pi,
                board_state.ACTION_SIZE + 1,
                movement,
                visited_tiles,
                previous_pos.get(unit_id),
                valid_actions=valid_actions,
            )

            if action == board_state.PASS_ACTION:
                fallback_dir = fallback_move_direction(
                    snapshot=snapshot,
                    movement=movement,
                    previous_pos=previous_pos.get(unit_id),
                    dir_ids=dir_ids,
                )
                if fallback_dir is not None:
                    success = client.move_dir_id(unit_id, fallback_dir)
                    print(
                        f"[step {steps}] unit={unit_desc} fallback move dir_id={fallback_dir} success={success}"
                    )
                    if success:
                        previous_pos[unit_id] = snapshot.player_pos
                    else:
                        previous_pos[unit_id] = None
                else:
                    print(f"[step {steps}] unit={unit_desc} pass (no valid moves)")
                    previous_pos[unit_id] = None
            elif action == board_state.BUILD_CITY_ACTION and is_settler:
                city_name = f"AutoCity{len(owned_cities) + 1}"
                built = client.found_city(unit_id, city_name)
                action_desc = f"founded city '{city_name}'"
                if not built:
                    built = client.build_city(unit_id)
                    action_desc = "built a city"
                print(f"[step {steps}] unit={unit_desc} {action_desc} success={built}")
                if built:
                    owned_cities = discover_player_cities(client, player_id)
                    controlled_units.remove(unit_id)
                    previous_pos.pop(unit_id, None)
            elif action == board_state.BUILD_CITY_ACTION and not is_settler:
                # Treat bad build choice as a move attempt to avoid idling.
                fallback_dir = fallback_move_direction(
                    snapshot=snapshot,
                    movement=movement,
                    previous_pos=previous_pos.get(unit_id),
                    dir_ids=dir_ids,
                )
                if fallback_dir is not None:
                    success = client.move_dir_id(unit_id, fallback_dir)
                    print(
                        f"[step {steps}] unit={unit_desc} fallback-from-build move dir_id={fallback_dir} success={success}"
                    )
                    if success:
                        previous_pos[unit_id] = snapshot.player_pos
                    else:
                        previous_pos[unit_id] = None
                else:
                    print(f"[step {steps}] unit={unit_desc} skip build (not a settler)")
                    previous_pos[unit_id] = None
            elif board_state.RESEARCH_ACTION_BASE <= action < board_state.RESEARCH_ACTION_BASE + board_state.RESEARCH_ACTION_COUNT:
                tech_idx = action - board_state.RESEARCH_ACTION_BASE
                tech_name = board_state.RESEARCH_TECHS[tech_idx]
                if not owned_cities:
                    print(
                        f"[step {steps}] unit={unit_desc} requested research {tech_name} but no city exists; skipping"
                    )
                    success = False
                else:
                    success = set_research_to_target(client, player_id, research_flags=snapshot.research_flags, tech_name=tech_name)
                    print(
                        f"[step {steps}] unit={unit_desc} set research tech={tech_name} success={success}"
                    )
                    if tech_name == TARGET_TECH_NAME:
                        research_confirmed_complete = is_target_researched(client, player_id)
                previous_pos[unit_id] = None
            elif 0 <= action < len(dir_ids):
                dir_id = dir_ids[action]
                success = client.move_dir_id(unit_id, dir_id)
                print(f"[step {steps}] unit={unit_desc} move dir_id={dir_id} success={success}")
                previous_pos[unit_id] = snapshot.player_pos
            else:
                print(f"[step {steps}] unit={unit_desc} unsupported action={action}; skipping")
                previous_pos[unit_id] = None
            time.sleep(args.sleep)
            steps += 1
            acted_this_turn = True

        client.end_turn()
        turns += 1
        if not acted_this_turn:
            time.sleep(args.sleep)

        if player_id is not None:
            latest_units, player_id = discover_controlled_units(client, player_id)
            for uid in latest_units:
                if uid not in controlled_units:
                    controlled_units.append(uid)
                    previous_pos[uid] = None
            for uid in list(controlled_units):
                if uid not in latest_units:
                    controlled_units.remove(uid)
                    previous_pos.pop(uid, None)
        owned_cities = discover_player_cities(client, player_id)
        if research_confirmed_complete and owned_cities:
            for cid, _cx, _cy in owned_cities:
                if cid in queued_city_production:
                    continue
                queued = queue_city_production(client, cid, research_flags=last_research_flags, unit_values=unit_values)
                print(f"[turn {turns}] queued production in city {cid} success={queued}")
                if queued:
                    queued_city_production.add(cid)

    if not controlled_units:
        print(f"No active units remain after {steps} steps ({turns} turns); exiting.")
    else:
        print(f"Completed {steps} steps across {turns} turns; exiting.")


if __name__ == "__main__":
    main()
