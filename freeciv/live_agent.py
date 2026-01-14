from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

try:
    from freeciv_rl.freeciv_luaremote import LuaRemoteClient  # type: ignore
    from freeciv_rl.freeciv_movement import FreecivMovement  # type: ignore
    from freeciv_rl.lua_helper import (  # type: ignore
        list_all_units,
        list_all_cities,
        list_city_walls,
        list_all_unit_types,
        get_player_research,
        player_knows_tech,
        set_player_research,
        list_visible_tiles_call,
        list_player_scores,
        parse_position_result,
        parse_vision_tiles,
        simple_find_unit_pos,
        auto_settler,
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
            list_city_walls,
            list_all_unit_types,
            get_player_research,
            player_knows_tech,
            set_player_research,
            list_visible_tiles_call,
            list_player_scores,
            parse_position_result,
            parse_vision_tiles,
            simple_find_unit_pos,
            auto_settler,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The live agent requires the freeciv_rl helpers. Ensure the freeciv_rl package is available."
        ) from exc

from freeciv_alpha_zero.freeciv.config import MapConfig
from freeciv_alpha_zero.freeciv.train import load_tech_unlocks, unit_value
from freeciv_alpha_zero.freeciv.game import CanonicalBoard, FreecivGame
from freeciv_alpha_zero.freeciv.multihead_game import MultiheadGame
from freeciv_alpha_zero.freeciv.multihead_state import MHUnit, MultiheadState
from freeciv_alpha_zero.freeciv.nnet import NNetWrapper
from freeciv_alpha_zero.freeciv.providers import GroundTruth
from freeciv_alpha_zero.freeciv.state import FreecivBoardState
from freeciv_alpha_zero.freeciv.research_policy import (
    TARGET_TECH_NAME,
    TECH_PREREQS,
    build_tech_costs,
    pick_next_goal_tech,
    pick_next_priority_tech,
)
from freeciv_alpha_zero.freeciv.explore_policy import (
    choose_action,
    fallback_move_direction,
)

# Units we want to avoid auto-producing even if unlocked (tempo killers).
EXCLUDED_PRODUCTION_UNITS = {"Migrants"}

def get_unit_rule_name(client: LuaRemoteClient, unit_id: int) -> Optional[str]:
    """
    Ask Lua for the rule_name of a unit by id.
    """
    lua = (
        "return (function() "
        f"local target_id={int(unit_id)}; "
        "for i=0,63 do "
        "  local pl = find.player and find.player(i); "
        "  if pl then "
        "    local iter = pl.units_iterate and pl:units_iterate() or nil; "
        "    if iter then "
        "      while true do "
        "        local u = iter(); "
        "        if not u then break end; "
        "        if u.id == target_id then "
        "          local ok, ut = pcall(function() return u:utype() end); "
        "          if not ok or not ut then ut = u.utype end; "
        "          if not ut then return '__NONE__' end; "
        "          local ok2, nm = pcall(function() return ut:rule_name() end); "
        "          if ok2 and nm then return nm end; "
        "          if ut.rule_name then return ut.rule_name end; "
        "          local ok3, tn = pcall(function() return ut:name_translation() end); "
        "          if ok3 and tn then return tn end; "
        "          return '__NONE__' "
        "        end "
        "      end "
        "    end "
        "  end "
        "end "
        "return '__NONE__' "
        "end)()"
    )
    try:
        res = client.eval(lua)
        val = res.last_return() if res else None
        if isinstance(val, str) and val != "__NONE__":
            return val
    except Exception:
        return None
    return None


def enemy_strength_map(
    client: LuaRemoteClient,
    player_id: Optional[int],
    unit_values: Dict[str, Tuple[str, float]],
) -> Dict[Tuple[int, int], float]:
    """
    Build a map of enemy unit strengths keyed by tile coordinates.
    """
    strengths: Dict[Tuple[int, int], float] = {}
    if player_id is None:
        return strengths
    try:
        units = list_all_units(client)
    except Exception:
        return strengths
    value_lookup = {name: val for name, (_tech, val) in unit_values.items()}
    for uid, ux, uy, owner in units:
        if owner == player_id:
            continue
        name = get_unit_rule_name(client, uid)
        val = value_lookup.get(name or "", 0.0)
        strengths[(ux, uy)] = val
    return strengths


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
        if name in EXCLUDED_PRODUCTION_UNITS:
            continue
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
    controlled_units: Optional[List[int]] = None,
    unit_type_labels: Optional[Dict[int, str]] = None,
) -> bool:
    """
    Set production to the best unlocked unit (prefers stronger tech-gated units).
    """
    unit_values = unit_values or {}
    target_name = pick_production_target(research_flags, unit_values)
    if unit_values:
        req = unit_values.get(target_name, ("", 0.0))[0]
        unlocked = True if not req else research_flags.get(req, False)
        print(f"[production] choose {target_name} (unlocked={unlocked})")
    else:
        print(f"[production] unit values unavailable; defaulting to {target_name}")
    return client.set_city_production(city_id, "UnitType", target_name)

def query_player_research(client: LuaRemoteClient, player_id: int) -> str:
    """
    Return '__TECH__ <RuleName>' or '__NORESEARCH__' for the player's current research target.
    Uses a direct Lua query to avoid helper return wrappers.
    """
    lua = (
        "return (function() "
        f"local pl = find.player and find.player({player_id}); "
        "if not pl then return '__NORESEARCH__' end; "
        "local tech=nil; "
        "local ok,res=pcall(function() "
        "  if pl.researching ~= nil then "
        "    if type(pl.researching)=='function' then return pl:researching() end "
        "    return pl.researching "
        "  end "
        "  return nil "
        "end); "
        "if ok and res then tech=res end; "
        "if not tech then "
        "  local ok2,res2=pcall(function() "
        "    if pl.research_goal ~= nil then "
        "      if type(pl.research_goal)=='function' then return pl:research_goal() end "
        "      return pl.research_goal "
        "    end "
        "    return nil "
        "  end); "
        "  if ok2 and res2 then tech=res2 end; "
        "end; "
        "if not tech then return '__NORESEARCH__' end; "
        "local ok3, name = pcall(function() return tech:rule_name() end); "
        "if ok3 and name and name ~= '' then return '__TECH__ '..name end; "
        "return '__NORESEARCH__' "
        "end)()"
    )
    try:
        res = client.eval(lua)
        val = res.last_return() if res else None
        return val if isinstance(val, str) else "__NORESEARCH__"
    except Exception:
        return "__NORESEARCH__"


def _resolve_score_log_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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
        tech_name = pick_next_priority_tech(flags, prereqs=TECH_PREREQS)
        if tech_name is None:
            tech_name = pick_next_goal_tech(TARGET_TECH_NAME, flags, prereqs=TECH_PREREQS)
        if tech_name is None:
            tech_name = TARGET_TECH_NAME
    try:
        ok = set_player_research(client, player_id, tech_name)
        # Log what actually got set after the request.
        actual = query_player_research(client, player_id)
        print(f"[research] request={tech_name} current={actual} ok={ok}")
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


def build_multihead_state(
    cfg: MapConfig,
    snapshot: Snapshot,
    unit_positions: List[Tuple[int, int]],
    unit_can_build_city: List[bool],
    max_units: int,
    unit_type_names: Optional[List[str]] = None,
) -> MultiheadState:
    """
    Best-effort adapter from live Snapshot -> MultiheadState for inference.
    This does not query full unit stats; it uses simple constant stats.
    """
    state = MultiheadState.__new__(MultiheadState)  # type: ignore[misc]
    state.cfg = cfg
    state.provider = None
    state.max_units = max_units
    state.rng = np.random.default_rng()
    state.movement = FreecivMovement(cfg.map_w, cfg.map_h)
    state.gt = GroundTruth(snapshot.au_map.copy(), snapshot.enemy_map.copy())
    state.units = {1: [], -1: []}
    state.cities = {1: [], -1: []}
    state.research_done = {1: {}, -1: {}}
    state.research_target = {1: None, -1: None}
    state.research_progress = {1: 0.0, -1: 0.0}
    state.tech_costs = build_tech_costs(
        TECH_PREREQS,
        style=cfg.tech_cost_style,
        base_cost=cfg.base_tech_cost,
        min_cost=cfg.min_tech_cost,
        cost_factor=getattr(cfg, "tech_cost_factor", 1.0),
    )
    state.turn = 0
    state.actions_this_turn = 0
    state.max_actions_per_turn = max(1, max_units * 2)
    state.winner = None
    state.terminal_reason = None
    state.scores = {1: 0.0, -1: 0.0}
    state.acted_unit_slots = {1: set(), -1: set()}
    state.acted_production_cities = {1: set(), -1: set()}

    state.RESEARCH_TECHS = MultiheadState.RESEARCH_TECHS
    state.MOVE_PER_UNIT = MultiheadState.MOVE_PER_UNIT
    state.ATTACK_PER_UNIT = MultiheadState.ATTACK_PER_UNIT

    # Friendly unit slots.
    for idx, ((x, y), can_build) in enumerate(
        zip(unit_positions[:max_units], unit_can_build_city[:max_units])
    ):
        label = ""
        if unit_type_names is not None and idx < len(unit_type_names):
            label = unit_type_names[idx] or ""
        unit_label = label or ("Settlers" if can_build else "Warriors")
        state.units[1].append(
            MHUnit(int(x), int(y), 10, 2, 1, 1, unit_label, True, bool(can_build), None)
        )
    while len(state.units[1]) < max_units:
        state.units[1].append(
            MHUnit(0, 0, 0, 0, 0, 1, "None", False, False, None)
        )

    # Enemy unit slots (approximate from known enemy tiles).
    enemy_coords = [(int(x), int(y)) for (y, x) in np.argwhere(snapshot.enemy_map)]
    for x, y in enemy_coords[:max_units]:
        state.units[-1].append(
            MHUnit(int(x), int(y), 10, 2, 1, 1, "Warriors", True, False, None)
        )
    while len(state.units[-1]) < max_units:
        state.units[-1].append(
            MHUnit(0, 0, 0, 0, 0, 1, "None", False, False, None)
        )

    state.research_done = {
        1: {tech: snapshot.research_flags.get(tech, False) for tech in state.RESEARCH_TECHS},
        -1: {tech: False for tech in state.RESEARCH_TECHS},
    }

    state.MOVE_SIZE = max_units * state.MOVE_PER_UNIT
    state.ATTACK_SIZE = max_units * state.ATTACK_PER_UNIT
    state.ECON_RESEARCH_OFFSET = 0
    state.ECON_BUILD_CITY_OFFSET = len(state.RESEARCH_TECHS)
    state.max_cities = 1
    state.ECON_PRODUCTION_OFFSET = state.ECON_BUILD_CITY_OFFSET + max_units
    state.PRODUCTION_ITEM_COUNT = len(MultiheadState.PRODUCTION_ITEM_NAMES)
    state.PRODUCTION_UNIT_COUNT = len(MultiheadState.PRODUCTION_UNIT_NAMES)
    state.ECON_PASS_OFFSET = (
        state.ECON_PRODUCTION_OFFSET + state.max_cities * state.PRODUCTION_ITEM_COUNT
    )
    state.ECON_SIZE = state.ECON_PASS_OFFSET + 1
    state.ACTION_SIZE = state.MOVE_SIZE + state.ATTACK_SIZE + state.ECON_SIZE
    state.PASS_ACTION = state.ACTION_SIZE - 1
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
    # Clear stale enemy info; we'll repopulate from current vision.
    known_enemy.clear()
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
            research_name = query_player_research(client, player_id)
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


def _start_client_process(client_cmd: str, luaremote_port: int) -> subprocess.Popen:
    cmd = shlex.split(client_cmd)
    if not cmd:
        raise RuntimeError("Client command is empty.")
    env = os.environ.copy()
    env.setdefault("ENABLE_LUAREMOTE", "1")
    env.setdefault("FREECIV_LUAREMOTE_PORT", str(luaremote_port))
    return subprocess.Popen(cmd, start_new_session=True, env=env)


def _stop_client_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def _connect_with_retry(client: LuaRemoteClient, timeout: float, wait: float) -> None:
    deadline = time.monotonic() + max(0.0, timeout)
    last_exc = None
    while time.monotonic() < deadline:
        try:
            client.connect()
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(max(0.0, wait))
    raise RuntimeError(
        f"Failed to connect to LuaRemote at {client.host}:{client.port}"
    ) from last_exc


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


def load_network_multihead(checkpoint: Path, map_cfg: MapConfig, max_units: int) -> tuple[MultiheadGame, NNetWrapper]:
    game = MultiheadGame(map_cfg, max_units=max_units)
    nnet = NNetWrapper(game)
    folder = str(checkpoint.parent) if checkpoint.parent != Path("") else "."
    try:
        nnet.load_checkpoint(folder, checkpoint.name)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load multihead checkpoint into network with map size "
            f"{map_cfg.map_w}x{map_cfg.map_h}."
        ) from exc
    return game, nnet


def run_multihead_agent(
    *,
    args: argparse.Namespace,
    client: LuaRemoteClient,
    game: MultiheadGame,
    nnet: NNetWrapper,
    map_cfg: MapConfig,
    dir_ids: List[int],
    tech_weights: Dict[str, float],
    unit_type_labels: Dict[int, str],
    unit_values: Dict[str, Tuple[str, float]],
    player_id: Optional[int],
    controlled_units: List[int],
    movement: FreecivMovement,
    known_tiles: Dict[Tuple[int, int], str],
    known_enemy: Dict[Tuple[int, int], bool],
    visited_tiles: Set[Tuple[int, int]],
) -> None:
    steps = 0
    turns = 0
    queued_city_production: Set[int] = set()
    last_research_flags: Dict[str, bool] = {}
    score_log_path = _resolve_score_log_path(getattr(args, "score_log", None))
    score_log_interval = int(getattr(args, "score_log_interval", 0) or 0)
    autosettler_units: Set[int] = set()

    def _maybe_log_scores(turn: int) -> None:
        if score_log_path is None or score_log_interval <= 0:
            return
        if turn <= 0 or turn % score_log_interval != 0:
            return
        scores = list_player_scores(client)
        p0 = scores.get(0, (None, None, ""))[0]
        p1 = scores.get(1, (None, None, ""))[0]
        payload = {
            "turn": turn,
            "player0_score": p0,
            "player1_score": p1,
        }
        with score_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def can_research_now(tech_name: str, flags: Dict[str, bool]) -> bool:
        """
        Allow research only if all prerequisites are already known.
        """
        reqs = TECH_PREREQS.get(tech_name, [])
        return all(flags.get(req, False) for req in reqs)

    def _unit_label(uid: int) -> str:
        label = unit_type_labels.get(uid, "") or ""
        if not label:
            try:
                label = get_unit_rule_name(client, uid) or ""
            except Exception:
                label = ""
            if label:
                unit_type_labels[uid] = label
        return label
    while steps < args.max_steps and turns < map_cfg.max_turns:
        controlled_units, player_id = discover_controlled_units(client, player_id)
        controlled_units = sorted(controlled_units)
        owned_cities = discover_player_cities(client, player_id)
        for uid in controlled_units:
            if uid in autosettler_units:
                continue
            label = _unit_label(uid).lower()
            if "worker" in label or "engineer" in label or "migrant" in label:
                try:
                    ok = auto_settler(client, uid)
                except Exception:
                    ok = False
                if ok:
                    autosettler_units.add(uid)
        if not controlled_units:
            # No active units (e.g., settler was consumed founding a city). Keep the game progressing
            # so production/research can create new units.
            if owned_cities:
                set_research_to_target(client, player_id, research_flags={})
                for cid, _cx, _cy in owned_cities:
                    if cid in queued_city_production:
                        continue
                    queued = queue_city_production(
                        client,
                        cid,
                        research_flags={},
                        unit_values=unit_values,
                        controlled_units=controlled_units,
                        unit_type_labels=unit_type_labels,
                    )
                    print(f"[production] city={cid} queued={queued}")
                    if queued:
                        queued_city_production.add(cid)
            client.end_turn()
            turns += 1
            time.sleep(args.sleep)
            continue

        # Update knowledge using the first unit as vision anchor.
        try:
            snapshot, player_id = gather_snapshot(
                client=client,
                movement=movement,
                cfg=map_cfg,
                unit_id=controlled_units[0],
                player_id=player_id,
                known_tiles=known_tiles,
                known_enemy=known_enemy,
                visited_tiles=visited_tiles,
            )
        except RuntimeError:
            # Likely the unit died; refresh unit list and continue the outer loop.
            controlled_units, player_id = discover_controlled_units(client, player_id)
            controlled_units = sorted(controlled_units)
            if not controlled_units:
                # No units left; let the outer loop handle city-only flow on next iteration.
                continue
            # Try again on the next loop iteration with the refreshed unit list.
            continue
        visited_tiles.add(snapshot.player_pos)
        if last_research_flags and snapshot.research_flags != last_research_flags:
            queued_city_production.clear()
        last_research_flags = dict(snapshot.research_flags)

        # Ensure production is queued for each owned city at least once (and after tech changes).
        for cid, _cx, _cy in owned_cities:
            if cid in queued_city_production:
                continue
            queued = queue_city_production(
                client,
                cid,
                research_flags=snapshot.research_flags,
                unit_values=unit_values,
                controlled_units=controlled_units,
                unit_type_labels=unit_type_labels,
            )
            print(f"[production] city={cid} queued={queued}")
            if queued:
                queued_city_production.add(cid)

        # Gather current positions for all controlled units.
        unit_positions: List[Tuple[int, int]] = []
        unit_type_names: List[str] = []
        unit_can_build_city: List[bool] = []
        for uid in controlled_units:
            pos_result = client.eval(simple_find_unit_pos(uid))
            pos_info = parse_position_result(pos_result)
            if pos_info is None:
                continue
            unit_positions.append((pos_info[0], pos_info[1]))
            label = _unit_label(uid)
            unit_type_names.append(label)
            unit_can_build_city.append("settler" in label.lower())

        # If an owned city is threatened (enemy within 2 tiles) and not garrisoned,
        # pull units back to that city for defense.
        city_coords = [(cx, cy) for (_cid, cx, cy) in owned_cities]
        enemy_coords = [(int(x), int(y)) for (y, x) in np.argwhere(snapshot.enemy_map)]
        threatened_city: Optional[Tuple[int, int]] = None
        garrisoned: Set[Tuple[int, int]] = set(unit_positions)
        for cx, cy in city_coords:
            has_enemy_near = any(abs(cx - ex) + abs(cy - ey) <= 2 for ex, ey in enemy_coords)
            if has_enemy_near and (cx, cy) not in garrisoned:
                threatened_city = (cx, cy)
                break
        # If any unit is already adjacent to an enemy (unit/city), force an attack before policy.
        for unit_idx, (ux, uy) in enumerate(unit_positions):
            neighbors = movement.get_native_neighbors(int(ux), int(uy))
            for dir_idx, (nx, ny) in enumerate(neighbors):
                if nx is None or ny is None:
                    continue
                status = snapshot.status_lookup.get((nx, ny))
                if not status:
                    continue
                _au_char, enemy_flag, enemy_units, friendly_units, *_rest = status
                if (enemy_flag or enemy_units or snapshot.enemy_map[ny, nx]) and not friendly_units:
                    unit_id = controlled_units[unit_idx]
                    success = client.attack_target(unit_id, int(nx), int(ny))
                    print(
                        f"[turn {turns}] forced attack unit={format_unit_label(unit_id, unit_type_labels)} "
                        f"target=({nx},{ny}) dir_idx={dir_idx} success={success}"
                    )
                    steps += 1
                    # Refresh snapshot/positions after the attack.
                    try:
                        snapshot, player_id = gather_snapshot(
                            client=client,
                            movement=movement,
                            cfg=map_cfg,
                            unit_id=controlled_units[0],
                            player_id=player_id,
                            known_tiles=known_tiles,
                            known_enemy=known_enemy,
                            visited_tiles=visited_tiles,
                        )
                    except RuntimeError:
                        controlled_units, player_id = discover_controlled_units(client, player_id)
                        controlled_units = sorted(controlled_units)
                        unit_positions = []
                        unit_type_names = []
                        unit_can_build_city = []
                        break
                    visited_tiles.add(snapshot.player_pos)
                    unit_positions = []
                    unit_type_names = []
                    unit_can_build_city = []
                    for uid in controlled_units:
                        pos_result = client.eval(simple_find_unit_pos(uid))
                        pos_info = parse_position_result(pos_result)
                        if pos_info is None:
                            continue
                        unit_positions.append((pos_info[0], pos_info[1]))
                        label = _unit_label(uid)
                        unit_type_names.append(label)
                        unit_can_build_city.append("settler" in label.lower())
                    enemy_coords = [(int(x), int(y)) for (y, x) in np.argwhere(snapshot.enemy_map)]
                    break
            else:
                continue
            break

        current_research: Optional[str] = None
        if isinstance(snapshot.research_name, str) and snapshot.research_name.startswith("__TECH__"):
            current_research = snapshot.research_name.replace("__TECH__", "", 1).strip() or None

        econ_used_this_turn = False

        # Execute up to K actions, then end turn.
        for _ in range(max(1, args.max_units * 2)):
            # Identify a target to focus movement/attacks: first known enemy tile (unit or city).
            target_coord: Optional[Tuple[int, int]] = None
            # Prefer explicit enemy tiles from vision status (covers cities even if enemy_map missed it).
            for (nx, ny), status in snapshot.status_lookup.items():
                if not status:
                    continue
                _au_char, enemy_flag, enemy_units, _friendly_units, *_rest = status
                if enemy_flag or enemy_units:
                    target_coord = (int(nx), int(ny))
                    break
            # Fallback to enemy_map if nothing in status_lookup.
            if target_coord is None:
                coords = np.argwhere(snapshot.enemy_map)
                if coords.size > 0:
                    # enemy_map uses (y, x); convert to (x, y)
                    y0, x0 = coords[0]
                    target_coord = (int(x0), int(y0))
            # If a city is threatened, override with defense target.
            if threatened_city is not None:
                target_coord = threatened_city

            board_state = build_multihead_state(
                map_cfg,
                snapshot,
                unit_positions,
                unit_can_build_city=unit_can_build_city,
                unit_type_names=unit_type_names,
                max_units=args.max_units,
            )
            canonical = game.getCanonicalForm(board_state, 1)
            pi, _v = nnet.predict(canonical)
            valids = board_state.valid_moves(1)

            econ_offset = board_state.MOVE_SIZE + board_state.ATTACK_SIZE
            research_slice = slice(econ_offset, econ_offset + board_state.ECON_BUILD_CITY_OFFSET)
            build_city_slice = slice(
                econ_offset + board_state.ECON_BUILD_CITY_OFFSET,
                econ_offset + board_state.ECON_PASS_OFFSET,
            )

            # Do not spam research when no city exists (but allow build-city).
            if not owned_cities:
                valids[research_slice] = 0

            # Mask out techs whose prereqs are not yet known.
            for idx, tech_name in enumerate(board_state.RESEARCH_TECHS):
                if not can_research_now(tech_name, snapshot.research_flags):
                    aidx = econ_offset + idx
                    if 0 <= aidx < len(valids):
                        valids[aidx] = 0

            # Do not switch research mid-progress: if already researching something, disable all research changes.
            if current_research:
                valids[research_slice] = 0

            # Avoid wasting actions: if already researching the selected tech, treat it as invalid.
            if current_research:
                for idx, tech_name in enumerate(board_state.RESEARCH_TECHS):
                    if tech_name == current_research:
                        aidx = econ_offset + idx
                        if 0 <= aidx < len(valids):
                            valids[aidx] = 0

            # Limit econ actions to at most one per Freeciv turn (except PASS).
            if econ_used_this_turn:
                valids[econ_offset : board_state.PASS_ACTION] = 0

            # Optional tech weights.
            if tech_weights:
                for idx, tech_name in enumerate(board_state.RESEARCH_TECHS):
                    w = tech_weights.get(tech_name)
                    if w is None:
                        continue
                    aidx = econ_offset + idx
                    if 0 <= aidx < len(pi):
                        pi[aidx] *= w

            # If a target is known, bias moves/attacks toward it to reduce circling/retreating.
            if target_coord is not None:
                tx, ty = target_coord
                for unit_idx, (ux, uy) in enumerate(unit_positions):
                    # If sitting on the threatened city with no adjacent enemy, hold position (defend).
                    if target_coord == threatened_city and (ux, uy) == target_coord:
                        move_start = unit_idx * board_state.MOVE_PER_UNIT
                        move_end = move_start + board_state.MOVE_PER_UNIT
                        pi[move_start:move_end] *= 0.0
                        continue
                    neighbors = movement.get_native_neighbors(int(ux), int(uy))
                    best_dir = None
                    best_dist = float("inf")
                    for dir_idx, (nx, ny) in enumerate(neighbors):
                        if nx is None or ny is None:
                            continue
                        dist = abs(tx - nx) + abs(ty - ny)
                        if dist < best_dist:
                            best_dist = dist
                            best_dir = dir_idx
                    if best_dir is not None:
                        move_idx = unit_idx * board_state.MOVE_PER_UNIT + best_dir
                        if 0 <= move_idx < len(pi):
                            pi[move_idx] *= 3.0
                        # If adjacent, also boost the corresponding attack action.
                        if best_dist == 0:
                            atk_idx = board_state.MOVE_SIZE + unit_idx * board_state.ATTACK_PER_UNIT + best_dir
                            if 0 <= atk_idx < len(pi):
                                pi[atk_idx] *= 5.0

            def choose_with_priority() -> int:
                # Option A (turn-aware):
                # - If no city exists, prioritize BUILD_CITY (if available) to bootstrap the economy.
                # - If no research is set yet and econ hasn't been used this turn, choose an econ action once.
                # - Otherwise: attack > move > econ(pass).
                if not owned_cities:
                    m_build = pi[build_city_slice] * valids[build_city_slice]
                    if m_build.sum() > 0:
                        return int(build_city_slice.start + np.argmax(m_build))

                if owned_cities and not econ_used_this_turn and not current_research:
                    m_econ = pi[research_slice] * valids[research_slice]
                    if m_econ.sum() > 0:
                        return int(research_slice.start + np.argmax(m_econ))

                slices = [
                    (board_state.MOVE_SIZE, board_state.MOVE_SIZE + board_state.ATTACK_SIZE),
                    (0, board_state.MOVE_SIZE),
                    (econ_offset, board_state.ACTION_SIZE),
                ]
                for start, end in slices:
                    m = pi[start:end] * valids[start:end]
                    if m.sum() > 0:
                        return int(start + np.argmax(m))
                return int(np.argmax(valids))

            action = choose_with_priority()

            if action == board_state.PASS_ACTION:
                break

            # Research action.
            if action >= board_state.MOVE_SIZE + board_state.ATTACK_SIZE:
                rel = action - econ_offset
                # research
                if 0 <= rel < len(board_state.RESEARCH_TECHS):
                    tech_name = board_state.RESEARCH_TECHS[rel]
                    if not can_research_now(tech_name, snapshot.research_flags):
                        print(f"[turn {turns}] research={tech_name} blocked (prereqs unmet)")
                        break
                    success = set_research_to_target(
                        client,
                        player_id,
                        research_flags=snapshot.research_flags,
                        tech_name=tech_name,
                    )
                    actual = query_player_research(client, player_id)
                    print(f"[turn {turns}] research={tech_name} ok={success} current={actual}")
                    if not isinstance(actual, str) or tech_name not in (actual or ""):
                        # If it didn't stick, treat as failed and skip further econ this turn.
                        econ_used_this_turn = True
                        break
                    steps += 1
                    econ_used_this_turn = True
                # build city (per unit slot)
                elif board_state.ECON_BUILD_CITY_OFFSET <= rel < board_state.ECON_PASS_OFFSET:
                    unit_idx = rel - board_state.ECON_BUILD_CITY_OFFSET
                    if unit_idx < len(controlled_units):
                        unit_id = controlled_units[unit_idx]
                        unit_desc = format_unit_label(unit_id, unit_type_labels)
                        city_name = f"AutoCity{len(owned_cities) + 1}"
                        built = client.found_city(unit_id, city_name)
                        action_desc = f"founded city '{city_name}'"
                        if not built:
                            built = client.build_city(unit_id)
                            action_desc = "built a city"
                        print(f"[turn {turns}] unit={unit_desc} {action_desc} success={built}")
                        steps += 1
                        econ_used_this_turn = True
                        if built:
                            # Settler is typically consumed; refresh controllable units.
                            controlled_units, player_id = discover_controlled_units(client, player_id)
                            controlled_units = sorted(controlled_units)
                else:
                    break

                # Refresh snapshot after econ actions as well.
                if not controlled_units:
                    break
                snapshot, player_id = gather_snapshot(
                    client=client,
                    movement=movement,
                    cfg=map_cfg,
                    unit_id=controlled_units[0],
                    player_id=player_id,
                    known_tiles=known_tiles,
                    known_enemy=known_enemy,
                    visited_tiles=visited_tiles,
                )
                visited_tiles.add(snapshot.player_pos)
                owned_cities = discover_player_cities(client, player_id)
                if isinstance(snapshot.research_name, str) and snapshot.research_name.startswith("__TECH__"):
                    current_research = snapshot.research_name.replace("__TECH__", "", 1).strip() or None
                unit_positions = []
                unit_type_names = []
                unit_can_build_city = []
                for uid in controlled_units:
                    pos_result = client.eval(simple_find_unit_pos(uid))
                    pos_info = parse_position_result(pos_result)
                    if pos_info is None:
                        continue
                    unit_positions.append((pos_info[0], pos_info[1]))
                    label = _unit_label(uid)
                    unit_type_names.append(label)
                    unit_can_build_city.append("settler" in label.lower())
                time.sleep(args.sleep)
                continue

            # Move/attack action.
            is_attack = action >= board_state.MOVE_SIZE
            if is_attack:
                rel = action - board_state.MOVE_SIZE
                unit_idx = rel // board_state.ATTACK_PER_UNIT
                dir_idx = rel % board_state.ATTACK_PER_UNIT
            else:
                unit_idx = action // board_state.MOVE_PER_UNIT
                dir_idx = action % board_state.MOVE_PER_UNIT

            if unit_idx >= len(controlled_units):
                break
            unit_id = controlled_units[unit_idx]
            unit_desc = format_unit_label(unit_id, unit_type_labels)
            if is_attack:
                if unit_idx < len(unit_positions):
                    ux, uy = unit_positions[unit_idx]
                    neighbors = movement.get_native_neighbors(int(ux), int(uy))
                    tx, ty = neighbors[dir_idx]
                else:
                    tx = ty = None
                if tx is None or ty is None:
                    success = False
                else:
                    success = client.attack_target(unit_id, int(tx), int(ty))
                print(f"[turn {turns}] attack unit={unit_desc} dir_idx={dir_idx} target=({tx},{ty}) success={success}")
                steps += 1
            else:
                dir_id = dir_ids[dir_idx]
                success = client.move_dir_id(unit_id, dir_id)
                print(f"[turn {turns}] move unit={unit_desc} dir_id={dir_id} success={success}")
                steps += 1

            # Refresh snapshot after acting.
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
            except RuntimeError:
                # Unit likely died; refresh and break out of action loop to start a new turn.
                controlled_units, player_id = discover_controlled_units(client, player_id)
                controlled_units = sorted(controlled_units)
                break
            visited_tiles.add(snapshot.player_pos)
            owned_cities = discover_player_cities(client, player_id)
            if isinstance(snapshot.research_name, str) and snapshot.research_name.startswith("__TECH__"):
                current_research = snapshot.research_name.replace("__TECH__", "", 1).strip() or None
            unit_positions = []
            unit_type_names = []
            unit_can_build_city = []
            for uid in controlled_units:
                pos_result = client.eval(simple_find_unit_pos(uid))
                pos_info = parse_position_result(pos_result)
                if pos_info is None:
                    continue
                unit_positions.append((pos_info[0], pos_info[1]))
                label = _unit_label(uid)
                unit_type_names.append(label)
                unit_can_build_city.append("settler" in label.lower())
            time.sleep(args.sleep)

        client.end_turn()
        turns += 1
        _maybe_log_scores(turns)

    print(f"Completed {steps} steps across {turns} turns; exiting.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a trained Freeciv AlphaZero model via LuaRemote.")
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=4444)
    ap.add_argument('--timeout', type=float, default=2.5)
    ap.add_argument('--unit-id', type=int, help='Control a single unit id (disables auto-discovery).')
    ap.add_argument('--player-id', type=int, help='Restrict auto-discovery to a specific player id.')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--mode', choices=['default', 'multihead'], default='default')
    ap.add_argument('--max-units', type=int, default=6, help='Max unit slots for multihead mode')
    ap.add_argument('--map-width', type=int, default=9)
    ap.add_argument('--map-height', type=int, default=9)
    ap.add_argument('--max-turns', type=int, default=64)
    # Default hex dir ids align with the mapping used in freeciv_rl.run_model_agent:
    # [N, NE, SE, S, SW, NW] -> [0, 1, 4, 7, 6, 3]
    ap.add_argument('--dir-ids', default='0,1,4,7,6,3')
    ap.add_argument('--sleep', type=float, default=0.1)
    ap.add_argument('--max-steps', type=int, default=800)
    ap.add_argument('--score-log', default=None, help='Write civ scores to JSONL every N turns.')
    ap.add_argument('--score-log-interval', type=int, default=25)
    ap.add_argument(
        '--client-cmd',
        help="Optional command to launch a Freeciv client (headless or GUI).",
    )
    ap.add_argument(
        '--no-client',
        action='store_true',
        help='Do not auto-start a client even if FREECIV_CLIENT_CMD is set.',
    )
    ap.add_argument(
        '--client-start-wait',
        type=float,
        default=1.0,
        help='Seconds between LuaRemote connection attempts when auto-starting a client.',
    )
    ap.add_argument(
        '--client-start-timeout',
        type=float,
        default=30.0,
        help='Max seconds to wait for LuaRemote when auto-starting a client.',
    )
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
    game: MultiheadGame | None = None
    if args.mode == "multihead":
        game, nnet = load_network_multihead(checkpoint_path, map_cfg, max_units=args.max_units)
    else:
        nnet = load_network(checkpoint_path, map_cfg)

    client_cmd = None
    if not args.no_client:
        client_cmd = args.client_cmd or os.getenv("FREECIV_CLIENT_CMD")

    client = None
    client_process = None
    try:
        if client_cmd:
            client_process = _start_client_process(client_cmd, args.port)

        client = LuaRemoteClient(args.host, args.port, timeout=args.timeout)
        if client_cmd:
            _connect_with_retry(client, args.client_start_timeout, args.client_start_wait)
        else:
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
        last_research_flags: Dict[str, bool] = {}
        known_enemy_target: Optional[Tuple[int, int]] = None

        # Preload unlock values for production selection
        unlock_path = Path(__file__).resolve().parent / "data" / "tech_unlocks.yaml"
        try:
            unit_values = load_unit_values(unlock_path)
            print(f"[config] loaded {len(unit_values)} unit values from {unlock_path}")
        except Exception as exc:
            unit_values = {}
            print(f"[config] failed to load unit values from {unlock_path}: {exc}")
        unit_strengths = {name: val for name, (_tech, val) in unit_values.items()}

        steps = 0
        turns = 0

        if args.mode == "multihead":
            assert game is not None
            run_multihead_agent(
                args=args,
                client=client,
                game=game,
                nnet=nnet,
                map_cfg=map_cfg,
                dir_ids=dir_ids,
                tech_weights=tech_weights,
                unit_type_labels=unit_type_labels,
                unit_values=unit_values,
                player_id=player_id,
                controlled_units=controlled_units,
                movement=movement,
                known_tiles=known_tiles,
                known_enemy=known_enemy,
                visited_tiles=visited_tiles,
            )
            return

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
                        print(f"[units] refreshed and found {len(refreshed_units)} unit(s)")
                # Refresh city info before deciding to exit.
                owned_cities = discover_player_cities(client, player_id)
                if not controlled_units:
                    if owned_cities:
                        # Ensure research is explicitly set even when no units are active.
                        set_research_to_target(client, player_id, research_flags=last_research_flags or {})
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
                # Precompute enemy strengths for this turn to guide attack decisions.
                enemy_strengths = enemy_strength_map(client, player_id, unit_values)
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
                    if last_research_flags and snapshot.research_flags != last_research_flags:
                        queued_city_production.clear()
                        print(f"[research-status] flags changed; will requeue production for all cities")
                    last_research_flags = snapshot.research_flags
                    print(f"[research-status] current={snapshot.research_name} flags={snapshot.research_flags}")
                except RuntimeError as exc:
                    print(f"[step {steps}] unit={unit_desc} unavailable: {exc}")
                    controlled_units.remove(unit_id)
                    previous_pos.pop(unit_id, None)
                    continue
                visited_tiles.add(snapshot.player_pos)
                # Track first seen enemy location (city or unit); cities do not move.
                if known_enemy_target is None:
                    for (nx, ny), status in snapshot.status_lookup.items():
                        if not status:
                            continue
                        _au_char, enemy_flag, enemy_units, _friendly_units, *_rest = status
                        if enemy_flag or enemy_units or snapshot.enemy_map[ny, nx]:
                            known_enemy_target = (nx, ny)
                            print(f"[target] locked enemy location at ({nx},{ny})")
                            break

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
                    # Ensure research is explicitly set (override client auto if needed).
                    if owned_cities:
                        set_research_to_target(client, player_id, research_flags=snapshot.research_flags)
                    # Queue production whenever needed (initially or after tech changes).
                    for cid, _cx, _cy in owned_cities:
                        if cid in queued_city_production:
                            continue
                        queued = queue_city_production(
                            client,
                            cid,
                            snapshot.research_flags,
                            unit_values,
                            controlled_units=controlled_units,
                            unit_type_labels=unit_type_labels,
                        )
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
                    _au_char, enemy_flag, enemy_units, friendly_units, *_rest = status
                    # Enemy present (unit or city) and no friendly stack blocking.
                    if (enemy_flag or enemy_units) and not friendly_units:
                        enemy_targets.append((idx, nx, ny))

                # Force an attack if any adjacent enemy is present to concentrate fire.
                if enemy_targets:
                    idx, nx, ny = enemy_targets[0]
                    enemy_val = enemy_strengths.get((nx, ny), 0.0)
                    self_val = unit_strengths.get(unit_type_labels.get(unit_id, ""), 0.0)
                    # Cities usually aren't in the strength map; treat enemy_val<=0 as city/unknown and attack.
                    if enemy_val <= 0 or self_val >= enemy_val * 0.9:
                        success = client.attack_target(unit_id, nx, ny)
                        print(
                            f"[step {steps}] unit={unit_desc} attack target=({nx},{ny}) dir_idx={idx} "
                            f"strength_self={self_val:.2f} enemy={enemy_val:.2f} success={success}"
                        )
                    else:
                        success = False
                        print(
                            f"[step {steps}] unit={unit_desc} skip attack; enemy stronger "
                            f"(self={self_val:.2f} enemy={enemy_val:.2f})"
                        )
                    print(
                        f"[step {steps}] unit={unit_desc} attack target=({nx},{ny}) dir_idx={idx} success={success}"
                    )
                    previous_pos[unit_id] = snapshot.player_pos
                    time.sleep(args.sleep)
                    steps += 1
                    acted_this_turn = True
                    continue

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
                    target_coord=known_enemy_target,
                    valid_actions=valid_actions,
                )

                if action == board_state.PASS_ACTION:
                    fallback_dir = fallback_move_direction(
                        snapshot=snapshot,
                        movement=movement,
                        previous_pos=previous_pos.get(unit_id),
                        dir_ids=dir_ids,
                        target_coord=known_enemy_target,
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
                        success = set_research_to_target(
                            client,
                            player_id,
                            research_flags=snapshot.research_flags,
                            tech_name=tech_name,
                        )
                        print(
                            f"[step {steps}] unit={unit_desc} set research tech={tech_name} success={success}"
                        )
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
            if owned_cities:
                for cid, _cx, _cy in owned_cities:
                    if cid in queued_city_production:
                        continue
                    queued = queue_city_production(
                        client,
                        cid,
                        research_flags=last_research_flags,
                        unit_values=unit_values,
                        controlled_units=controlled_units,
                        unit_type_labels=unit_type_labels,
                    )
                    print(f"[turn {turns}] queued production in city {cid} success={queued}")
                    if queued:
                        queued_city_production.add(cid)

        if not controlled_units:
            print(f"No active units remain after {steps} steps ({turns} turns); exiting.")
        else:
            print(f"Completed {steps} steps across {turns} turns; exiting.")
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        if client_process is not None:
            _stop_client_process(client_process)


if __name__ == "__main__":
    main()
