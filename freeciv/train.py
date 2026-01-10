from __future__ import annotations

import argparse
import datetime
import time
import logging
from pathlib import Path

try:
    from freeciv_alpha_zero.Coach import Coach  # type: ignore
    from freeciv_alpha_zero.utils import dotdict  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from ..Coach import Coach
    from ..utils import dotdict

from .config import MapConfig, TrainingConfig
from .game import FreecivGame
from .nnet import NNetWrapper
from .providers import RandomMapProvider
from .research_policy import (
    TARGET_TECH_NAME,
    TECH_CHILD_INHERITANCE,
    TECH_PREREQS,
    reward_map_from_goals,
)
from .ruleset_loader import load_civ2civ3_unlocks
from .combat_game import CombatGame

log = logging.getLogger(__name__)

# Optional per-unit multipliers to adjust unlock-derived values.
DEFAULT_UNIT_VALUE_MULTIPLIERS: dict[str, float] = {
    # Situational units: down-weight to avoid overvaluing rare production.
    "Migrants": 0.5,
    "Diplomat": 0.5,
}
# Optional per-building multipliers to adjust unlock-derived values.
DEFAULT_BUILDING_VALUE_MULTIPLIERS: dict[str, float] = {}


def load_unit_value_multipliers(path: Path) -> dict[str, float]:
    """
    Load per-unit multipliers from YAML (name -> multiplier). Missing/invalid
    files are ignored so that defaults still apply.
    """
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        log.warning("PyYAML missing; skipping unit multipliers: %s", exc)
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        log.warning("Unit multiplier file must be a mapping; got %r", type(data))
        return {}

    multipliers: dict[str, float] = {}
    for name, val in data.items():
        try:
            multipliers[str(name)] = float(val)
        except Exception:
            log.warning("Skipping invalid multiplier for %s: %r", name, val)
    return multipliers


# Merge defaults with file-based overrides (if present).
UNIT_VALUE_MULTIPLIERS = dict(DEFAULT_UNIT_VALUE_MULTIPLIERS)
_mult_path = Path(__file__).resolve().parent / "data" / "unit_value_multipliers.yaml"
UNIT_VALUE_MULTIPLIERS.update(load_unit_value_multipliers(_mult_path))


def load_building_value_multipliers(path: Path) -> dict[str, float]:
    """
    Load per-building multipliers from YAML (name -> multiplier). Missing/invalid
    files are ignored so that defaults still apply.
    """
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        log.warning("PyYAML missing; skipping building multipliers: %s", exc)
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        log.warning("Building multiplier file must be a mapping; got %r", type(data))
        return {}

    multipliers: dict[str, float] = {}
    for name, val in data.items():
        try:
            multipliers[str(name)] = float(val)
        except Exception:
            log.warning("Skipping invalid multiplier for %s: %r", name, val)
    return multipliers


# Merge defaults with file-based overrides (if present).
BUILDING_VALUE_MULTIPLIERS = dict(DEFAULT_BUILDING_VALUE_MULTIPLIERS)
_bmult_path = Path(__file__).resolve().parent / "data" / "building_value_multipliers.yaml"
BUILDING_VALUE_MULTIPLIERS.update(load_building_value_multipliers(_bmult_path))

# ----------------------------------------------------------------------
# Tech unlock loading
# ----------------------------------------------------------------------
def load_tech_unlocks(path: str) -> list[dict]:
    try:
        return load_civ2civ3_unlocks()
    except Exception:
        pass
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("PyYAML is required to load tech_unlocks.yaml") from exc

    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"tech unlocks file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError("tech unlocks file must be a list of tech entries")
    return data


def unit_value(entry: dict) -> float:
    """Rough value score for a unit, with optional per-unit multipliers."""
    atk = float(entry.get("attack", 0))
    df = float(entry.get("defense", 0))
    hp = float(entry.get("hp", 1))
    cost = max(float(entry.get("cost", 1)), 1.0)
    mult = UNIT_VALUE_MULTIPLIERS.get(str(entry.get("name", "")), 1.0)
    return mult * (atk + df) * hp / cost


def building_value(entry: dict) -> float:
    """Flat value for a building; cheap heuristic with per-building multipliers."""
    base = 0.5
    cost = max(float(entry.get("cost", 1)), 1.0)
    mult = BUILDING_VALUE_MULTIPLIERS.get(str(entry.get("name", "")), 1.0)
    return mult * base * (cost / 30.0)


def reward_map_from_unlocks(unlocks: list[dict], base_reward: float) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in unlocks:
        tech = item.get("tech")
        if not tech:
            continue
        entries = item.get("unlocks", [])
        val = 0.0
        for ent in entries:
            kind = ent.get("kind")
            if kind == "unit":
                val += unit_value(ent)
            elif kind == "building":
                val += building_value(ent)
        # scale by base_reward to keep magnitude similar
        values[tech] = base_reward * (1.0 + val)
    return values


def parse_args() -> argparse.Namespace:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="AlphaZero training harness for Freeciv board abstraction")
    parser.add_argument('--map-width', type=int, default=9)
    parser.add_argument('--map-height', type=int, default=9)
    parser.add_argument('--max-turns', type=int, default=64, help='Max steps per episode (alias: --max-moves)')
    parser.add_argument('--max-steps', type=int, default=None, help='Alias for --max-turns')
    parser.add_argument('--max-moves', type=int, default=None, help='Alias for --max-turns')
    parser.add_argument('--num-iters', type=int, default=2)
    parser.add_argument('--num-eps', type=int, default=10, help='Self-play episodes per iteration')
    parser.add_argument('--num-mcts-sims', type=int, default=64)
    parser.add_argument('--arena-compare', type=int, default=10)
    parser.add_argument('--checkpoint', default=None, help='Directory for checkpoints (default: results/freeciv_alpha_zero/DATE)')
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('--load-folder', default=None)
    parser.add_argument('--load-file', default=None)
    parser.add_argument('--stats-path', default=None)
    parser.add_argument('--mode', choices=['default', 'combat', 'multihead'], default='multihead', help='Training environment')
    parser.add_argument('--max-units', type=int, default=6, help='Max units per side for multihead mode')
    return parser.parse_args()


def _default_checkpoint_dir() -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    stamp = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    return base_dir / "results" / "freeciv_alpha_zero" / stamp

def format_duration(seconds: float) -> str:
    days, rem = divmod(float(seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(days):02}d {int(hours):02}:{int(minutes):02}:{secs:05.2f}"

def main():
    start_time = time.perf_counter()
    args = parse_args()
    if args.max_steps is not None:
        args.max_turns = args.max_steps
    if args.max_moves is not None:
        args.max_turns = args.max_moves
    if not args.checkpoint:
        args.checkpoint = str(_default_checkpoint_dir())
    args.tensorboard_dir = args.checkpoint
    logging.info("Starting training with args: %s", args)
    map_cfg = MapConfig(map_w=args.map_width, map_h=args.map_height, max_turns=args.max_turns)
    # Build research rewards from civ2civ3 ruleset unlock values.
    try:
        unlock_path = Path(__file__).resolve().parent / "data" / "tech_unlocks.yaml"
        unlocks = load_tech_unlocks(unlock_path)
        unlock_reward_map = reward_map_from_unlocks(unlocks, base_reward=map_cfg.research_reward)
        map_cfg.research_reward_map = unlock_reward_map
        logging.info("Loaded tech unlocks from civ2civ3 ruleset")
        logging.info("Research reward map (unlock-derived): %s", map_cfg.research_reward_map)
    except Exception as exc:
        logging.warning("Failed to load tech unlocks; falling back to goal propagation: %s", exc)
        logging.info("Building research reward map for target tech %s", TARGET_TECH_NAME)
        logging.info("Using TECH_CHILD_INHERITANCE: %s", TECH_CHILD_INHERITANCE.get(TARGET_TECH_NAME, {}))
        map_cfg.research_reward_map = reward_map_from_goals(
            {TARGET_TECH_NAME: 1.0},
            base_reward=map_cfg.research_reward,
            prereqs=TECH_PREREQS,
            child_ratios=TECH_CHILD_INHERITANCE,
        )
        logging.info("Research reward map built: %s", map_cfg.research_reward_map)
    provider = RandomMapProvider(map_cfg.map_w, map_cfg.map_h, p_open=1.0)
    if args.mode == 'combat':
        game = CombatGame(map_cfg, provider)
        logging.info("Using combat training mode (multi-unit attack-focused).")
    elif args.mode == 'multihead':
        from .multihead_game import MultiheadGame  # local import to avoid circulars when unused
        game = MultiheadGame(map_cfg, provider, max_units=args.max_units)
        logging.info("Using multihead training mode (move/attack + research, multiple units).")
    else:
        game = FreecivGame(map_cfg, provider)
        logging.info("Using default training mode.")
    nnet = NNetWrapper(game, log_dir=args.tensorboard_dir)

    train_cfg = TrainingConfig()
    coach_args = dotdict({
        'numIters': args.num_iters,
        'numEps': args.num_eps,
        'tempThreshold': train_cfg.temp_threshold,
        'updateThreshold': train_cfg.update_threshold,
        'maxlenOfQueue': train_cfg.maxlen_of_queue,
        'numMCTSSims': args.num_mcts_sims,
        'arenaCompare': args.arena_compare,
        'cpuct': train_cfg.cpuct,
        'checkpoint': args.checkpoint,
        'load_model': args.load_model,
        'load_folder_file': (args.load_folder, args.load_file) if args.load_folder and args.load_file else None,
        'numItersForTrainExamplesHistory': 4,
        'stats_path': args.stats_path,
        'tensorboard_dir': args.tensorboard_dir,
    })

    coach = Coach(game, nnet, coach_args)

    logging.info("Coach constructed; beginning learn loop")
    if coach_args.load_model and coach_args.load_folder_file:
        best_folder, best_file = coach_args.load_folder_file
        nnet.load_checkpoint(best_folder, best_file)
        coach.loadTrainExamples()

    coach.learn()
    nnet.save_checkpoint(folder=args.checkpoint, filename="model.checkpoint")
    elapsed = time.perf_counter() - start_time
    print(f"Total runtime: {format_duration(elapsed)}")


if __name__ == "__main__":
    main()
