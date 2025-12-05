from __future__ import annotations

import argparse
import time
import logging

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


def parse_args() -> argparse.Namespace:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="AlphaZero training harness for Freeciv board abstraction")
    parser.add_argument('--map-width', type=int, default=9)
    parser.add_argument('--map-height', type=int, default=9)
    parser.add_argument('--max-turns', type=int, default=64)
    parser.add_argument('--num-iters', type=int, default=2)
    parser.add_argument('--num-eps', type=int, default=10, help='Self-play episodes per iteration')
    parser.add_argument('--num-mcts-sims', type=int, default=64)
    parser.add_argument('--arena-compare', type=int, default=10)
    parser.add_argument('--checkpoint', default='temp/fcaz', help='Directory for checkpoints')
    parser.add_argument('--enemy-density', type=float, default=0.0, help='Probability of trap tiles in random maps')
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('--load-folder', default=None)
    parser.add_argument('--load-file', default=None)
    parser.add_argument('--stats-path', default=None)
    return parser.parse_args()

def format_duration(seconds: float) -> str:
    days, rem = divmod(float(seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(days):02}d {int(hours):02}:{int(minutes):02}:{secs:05.2f}"

def main():
    start_time = time.perf_counter()
    args = parse_args()
    logging.info("Starting training with args: %s", args)
    map_cfg = MapConfig(map_w=args.map_width, map_h=args.map_height, max_turns=args.max_turns)
    # Boost research rewards toward the default target tech to ensure research appears during training.
    logging.info("Building research reward map for target tech %s", TARGET_TECH_NAME)
    logging.info("Using TECH_CHILD_INHERITANCE: %s", TECH_CHILD_INHERITANCE.get(TARGET_TECH_NAME, {}))
    map_cfg.research_reward_map = reward_map_from_goals(
        {TARGET_TECH_NAME: 1.0},
        base_reward=map_cfg.research_reward,
        prereqs=TECH_PREREQS,
        child_ratios=TECH_CHILD_INHERITANCE,
    )
    logging.info("Research reward map built: %s", map_cfg.research_reward_map)
    provider = RandomMapProvider(map_cfg.map_w, map_cfg.map_h, enemy_density=args.enemy_density)
    game = FreecivGame(map_cfg, provider)
    nnet = NNetWrapper(game)

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
    })

    coach = Coach(game, nnet, coach_args)

    logging.info("Coach constructed; beginning learn loop")
    if coach_args.load_model and coach_args.load_folder_file:
        best_folder, best_file = coach_args.load_folder_file
        nnet.load_checkpoint(best_folder, best_file)
        coach.loadTrainExamples()

    coach.learn()
    elapsed = time.perf_counter() - start_time
    print(f"Total runtime: {format_duration(elapsed)}")


if __name__ == "__main__":
    main()
