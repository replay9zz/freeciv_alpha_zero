from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
AZG_PATH = REPO_ROOT / "alpha-zero-general"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(AZG_PATH) not in sys.path:
    sys.path.append(str(AZG_PATH))

from alpha_zero_general.Coach import Coach  # type: ignore
from alpha_zero_general.utils import dotdict  # type: ignore

from freeciv_alpha_zero.config import MapConfig, TrainingConfig
from freeciv_alpha_zero.freeciv_game import FreecivGame
from freeciv_alpha_zero.nnet import NNetWrapper
from freeciv_alpha_zero.providers import RandomMapProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaZero training harness for Freeciv board abstraction")
    parser.add_argument('--map-width', type=int, default=9)
    parser.add_argument('--map-height', type=int, default=9)
    parser.add_argument('--max-turns', type=int, default=64)
    parser.add_argument('--num-iters', type=int, default=2)
    parser.add_argument('--num-eps', type=int, default=10, help='Self-play episodes per iteration')
    parser.add_argument('--num-mcts-sims', type=int, default=64)
    parser.add_argument('--arena-compare', type=int, default=10)
    parser.add_argument('--checkpoint', default='temp/fcaz', help='Directory for checkpoints')
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('--load-folder', default=None)
    parser.add_argument('--load-file', default=None)
    parser.add_argument('--stats-path', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    map_cfg = MapConfig(map_w=args.map_width, map_h=args.map_height, max_turns=args.max_turns)
    provider = RandomMapProvider(map_cfg.map_w, map_cfg.map_h)
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

    if coach_args.load_model and coach_args.load_folder_file:
        best_folder, best_file = coach_args.load_folder_file
        nnet.load_checkpoint(best_folder, best_file)
        coach.loadTrainExamples()

    coach.learn()


if __name__ == "__main__":
    main()
