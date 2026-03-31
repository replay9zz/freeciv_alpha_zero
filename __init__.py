"""AlphaZero-ready Freeciv adapters.

Keep package import side effects minimal so live-control utilities can import
`freeciv_alpha_zero.freeciv.*` without pulling optional training UI deps.
"""

__all__ = [
    "Arena",
    "Coach",
    "Game",
    "MCTS",
    "NeuralNet",
    "utils",
    "FreecivGame",
    "FreecivNNet",
]


def __getattr__(name):
    if name == "Arena":
        from .Arena import Arena

        return Arena
    if name == "Coach":
        from .Coach import Coach

        return Coach
    if name == "Game":
        from .Game import Game

        return Game
    if name == "MCTS":
        from .MCTS import MCTS

        return MCTS
    if name == "NeuralNet":
        from .NeuralNet import NeuralNet

        return NeuralNet
    if name == "utils":
        from . import utils

        return utils
    if name == "FreecivGame":
        from .freeciv.game import FreecivGame

        return FreecivGame
    if name == "FreecivNNet":
        from .freeciv.nnet import NNetWrapper as FreecivNNet

        return FreecivNNet
    raise AttributeError(name)
