"""AlphaZero-ready Freeciv adapters."""

# AlphaZero primitives (vendored from alpha-zero-general)
from .Arena import Arena  # noqa: F401
from .Coach import Coach  # noqa: F401
from .Game import Game  # noqa: F401
from .MCTS import MCTS  # noqa: F401
from .NeuralNet import NeuralNet  # noqa: F401
from . import utils  # noqa: F401

# Freeciv-specific helpers
from .freeciv.game import FreecivGame  # re-export for convenience
from .freeciv.nnet import NNetWrapper as FreecivNNet  # noqa: F401
