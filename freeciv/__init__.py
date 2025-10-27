"""Freeciv-specific helpers and training entry points."""

from __future__ import annotations

import sys
from pathlib import Path

try:  # pragma: no cover - only runs in local checkout contexts
    import freeciv_alpha_zero as _fcz  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _pkg_root = Path(__file__).resolve().parent
    _repo_root = _pkg_root.parent.parent
    if _repo_root.is_dir():
        sys.path.insert(0, str(_repo_root))
    try:
        import freeciv_alpha_zero as _fcz  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        pass

from .config import MapConfig, TrainingConfig  # noqa: F401
from .game import FreecivGame, CanonicalBoard  # noqa: F401
from .nnet import NNetWrapper  # noqa: F401
from .providers import BaseProvider, RandomMapProvider, GroundTruth  # noqa: F401
from .state import FreecivBoardState, Player  # noqa: F401
