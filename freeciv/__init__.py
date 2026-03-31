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

__all__ = [
    "MapConfig",
    "TrainingConfig",
    "FreecivGame",
    "CanonicalBoard",
    "NNetWrapper",
    "BaseProvider",
    "RandomMapProvider",
    "GroundTruth",
    "FreecivBoardState",
    "Player",
]


def __getattr__(name):
    if name in {"MapConfig", "TrainingConfig"}:
        from .config import MapConfig, TrainingConfig

        return {"MapConfig": MapConfig, "TrainingConfig": TrainingConfig}[name]
    if name in {"FreecivGame", "CanonicalBoard"}:
        from .game import CanonicalBoard, FreecivGame

        return {"FreecivGame": FreecivGame, "CanonicalBoard": CanonicalBoard}[name]
    if name == "NNetWrapper":
        from .nnet import NNetWrapper

        return NNetWrapper
    if name in {"BaseProvider", "RandomMapProvider", "GroundTruth"}:
        from .providers import BaseProvider, GroundTruth, RandomMapProvider

        return {
            "BaseProvider": BaseProvider,
            "RandomMapProvider": RandomMapProvider,
            "GroundTruth": GroundTruth,
        }[name]
    if name in {"FreecivBoardState", "Player"}:
        from .state import FreecivBoardState, Player

        return {"FreecivBoardState": FreecivBoardState, "Player": Player}[name]
    raise AttributeError(name)
