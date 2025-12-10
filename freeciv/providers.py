from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from freeciv_rl.freeciv_luaremote import LuaRemoteClient  # type: ignore
    from freeciv_rl.lua_helper import (
        list_visible_tiles_call,
        parse_vision_tiles,
    )
except Exception:  # pragma: no cover - optional dependency
    LuaRemoteClient = None  # type: ignore

Coord = Tuple[int, int]


@dataclass
class GroundTruth:
    au_map: np.ndarray  # str array with 'A' or 'U'
    enemy_map: np.ndarray  # bool array

    def copy(self) -> "GroundTruth":
        return GroundTruth(self.au_map.copy(), self.enemy_map.copy())


class BaseProvider:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def resample(self) -> GroundTruth:
        raise NotImplementedError


class RandomMapProvider(BaseProvider):
    """Generates a new random AU/enemy map each call."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        p_open: float = 0.85,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(width, height)
        self.p_open = p_open
        self.rng = rng or np.random.default_rng()

    def resample(self) -> GroundTruth:
        au = np.where(self.rng.random((self.height, self.width)) < self.p_open, 'A', 'U')
        enemy = np.zeros((self.height, self.width), dtype=bool)
        return GroundTruth(au.astype('<U1'), enemy)


class FixedMapProvider(BaseProvider):
    def __init__(self, gt: GroundTruth):
        super().__init__(gt.au_map.shape[1], gt.au_map.shape[0])
        self._gt = gt

    def resample(self) -> GroundTruth:
        return self._gt.copy()


class LuaRemoteProvider(BaseProvider):
    """Example stub that mirrors Freeciv's LuaRemote visibility.

    The idea is to query a running Freeciv client for the tiles visible to the
    controllable unit and map them into the AU/enemy tensors used by
    FreecivBoardState. For now we only demonstrate how to pull the data; the
    caller is responsible for merging it with previously discovered tiles.
    """

    def __init__(self, width: int, height: int, *, host: str = "127.0.0.1", port: int = 4444, timeout: float = 2.5):
        if LuaRemoteClient is None:
            raise RuntimeError("freeciv_rl LuaRemote helpers are required for LuaRemoteProvider")
        super().__init__(width, height)
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client: Optional[LuaRemoteClient] = None

    def connect(self):
        if self.client is None:
            self.client = LuaRemoteClient(self.host, self.port, timeout=self.timeout)
            self.client.connect()

    def resample(self) -> GroundTruth:
        self.connect()
        raise NotImplementedError("Implement LuaRemote synchronization for your specific Freeciv run")

    def pull_visible_tiles(self, player_id: int, unit_id: int) -> List[Coord]:
        self.connect()
        assert self.client is not None
        result = self.client.eval(list_visible_tiles_call(player_id, unit_id))
        return list(parse_vision_tiles(result))
