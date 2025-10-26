# Freeciv AlphaZero Bridge

`freeciv_alpha_zero/` exposes a minimal AlphaZero-style pipeline on top of
Freeciv-inspired hex exploration so we can experiment with MCTS + self-play.
The goal is to keep the environment simple enough to train quickly while
retaining hooks into the real Freeciv LuaRemote runner for future expansion.

## Directory layout
```
freeciv_alpha_zero/
  config.py           # Env/training hyper parameters (map size, rewards)
  providers.py        # Map providers (random generator or LuaRemote stubs)
  state.py            # Deterministic board representation & helpers
  freeciv_game.py     # AlphaZero Game implementation (two-player explore/fight)
  nnet.py             # NeuralNet wrapper that Coach expects
  pytorch/NNet.py     # Actual PyTorch module (policy + value heads)
  train.py            # CLI that wires FreecivGame + Coach + NNet
```

## High level idea
- Each player controls one scout on a shared hex grid. Tiles can be
  passable (`A`) or blocked (`U`); some tiles hide enemies (negative reward).
- Players alternate turns. On their turn they move to one of the six
  neighboring hexes or choose `pass`. Moving into unexplored territory reveals
  nearby tiles (frontier bonus) similar to `freeciv_rl`.
- The provider decides the ground-truth map. By default we use
  `RandomMapProvider`, but the `LuaRemoteProvider` stub shows how to stream real
  visibility data from a live Freeciv game.
- Rewards are translated into AlphaZero’s terminal values by comparing
  territory/control when the episode ends (max turns, elimination, etc.).

## Training
```
python freeciv_alpha_zero/train.py --episodes 20 --map-width 9 --map-height 9
```
This CLI wraps `alpha-zero-general/Coach.py`; see the file for all options and
how to point it at an existing experiment directory. Because `alpha-zero-general`
expects `Game`/`NeuralNet` implementations inside `sys.path`, the script
auto-appends both the current repo root and `alpha-zero-general/`.

## Next steps
- Flesh out `LuaRemoteProvider` to mirror the actual Freeciv game state and
  allow self-play against the in-game AI or scripted opponents.
- Extend `FreecivBoardState` with multiple units, combat, and production queues
  so the state/action space is closer to full Freeciv.
- Add visualization tooling or logging hooks for TensorBoard / replay export.
