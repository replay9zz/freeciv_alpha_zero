# Freeciv AlphaZero Bridge

`freeciv_alpha_zero/` exposes a minimal AlphaZero-style pipeline on top of
Freeciv-inspired hex exploration so we can experiment with MCTS + self-play.
The goal is to keep the environment simple enough to train quickly while
retaining hooks into the real Freeciv LuaRemote runner for future expansion.

## Directory layout
```
freeciv_alpha_zero/
  Arena.py / Coach.py / MCTS.py / utils.py  # vendored alpha-zero-general core
  freeciv/
    config.py        # Env/training hyper parameters (map size, rewards)
    game.py          # AlphaZero Game implementation (two-player explore/fight)
    nnet.py          # NeuralNet wrapper that Coach expects
    providers.py     # Map providers (random generator + LuaRemote bridge)
    state.py         # Deterministic board representation & helpers
    train.py         # Self-play + training CLI
    live_agent.py    # Run a trained agent against a live Freeciv client
  pytorch/NNet.py    # Actual PyTorch module (policy + value heads)
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
python -m freeciv.train --num-iters 5 --num-eps 50 --num-mcts-sims 128 --enemy-density 0.0
```
The module wires `FreecivGame`, `Coach`, and `NNetWrapper` together using the
vendored AlphaZero utilities. See `freeciv/train.py` for the full list of CLI
flags (checkpoint paths, map dimensions, resume options, etc.).

## Running against a live Freeciv client
`freeciv/live_agent.py` connects to a Freeciv GTK client with LuaRemote enabled
and lets a trained checkpoint drive a specific unit. Pass the same map window
size that the model was trained on (the default training scripts use 9x9):

```
python -m freeciv.live_agent \
  --unit-id 42 \
  --checkpoint checkpoints/best.pth.tar \
  --map-width 9 --map-height 9 \
  --host 127.0.0.1 --port 4444
```

Keep a client open with `ENABLE_LUAREMOTE=1`, ensure the Lua helper scripts from
`freeciv/lua/` are loaded, and pass the unit id you want to control (see the
client's Lua console or use `freeciv_rl/run_model_agent.py` to list units). The
agent mirrors the player's local vision to create an observation for the neural
network and issues moves via `client.move_dir`.

## Next steps
- Improve the live agent to blend AlphaZero policy with LuaRemote combat orders
  (auto-attack, defensive moves, etc.).
- Extend `FreecivBoardState` with multiple units, combat, and production queues
  so the state/action space is closer to full Freeciv.
- Add visualization tooling or logging hooks for TensorBoard / replay export.
