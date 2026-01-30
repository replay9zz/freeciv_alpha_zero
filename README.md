# Freeciv Learning Environment for AlphaZero
This repo is Freeciv Learning Environment for AlphaZero based on [AlphaZero General](https://github.com/suragnair/alpha-zero-general)

## Train
Example:
```bash
python -m freeciv.train \
	--map-width 4 --map-height 16 --max-turns 300 \
	--max-actions-per-turn 50 \
	--num-iters 10 --num-eps 10 --num-mcts-sims 50 \
	--arena-compare 20 \
	--no-sea-units
```

## Test
Example:
```bash
python -m freeciv.live_agent \
    --checkpoint path/to/checkpoint/model.checkpoint \
    --mode multihead --max-units 6 \
    --map-width 4 --map-height 16 \
    --max-turns 300 --max-steps 20000 \
    --sleep 0 \
    --no-sea-units \
    --host 127.0.0.1 --port 4444 \
    --player-id 0
```