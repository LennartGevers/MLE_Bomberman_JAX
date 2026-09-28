# Jax-Bomberman

**Authors:** Joshua Marwin Müller and Lennart Gevers, Heidelberg University

This repository contains the code for our report: a Bomberman engine written in JAX, the agents we
trained with it, and the notebooks for the experiments in chapter 6. The engine plays by the same rules as
the original `bomberman_rl` game from the course. Because the whole game is stored in arrays of fixed size
and every rule is a plain function, JAX can compile it and run thousands of games in parallel on a GPU.

## Setup

```bash
uv sync --extra notebooks        # or: pip install -e ".[notebooks]"
```

You need Python 3.12 or newer. The `notebooks` extra installs Jupyter, pandas, matplotlib, seaborn and Optuna.

## Where to start

- **`showcase.ipynb`**: our two final agents play against the rule-based agent, and you can watch one
  game of each.
- **`agent_code/`**: the two final agents, ready to play in the original game (see below).
- **`experiments/`**: the training and analysis notebooks behind every figure and table of chapter 6.
  See [`experiments/README.md`](experiments/README.md).

## Layout

```
bomberman/
  game/                  the game engine (independent of the agents and training code)
    settings.py            Scenario, GameRules, Settings
    environment.py         board, bombs, explosions, coins; world generation
    dynamics.py            one tick: actions, bombs, blasts, coins, kills, termination
    observe.py             Observation: what an agent sees (coins under crates are hidden)
    game.py                Game / GameState: init, step, observe, legal_action_mask
  agents/
    base.py                Agent interface
    features.py            ChannelStack: turns an Observation into network input
    networks/              actor-critic CNNs: KartalCNN (FC128) and SkynetCNN (Conv1x1)
    policygradient/        PolicyAgent with the REINFORCE and A2C objectives
    alphazero.py           AlphaZeroAgent: tree search with a network (mctx), plus self-play training
    scripted/              RuleBasedAgent (the original rule-based agent in JAX), coin collector, random
  training/              training loop, Rollout, reward functions, curriculum boards, logging
  rollout.py             RolloutOutput: everything recorded while playing games
  visualize.py           draw a game state as SVG and animate games in a notebook
  profiles.py            training settings for CPU, GPU and TPU
  trained.py             load the two final agents from their weights
  interop.py             turn the original game's game_state into our Observation
agent_code/
  a2c_kartal/            A2C agent: callbacks.py + weights.eqx
  mcts_kartal/           MCTS agent (16 simulations per move) with the same network
experiments/
  training/              T02–T05: train and evaluate agents, save results in runs/
  analysis/              A04–A08: make figures from runs/, save them in figures/
  runs/, figures/        our results
showcase.ipynb
```

## The engine in a few lines

```python
import jax
from bomberman.game import Game
from bomberman.training import versus_task_settings

game = Game(settings=versus_task_settings(size=17, crate_density=0.35, coins=9, max_steps=200, agents=2))
state = game.init(jax.random.key(0))
obs = game.observe(state, 0)                 # what agent 0 sees
legal = game.legal_action_mask(state, 0)     # UP, DOWN, LEFT, RIGHT, BOMB, WAIT
state = game.step(state, jax.numpy.array([5, 5]))  # play one tick, both agents wait
```

## Playing in the original game

Each folder in `agent_code/` is a normal agent for the original `bomberman_rl` game, with a
`callbacks.py` that defines `setup` and `act`. In every turn, `bomberman.interop` turns the
`game_state` of the original game into our `Observation`, and the agent picks its best action. The
original game leaves out some information our agents use, such as bomb owners and cooldowns, so
`interop` fills it in from what it has seen so far.

```bash
pip install -e /path/to/Jax-Bomberman
cp -r agent_code/a2c_kartal agent_code/mcts_kartal /path/to/bomberman_rl/agent_code/
cd /path/to/bomberman_rl
python main.py play --agents a2c_kartal rule_based_agent
```

The agent is compiled in `setup`, so compiling does not count towards the time limit per move. The MCTS
agent is slower than the A2C agent. If its moves take longer than the time limit on your machine, lower
`NUM_SIMULATIONS` in its `callbacks.py`.
