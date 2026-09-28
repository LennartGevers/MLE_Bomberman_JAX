"""AlphaZero for Bomberman, based on the AlphaZero example of pgx and the MCTS of ``mctx``."""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax
from jaxtyping import Array, PRNGKeyArray

from bomberman.agents.base import Agent, Batch
from bomberman.agents.features import ChannelStack, FeatureShaping
from bomberman.agents.networks import Network
from bomberman.game import ACTION, FIELD, EnvironmentState, Game, GameState, Observation
from bomberman.training.base import RewardFn
from bomberman.training.checkpoint import Checkpointer, NoCheckpointer
from bomberman.training.logger import Logger, NoLogger
from bomberman.training.signals import coins_collected

N_ACTIONS = len(ACTION)

# Logit given to illegal actions. We avoid `-inf` because it produces `nan`
# gradients in the softmax.
ILLEGAL_LOGIT = -1e9


@dataclasses.dataclass(frozen=True)
class Config:
    """Settings for self-play and training (pgx's ``Config`` without the search settings)."""

    num_envs: int = 16  # pgx: selfplay_batch_size
    rounds: int | None = None  # pgx: max_num_steps; None means rules.max_steps
    batch_size: int = 1024  # pgx: training_batch_size
    learning_rate: float = 1e-3
    max_grad_norm: float = 0.5
    value_coef: float = 1.0
    # For this many ticks, self-play samples moves from the search result; after that
    # it plays the best move (as in AlphaGo Zero). None means max_steps // 5.
    explore_moves: int | None = None

    def __post_init__(self):
        if self.num_envs < 1 or self.batch_size < 1:
            raise ValueError("num_envs and batch_size must be at least 1")
        if self.rounds is not None and self.rounds < 1:
            raise ValueError("rounds must be at least 1")


def hidden_coins(board: EnvironmentState) -> Array:
    """Boolean mask of the coins nobody can see: not collected and still under a crate."""
    return board.coins.hidden(board.field_map)


def resample_hidden_coins(
    board: EnvironmentState, key: PRNGKeyArray
) -> EnvironmentState:
    """Put every hidden coin under a randomly chosen crate."""
    coins, field_map = board.coins, board.field_map
    hidden = hidden_coins(board)
    num_coins = hidden.shape[0]

    crate = (field_map == FIELD.CRATE).ravel()
    rank = jnp.argsort(jnp.where(crate, jax.random.uniform(key, crate.shape), jnp.inf))
    drawn = jnp.stack(
        jnp.unravel_index(rank[:num_coins], field_map.shape), axis=-1
    ).astype(jnp.int32)

    # The i-th hidden coin goes under the i-th chosen crate; `cumsum` numbers them.
    slot = jnp.clip(jnp.cumsum(hidden) - 1, 0, max(num_coins - 1, 0))
    pos = jnp.where(hidden[:, None], drawn[slot], coins.pos)
    return dataclasses.replace(board, coins=dataclasses.replace(coins, pos=pos))


def determinize(observation: Observation, key: PRNGKeyArray) -> GameState:
    """A full game state that matches ``observation``, with a random turn order and random hidden coins."""
    coin_key, order_key, chance_key = jax.random.split(key, 3)
    return GameState(
        board=resample_hidden_coins(observation.board, coin_key),
        turn_order=jax.random.permutation(
            order_key, jnp.arange(observation.num_agents)
        ),
        bomb_cooldown=observation.bomb_cooldown,
        alive=observation.alive,
        score=observation.score,
        _key=chance_key,
        step_count=observation.step_count,
    )


class AlphaZeroAgent(Agent[None]):
    """An agent that picks moves by searching with a network and the game engine."""

    network: Network
    game: Game
    reward_fn: RewardFn
    features: FeatureShaping = ChannelStack()
    num_simulations: int = eqx.field(static=True, default=32)
    determinizations: int = eqx.field(static=True, default=1)
    discount: float = eqx.field(static=True, default=0.99)
    opponent_temperature: float = eqx.field(static=True, default=1.0)

    def __check_init__(self):
        if not self.network.has_value_head:
            raise TypeError(
                f"AlphaZero needs a value estimate, but {type(self.network).__name__} "
                "has no critic head -- use e.g. CNNActorCriticNetwork."
            )
        if self.num_simulations < 0:
            raise ValueError("num_simulations must be >= 0")
        if self.determinizations < 1:
            raise ValueError("determinizations must be at least 1")
        if self.opponent_temperature <= 0.0:
            raise ValueError("opponent_temperature must be > 0")

    @property
    def num_agents(self) -> int:
        return self.game.settings.scenario.agent_count

    def forward(self, obs: Observation, key: PRNGKeyArray, *, inference: bool = True):
        """Run the network on one observation and return ``(logits, value)``."""
        return self.network(
            self.features(obs).astype(jnp.float32), key, inference=inference
        )

    def act(self, obs: Observation, legal_mask: Array, key: PRNGKeyArray, state=None):
        """Search, then sample a move from the search result."""
        search_key, sample_key = jax.random.split(key)
        weights, value = _search_one(self, obs, search_key)
        logits = _policy_logits(weights, legal_mask)
        return (
            jax.random.categorical(sample_key, logits).astype(jnp.int32),
            value,
            state,
        )

    def greedy_action(
        self, obs: Observation, legal_mask: Array, key: PRNGKeyArray, state=None
    ):
        """Search, then play the best move found."""
        weights, value = _search_one(self, obs, key)
        return (
            jnp.argmax(_policy_logits(weights, legal_mask)).astype(jnp.int32),
            value,
            state,
        )

    def loss(self, batch: Batch):
        raise TypeError(
            "AlphaZeroAgent learns from search targets, not from a Batch: train it "
            "with bomberman.agents.alphazero.train, or pass None as its optimiser "
            "to keep it in a Training line-up as a frozen opponent."
        )


def _policy_logits(weights: Array, legal_mask: Array) -> Array:
    return jnp.where(legal_mask, jnp.log(weights + 1e-12), -jnp.inf)


class Node(eqx.Module):
    """What ``mctx`` stores in each node of the search tree."""

    state: GameState  # the game state at this node
    ego: Array  # () int32, the agent the search plans for
    legal: Array  # (N_ACTIONS,) bool, its legal actions
    prior: Array  # (N_ACTIONS,) its logits, illegal actions masked
    value: Array  # () its value estimate
    terminated: Array  # () bool, the game is over or the agent is dead
    seat_logits: Array  # (num_agents, N_ACTIONS) masked logits of every agent


def _node(
    agent: AlphaZeroAgent, state: GameState, ego: Array, key: PRNGKeyArray
) -> Node:
    """Run the network for every agent on its own observation of ``state``."""
    seats = jnp.arange(agent.num_agents)
    obs = jax.vmap(agent.game.observe, in_axes=(None, 0))(state, seats)
    legal = jax.vmap(agent.game.legal_action_mask, in_axes=(None, 0))(state, seats)
    logits, value = jax.vmap(agent.forward)(
        obs, jax.random.split(key, agent.num_agents)
    )
    logits = jnp.where(legal, logits, ILLEGAL_LOGIT)
    return Node(
        state=state,
        ego=ego,
        legal=legal[ego],
        prior=logits[ego],
        value=value[ego].astype(jnp.float32),
        terminated=agent.game.is_terminal(state) | ~state.alive[ego],
        seat_logits=logits,
    )


def recurrent_fn(agent: AlphaZeroAgent, key: PRNGKeyArray, action: Array, node: Node):
    """Expand a batch of nodes by playing one tick (pgx's ``recurrent_fn``)."""

    def step(node: Node, action: Array, key: PRNGKeyArray):
        order_key, opponent_key, chance_key, network_key = jax.random.split(key, 4)
        before = dataclasses.replace(
            node.state,
            turn_order=jax.random.permutation(order_key, jnp.arange(agent.num_agents)),
            _key=chance_key,
        )
        actions = jax.random.categorical(
            opponent_key, node.seat_logits / agent.opponent_temperature
        ).astype(jnp.int32)
        actions = actions.at[node.ego].set(action.astype(jnp.int32))
        after = agent.game.step(before, actions)
        reward = agent.reward_fn(before, actions, after)[node.ego]
        return _node(agent, after, node.ego, network_key), reward.astype(jnp.float32)

    child, reward = jax.vmap(step)(node, action, jax.random.split(key, action.shape[0]))
    output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=jnp.where(child.terminated, 0.0, agent.discount).astype(jnp.float32),
        prior_logits=child.prior,
        value=jnp.where(child.terminated, 0.0, child.value),
    )
    return output, child


def search(agent: AlphaZeroAgent, observations: Observation, key: PRNGKeyArray):
    """Search from a batch of observations and return the move probabilities ``(B, A)`` and values ``(B,)``."""
    copies = agent.determinizations
    observations = jax.tree_util.tree_map(
        lambda x: jnp.repeat(x, copies, axis=0), observations
    )
    batch = observations.agent_id.shape[0]

    root_key, belief_key, search_key = jax.random.split(key, 3)
    worlds = jax.vmap(determinize)(observations, jax.random.split(belief_key, batch))
    roots = jax.vmap(lambda w, ego, k: _node(agent, w, ego, k))(
        worlds, observations.agent_id, jax.random.split(root_key, batch)
    )
    root_value = jnp.where(roots.terminated, 0.0, roots.value)

    if agent.num_simulations == 0:
        weights, value = jax.nn.softmax(roots.prior, axis=-1), root_value
    else:
        output = mctx.gumbel_muzero_policy(
            params=agent,
            rng_key=search_key,
            root=mctx.RootFnOutput(
                prior_logits=roots.prior, value=root_value, embedding=roots
            ),
            recurrent_fn=recurrent_fn,
            num_simulations=agent.num_simulations,
            invalid_actions=~roots.legal,
            qtransform=mctx.qtransform_completed_by_mix_value,
            max_num_considered_actions=N_ACTIONS,
            gumbel_scale=1.0,
        )
        weights, value = output.action_weights, output.search_tree.summary().value

    weights = weights.reshape(-1, copies, N_ACTIONS).mean(axis=1)
    return weights, value.reshape(-1, copies).mean(axis=1)


def _search_one(agent: AlphaZeroAgent, obs: Observation, key: PRNGKeyArray):
    weights, value = search(agent, jax.tree_util.tree_map(lambda x: x[None], obs), key)
    return weights[0], value[0]


class SelfplayOutput(eqx.Module):
    """What self-play records for every agent and tick, shape ``(rounds, num_envs, num_agents, ...)``."""

    obs: Observation  # what the agent saw
    legal: Array  # (..., N_ACTIONS) bool
    action_weights: Array  # (..., N_ACTIONS) move probabilities from the search
    reward: Array  # the agent's reward for this tick
    root_value: Array  # the search's value of the current state
    active: Array  # bool, the agent was alive and actually made a move
    done: Array  # bool, the agent died or the game ended


def _where(cond: Array, x, y):
    """Pick ``x`` where ``cond`` is True and ``y`` elsewhere, one flag per row, for whole pytrees."""
    return jax.tree_util.tree_map(
        lambda a, b: jnp.where(cond.reshape((-1,) + (1,) * (a.ndim - 1)), a, b), x, y
    )


def selfplay(
    agent: AlphaZeroAgent,
    key: PRNGKeyArray,
    num_envs: int,
    rounds: int,
    explore_moves: int,
) -> tuple[SelfplayOutput, dict[str, Array]]:
    """Play ``num_envs`` games for ``rounds`` ticks, restarting games that end (pgx's ``selfplay``)."""
    game, n = agent.game, agent.num_agents
    seats = jnp.arange(n)
    flat = lambda tree: jax.tree_util.tree_map(
        lambda a: a.reshape(num_envs * n, *a.shape[2:]), tree
    )
    unflat = lambda a: a.reshape(num_envs, n, *a.shape[1:])

    def step_fn(carry, key):
        state, coins, stats = carry
        search_key, sample_key, reset_key = jax.random.split(key, 3)

        obs = jax.vmap(lambda s: jax.vmap(game.observe, in_axes=(None, 0))(s, seats))(
            state
        )
        legal = jax.vmap(
            lambda s: jax.vmap(game.legal_action_mask, in_axes=(None, 0))(s, seats)
        )(state)
        weights, root_value = search(agent, flat(obs), search_key)

        logits = _policy_logits(weights, flat(legal))
        sampled = jax.random.categorical(sample_key, logits, axis=-1)
        explore = jnp.repeat(state.step_count < explore_moves, n)
        action = unflat(
            jnp.where(explore, sampled, jnp.argmax(logits, axis=-1)).astype(jnp.int32)
        )

        nxt = jax.vmap(game.step)(state, action)
        reward = jax.vmap(agent.reward_fn)(state, action, nxt)
        coins = coins + jax.vmap(coins_collected)(state, nxt)
        terminated = jax.vmap(game.is_terminal)(nxt)

        # Results of the games that ended this tick, summed for logging.
        top = nxt.score == nxt.score.max(axis=-1, keepdims=True)
        winner = top & (top.sum(axis=-1, keepdims=True) == 1)
        ended = terminated.astype(jnp.float32)
        stats = {
            "episodes": stats["episodes"] + ended.sum(),
            "coins": stats["coins"] + (ended * coins.mean(-1)).sum(),
            "death": stats["death"] + (ended * (~nxt.alive).mean(-1)).sum(),
            "win": stats["win"] + (ended * winner.mean(-1)).sum(),
            "world_steps": stats["world_steps"] + (ended * nxt.step_count).sum(),
        }

        record = SelfplayOutput(
            obs=obs,
            legal=legal,
            action_weights=unflat(weights),
            reward=reward,
            root_value=unflat(root_value),
            active=state.alive,
            done=terminated[:, None] | ~nxt.alive,
        )

        # A finished game starts over (pgx's `auto_reset`).
        fresh = jax.vmap(game.init)(jax.random.split(reset_key, num_envs))
        state = _where(terminated, fresh, nxt)
        coins = jnp.where(terminated[:, None], 0.0, coins)
        return (state, coins, stats), record

    init_key, scan_key = jax.random.split(key)
    state = jax.vmap(game.init)(jax.random.split(init_key, num_envs))
    stats = dict.fromkeys(
        ("episodes", "coins", "death", "win", "world_steps"), jnp.float32(0.0)
    )
    carry = (state, jnp.zeros((num_envs, n), jnp.float32), stats)
    (_, _, stats), data = jax.lax.scan(
        step_fn, carry, jax.random.split(scan_key, rounds)
    )

    episodes = jnp.maximum(stats["episodes"], 1.0)
    metrics = {k: v / episodes for k, v in stats.items() if k != "episodes"}
    metrics["episodes"] = stats["episodes"]
    if n == 1:
        del metrics["win"]
    return data, metrics


class Sample(eqx.Module):
    """One move chosen by the search, used as a training sample."""

    obs: Observation
    legal: Array  # (B, N_ACTIONS) bool
    policy_tgt: Array  # (B, N_ACTIONS)
    value_tgt: Array  # (B,)
    mask: Array  # (B,) bool, the agent actually made a move
    value_mask: Array  # (B,) bool, and the game ended before self-play stopped


def compute_loss_input(data: SelfplayOutput, discount: float) -> Sample:
    """Turn self-play records into training samples, one per agent and tick (pgx's ``compute_loss_input``)."""

    def body(carry, x):
        reward, done = x
        value = reward + discount * (1.0 - done) * carry
        return value, value

    _, value_tgt = jax.lax.scan(
        body,
        jnp.zeros_like(data.reward[0]),
        (data.reward, data.done.astype(jnp.float32)),
        reverse=True,
    )
    complete = jnp.cumsum(data.done[::-1], axis=0)[::-1] >= 1

    rows = lambda tree: jax.tree_util.tree_map(
        lambda a: a.reshape(-1, *a.shape[3:]), tree
    )
    return Sample(
        obs=rows(data.obs),
        legal=rows(data.legal),
        policy_tgt=rows(data.action_weights),
        value_tgt=rows(value_tgt),
        mask=rows(data.active),
        value_mask=rows(data.active & complete),
    )


def loss_fn(
    network: Network,
    features: FeatureShaping,
    samples: Sample,
    key: PRNGKeyArray,
    value_coef: float = 1.0,
):
    """Loss: cross-entropy to the search's move probabilities plus a value error (pgx's ``loss_fn``)."""
    x = jax.vmap(features)(samples.obs).astype(jnp.float32)
    keys = jax.random.split(key, samples.value_tgt.shape[0])
    logits, value = jax.vmap(lambda o, k: network(o, k, inference=False))(x, keys)
    logits = jnp.where(samples.legal, logits, ILLEGAL_LOGIT)

    decided = samples.mask.astype(jnp.float32)
    policy_loss = (
        optax.softmax_cross_entropy(logits, samples.policy_tgt) * decided
    ).sum()
    policy_loss = policy_loss / jnp.maximum(decided.sum(), 1.0)

    scored = samples.value_mask.astype(jnp.float32)
    value_loss = (optax.l2_loss(value, samples.value_tgt) * scored).sum()
    value_loss = value_loss / jnp.maximum(scored.sum(), 1.0)

    loss = policy_loss + value_coef * value_loss
    return loss, {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss}


def _optimizer(config: Config) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )


def update(
    agent: AlphaZeroAgent, opt_state, samples: Sample, key: PRNGKeyArray, config: Config
):
    """One gradient step on ``agent.network`` (pgx's ``train``)."""
    params, static = eqx.partition(agent.network, eqx.is_inexact_array)
    grads, metrics = jax.grad(
        lambda p: loss_fn(
            eqx.combine(p, static), agent.features, samples, key, config.value_coef
        ),
        has_aux=True,
    )(params)
    updates, opt_state = _optimizer(config).update(grads, opt_state, params)
    network = eqx.combine(optax.apply_updates(params, updates), static)
    return dataclasses.replace(agent, network=network), opt_state, metrics


def train(
    agent: AlphaZeroAgent,
    key: PRNGKeyArray,
    iterations: int,
    config: Config | None = None,
    *,
    logger: Logger | None = None,
    checkpointer: Checkpointer | None = None,
    progress: bool = False,
) -> tuple[AlphaZeroAgent, dict[str, np.ndarray]]:
    """The training loop: play games against itself, shuffle the samples into batches, update, repeat."""
    config = config or Config()
    logger = logger or NoLogger()
    checkpointer = checkpointer or NoCheckpointer()
    rules = agent.game.settings.rules
    rounds = config.rounds or rules.max_steps
    explore_moves = (
        config.explore_moves
        if config.explore_moves is not None
        else rules.max_steps // 5
    )
    rows = rounds * config.num_envs * agent.num_agents
    size = min(config.batch_size, rows)
    num_updates = rows // size

    @eqx.filter_jit
    def iteration(agent, opt_state, key):
        play_key, shuffle_key, update_key = jax.random.split(key, 3)
        data, metrics = selfplay(
            agent, play_key, config.num_envs, rounds, explore_moves
        )
        samples = compute_loss_input(data, agent.discount)

        order = jax.random.permutation(shuffle_key, rows)[: num_updates * size]
        minibatches = jax.tree_util.tree_map(
            lambda x: x[order].reshape(num_updates, size, *x.shape[1:]), samples
        )
        params, static = eqx.partition(agent, eqx.is_array)

        def step(carry, xs):
            params, opt_state = carry
            minibatch, key = xs
            agent, opt_state, losses = update(
                eqx.combine(params, static), opt_state, minibatch, key, config
            )
            return (eqx.filter(agent, eqx.is_array), opt_state), losses

        (params, opt_state), losses = jax.lax.scan(
            step,
            (params, opt_state),
            (minibatches, jax.random.split(update_key, num_updates)),
        )
        active = data.active.astype(jnp.float32)
        decided = lambda x: (x * active).sum() / jnp.maximum(active.sum(), 1.0)
        weights = data.action_weights
        metrics = {
            **metrics,
            **{k: v.mean() for k, v in losses.items()},
            "root_value": decided(data.root_value),
            "search_entropy": decided(-(weights * jnp.log(weights + 1e-12)).sum(-1)),
        }
        return eqx.combine(params, static), opt_state, metrics

    opt_state = _optimizer(config).init(eqx.filter(agent.network, eqx.is_inexact_array))
    history: list[dict[str, float]] = []
    steps = range(iterations)
    if progress:
        from tqdm.auto import tqdm

        steps = tqdm(steps)
    for i in steps:
        key, subkey = jax.random.split(key)
        agent, opt_state, metrics = iteration(agent, opt_state, subkey)
        metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
        history.append(metrics)
        if not isinstance(logger, NoLogger) and i % logger.every == 0:
            logger(i, metrics)
        if not isinstance(checkpointer, NoCheckpointer) and i % checkpointer.every == 0:
            checkpointer(i, (agent,), (opt_state,))
    checkpointer.wait()
    return agent, {
        k: np.asarray([m[k] for m in history]) for k in (history[0] if history else {})
    }
