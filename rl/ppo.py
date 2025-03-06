import argparse
import functools
import math
from typing import Iterator, Tuple

import chex
import distrax
import flax.linen as nn
import flax.struct
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
from jax.random import PRNGKey
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import prng_sequence

AgentCarry = Tuple[chex.Array, chex.Array]


@flax.struct.dataclass
class TrainState:
    params: chex.ArrayTree
    batch_stats: chex.ArrayTree
    opt_state: chex.ArrayTree


class ActorCritic(nn.Module):
    dim: int
    num_actions: int

    def setup(self):
        self.lstm = nn.LSTMCell(self.dim, carry_init=nn.initializers.zeros_init())

    @nn.compact
    def __call__(
        self,
        carry: AgentCarry,
        s: chex.Array,
        use_running_average: bool = True,
    ) -> Tuple[AgentCarry, Tuple[chex.Array, chex.Array]]:
        carry, x = self.hidden(carry, s, use_running_average=use_running_average)
        logits = self.logits(x)
        value = self.value(x)
        return carry, (logits, value)

    @nn.compact
    def hidden(self, carry: AgentCarry, s: chex.Array, use_running_average: bool) -> Tuple[AgentCarry, chex.Array]:
        x = s
        x = nn.BatchNorm(momentum=0.99, use_scale=False, use_bias=False)(x, use_running_average=use_running_average)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        return self.lstm(carry, x)

    @nn.compact
    def logits(self, x: chex.Array) -> chex.Array:
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        return x

    @nn.compact
    def value(self, x: chex.Array) -> chex.Array:
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        x = jnp.squeeze(x, -1)
        return x

    @nn.compact
    def scan(
        self,
        carry: AgentCarry,
        s_tm1: chex.Array,
        done_t: chex.Array,
        use_running_average: bool = True,
    ) -> Tuple[AgentCarry, Tuple[chex.Array, chex.Array]]:
        chex.assert_rank([s_tm1, done_t], [3, 2])
        chex.assert_equal_shape_prefix([s_tm1, done_t], 2)
        chex.assert_tree_shape_prefix((carry, s_tm1, done_t), s_tm1.shape[:1])

        def body_fn(self, carry, s_tm1, done_t):
            chex.assert_rank([s_tm1, done_t], [2, 1])
            chex.assert_tree_shape_prefix([carry, s_tm1, done_t], s_tm1.shape[:1])

            carry, (logits_tm1, v_tm1) = self(carry, s_tm1, use_running_average=use_running_average)
            carry = self.reset_carry(carry, done_t)
            return carry, (logits_tm1, v_tm1)

        if use_running_average:
            variable_broadcast = ["params", "batch_stats"]
            variable_carry = False
        else:
            variable_broadcast = ["params"]
            variable_carry = "batch_stats"

        scan = nn.scan(
            body_fn,
            variable_broadcast=variable_broadcast,
            variable_carry=variable_carry,
            in_axes=1,
            out_axes=1,
        )
        return scan(
            self,
            carry,
            s_tm1,
            done_t,
        )

    @nn.nowrap
    def initialize_carry(self, key: chex.PRNGKey, batch_shape: Tuple[int, ...]) -> AgentCarry:
        return nn.LSTMCell(self.dim, carry_init=nn.initializers.zeros_init()).initialize_carry(key, batch_shape)

    @nn.nowrap
    def reset_carry(self, carry: AgentCarry, done_t: chex.Array) -> AgentCarry:
        chex.assert_rank([done_t], 1)
        chex.assert_tree_shape_prefix([done_t, carry], done_t.shape)

        return jax.tree.map(
            lambda c: jnp.where(
                jnp.expand_dims(done_t, -1),
                jnp.zeros_like(c),
                c,
            ),
            carry,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--horizon",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num_minibatches",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    num_train_observations = 1_000_000
    num_train_steps = math.ceil(
        num_train_observations / (args.num_envs * args.horizon) * args.num_epochs * args.num_minibatches
    )
    print(f"{num_train_observations=}, {num_train_steps=}")

    # init env
    env_name = "LunarLander-v3"
    # obs_space = gym.make(env_name).observation_space
    envs = gym.make_vec(
        env_name,
        num_envs=args.num_envs,
        vectorization_mode="async",
        wrappers=[
            gym.wrappers.RecordEpisodeStatistics,
        ],
    )

    # init model
    agent = ActorCritic(args.dim, envs.single_action_space.n)

    # init optimizer
    lr_schedule = optax.cosine_decay_schedule(2.5e-4, num_train_steps)
    opt = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(lr_schedule),
    )
    # reward_norm = NormStdEma(0.8)

    # init states
    s_tm1, _ = envs.reset(seed=42)
    rng = prng_sequence(PRNGKey(42))
    agent_carry_tm1 = agent.initialize_carry(next(rng), s_tm1.shape)
    state = agent.init(next(rng), agent_carry_tm1, s_tm1)
    params = state.pop("params")
    train_state = TrainState(
        params=params,
        batch_stats=state.pop("batch_stats"),
        opt_state=opt.init(params),
    )
    assert not state
    del state, params

    # make train step functino
    opt_step = make_opt_step(
        agent,
        opt,
        discount=args.discount,
        lambda_=args.lambda_,
        num_epochs=args.num_epochs,
        num_minibatches=args.num_minibatches,
    )

    with SummaryWriter(f"./tf_logs/{args.run_id}") as tb_writer:
        train_step = 0
        observations_seen = 0
        pbar = tqdm()

        while train_step < num_train_steps:
            s_tm1, agent_carry_tm1, observations_seen, batch, batch_stats = collect_trajectory_batch(
                params=train_state.params,
                batch_stats=train_state.batch_stats,
                s_tm1=s_tm1,
                agent_carry_tm1=agent_carry_tm1,
                observations_seen=observations_seen,
                envs=envs,
                agent=agent,
                tb_writer=tb_writer,
                horizon=args.horizon,
                rng=rng,
                pbar=pbar,
            )
            train_state = train_state.replace(batch_stats=batch_stats)
            train_state, aux = opt_step(train_state, next(rng), **batch)
            train_step = optax.tree_utils.tree_get_all_with_path(train_state.opt_state, "count")[0][-1]

            tb_writer.add_scalar("lr", lr_schedule(train_step), global_step=observations_seen)
            tb_writer.add_scalar("grad_norm", aux["grad_norm"].mean(), global_step=observations_seen)

            tb_writer.add_scalar("loss/total", aux["loss"].mean(), global_step=observations_seen)
            tb_writer.add_scalar("loss/pg", aux["pg_loss"].mean(), global_step=observations_seen)
            tb_writer.add_scalar("loss/entropy", aux["entropy_loss"].mean(), global_step=observations_seen)
            tb_writer.add_scalar("loss/critic", aux["critic_loss"].mean(), global_step=observations_seen)

            tb_writer.add_scalar("adv/mean", aux["adv"].mean(), global_step=observations_seen)
            tb_writer.add_scalar("adv/std", jnp.std(aux["adv"]), global_step=observations_seen)
            tb_writer.add_scalar("td_error/mean", aux["td_error"].mean(), global_step=observations_seen)
            tb_writer.add_scalar("td_error/std", jnp.std(aux["td_error"]), global_step=observations_seen)
            tb_writer.add_scalar("prob_ratios/mean", aux["prob_ratios_tm1"].mean(), global_step=observations_seen)
            tb_writer.add_scalar("prob_ratios/std", jnp.std(aux["prob_ratios_tm1"]), global_step=observations_seen)

            # if train_step % (100 * args.num_epochs * args.num_minibatches) == 0:
            #     frames, fps = render_video(
            #         env_name,
            #         lambda a_state, s: distrax.Softmax(agent.apply(params, a_state, s)[0]).sample(seed=next(rng)),
            #     )
            #     tb_writer.add_video("video", [frames], global_step=observations_seen, dataformats="NTHWC", fps=fps)

            tb_writer.flush()


def collect_trajectory_batch(
    *,
    params: chex.ArrayTree,
    batch_stats: chex.ArrayTree,
    s_tm1: chex.Array,
    agent_carry_tm1: AgentCarry,
    observations_seen: int,
    envs,
    agent: ActorCritic,
    tb_writer: SummaryWriter,
    horizon: int,
    rng: Iterator[chex.PRNGKey],
    pbar: tqdm,
) -> Tuple[chex.Array, AgentCarry, int, chex.ArrayTree, chex.ArrayTree]:
    agent_carry_tm1_ = agent_carry_tm1
    batch = []
    for _ in range(horizon):
        (agent_carry_t, (logits_tm1, v_tm1)), batch_stats_update = agent.apply(
            {"params": params, "batch_stats": batch_stats},
            agent_carry_tm1,
            s_tm1,
            use_running_average=False,
            mutable=["batch_stats"],
        )
        batch_stats = batch_stats_update.pop("batch_stats")
        assert not batch_stats_update

        dist_tm1 = distrax.Softmax(logits_tm1)
        a_tm1 = dist_tm1.sample(seed=next(rng))
        logprob_tm1 = dist_tm1.log_prob(a_tm1)

        s_t, r_t, terminations_t, truncations_t, infos_t = envs.step(np.array(a_tm1))
        done_t = terminations_t | truncations_t
        del terminations_t, truncations_t

        batch.append(
            {
                "s_tm1": s_tm1,
                "a_tm1": a_tm1,
                "v_tm1": v_tm1,
                "logprob_tm1": logprob_tm1,
                "r_t": r_t,
                "done_t": done_t,
            }
        )

        agent_carry_tm1 = agent.reset_carry(agent_carry_t, done_t)
        s_tm1 = s_t
        observations_seen += envs.num_envs
        pbar.update(envs.num_envs)

        if jnp.any(done_t):
            infos_t = jax.tree.map(lambda x: x[done_t], infos_t)
            tb_writer.add_scalar("episode/reward", infos_t["episode"]["r"].mean(), global_step=observations_seen)
            tb_writer.add_scalar("episode/length", infos_t["episode"]["l"].mean(), global_step=observations_seen)
            tb_writer.add_scalar(
                "episode/reward_over_length",
                infos_t["episode"]["r"].mean() / infos_t["episode"]["l"].mean(),
                global_step=observations_seen,
            )

    batch = jax.tree.map(lambda *x: jnp.stack(x, 1), *batch)
    _, (_, batch["v_t"]) = agent.apply(
        {"params": params, "batch_stats": batch_stats},
        agent_carry_t,
        s_t,
    )
    batch["agent_carry_tm1"] = agent_carry_tm1_

    return (
        s_tm1,
        agent_carry_tm1,
        observations_seen,
        batch,
        batch_stats,
    )


def render_video(env_name, get_action):
    env = gym.wrappers.RenderCollection(gym.make(env_name, render_mode="rgb_array"))
    s, _ = env.reset(seed=42)
    while True:
        (a,) = get_action([s])
        s, _, termination, truncation, info = env.step(a.item())
        if truncation or termination:
            break
    frames = env.render()
    return frames, env.metadata["render_fps"]


def make_opt_step(
    agent: ActorCritic,
    opt: optax.GradientTransformation,
    *,
    discount: float,
    lambda_: float,
    num_epochs: int,
    num_minibatches: int,
):
    @jax.jit
    def compute_loss(
        params: chex.ArrayTree,
        *,
        batch_stats: chex.ArrayTree,
        agent_carry_tm1: AgentCarry,
        s_tm1: chex.Array,
        a_tm1: chex.Array,
        done_t: chex.Array,
        logprob_old_tm1: chex.Array,
        adv: chex.Array,
        td_target: chex.Array,
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        dim = chex.Dimensions()
        dim["EHS"] = s_tm1.shape
        chex.assert_shape([s_tm1], dim["EHS"])
        chex.assert_shape([a_tm1, done_t, logprob_old_tm1, adv, td_target], dim["EH"])
        chex.assert_tree_shape_prefix(agent_carry_tm1, dim["E"])

        _, (logits_tm1, v_tm1) = agent.apply(
            {"params": params, "batch_stats": batch_stats},
            agent_carry_tm1,
            s_tm1,
            done_t,
            method="scan",
        )

        logprob_tm1 = distrax.Softmax(logits_tm1).log_prob(a_tm1)
        prob_ratios_tm1 = jnp.exp(logprob_tm1 - logprob_old_tm1)

        # compute loss
        pg_loss = compute_ppo_loss(prob_ratios_tm1=prob_ratios_tm1, adv=normalize(adv), epsilon=0.2)
        entropy_loss = compute_entropy_loss(logits_tm1)
        critic_loss = compute_value_loss(td_target=td_target, v_tm1=v_tm1)

        # total loss
        loss = pg_loss + 0.01 * entropy_loss + 0.5 * critic_loss

        return (
            loss,
            {
                "td_error": td_target - v_tm1,
                "pg_loss": pg_loss,
                "entropy_loss": entropy_loss,
                "critic_loss": critic_loss,
                "adv": adv,
                "prob_ratios_tm1": prob_ratios_tm1,
            },
        )

    @jax.jit
    def minibatch_train_step(
        train_state: TrainState,
        *,
        s_tm1: chex.Array,
        a_tm1: chex.Array,
        logprob_old_tm1: chex.Array,
        adv: chex.Array,
        td_target: chex.Array,
        done_t: chex.Array,
        agent_carry_tm1: chex.Array,
    ) -> Tuple[TrainState, chex.ArrayTree]:
        (loss, aux), grads = jax.value_and_grad(compute_loss, has_aux=True)(
            train_state.params,
            batch_stats=train_state.batch_stats,
            s_tm1=s_tm1,
            a_tm1=a_tm1,
            logprob_old_tm1=logprob_old_tm1,
            adv=adv,
            td_target=td_target,
            done_t=done_t,
            agent_carry_tm1=agent_carry_tm1,
        )
        updates, opt_state = opt.update(grads, train_state.opt_state)
        params = optax.apply_updates(train_state.params, updates)
        aux = {
            **aux,
            "loss": loss,
            "grad_norm": optax.global_norm(updates),
        }
        return (
            TrainState(
                params=params,
                batch_stats=train_state.batch_stats,
                opt_state=opt_state,
            ),
            aux,
        )

    @jax.jit
    def batch_train_step(
        train_state: TrainState,
        key: chex.PRNGKey,
        *,
        s_tm1: chex.Array,
        a_tm1: chex.Array,
        logprob_tm1: chex.Array,
        r_t: chex.Array,
        done_t: chex.Array,
        v_tm1: chex.Array,
        v_t: chex.Array,
        agent_carry_tm1: chex.Array,
    ):
        dim = chex.Dimensions()
        dim["NT"] = r_t.shape
        chex.assert_shape(s_tm1, dim["NT*"])
        chex.assert_shape([a_tm1, logprob_tm1, r_t, done_t, v_tm1], dim["NT"])
        chex.assert_shape([v_t], dim["N"])
        chex.assert_tree_shape_prefix(agent_carry_tm1, dim["N"])

        adv = compute_gae(
            r_t=r_t,
            discount_t=jnp.where(done_t, 0, discount),
            v_tm1=v_tm1,
            v_t=v_t,
            lambda_=lambda_,
        )

        loss_batch = {
            "s_tm1": s_tm1,
            "a_tm1": a_tm1,
            "logprob_old_tm1": logprob_tm1,
            "adv": adv,
            "td_target": adv + v_tm1,
            "done_t": done_t,
            "agent_carry_tm1": agent_carry_tm1,
        }

        (num_envs,) = dim["N"]
        minibatch_indices = []
        for _ in range(num_epochs):
            key, subkey = jax.random.split(key)
            for iis in jnp.split(jax.random.permutation(subkey, num_envs), num_minibatches):
                minibatch_indices.append(iis)
        minibatch_indices = jnp.stack(minibatch_indices, 0)

        def scan_body_fn(
            train_state: TrainState,
            minibatch_indices: chex.Array,
        ) -> Tuple[TrainState, Tuple[chex.Array, chex.ArrayTree]]:
            loss_minibatch = jax.tree.map(lambda x: x[minibatch_indices], loss_batch)
            train_state, aux = minibatch_train_step(train_state, **loss_minibatch)
            return train_state, aux

        train_state, aux = jax.lax.scan(
            scan_body_fn,
            train_state,
            minibatch_indices,
        )

        return train_state, aux

    return batch_train_step


def compute_gae(
    *,
    r_t: chex.Array,
    discount_t: chex.Array,
    v_tm1: chex.Array,
    v_t: chex.Array,
    lambda_: float,
) -> chex.Array:
    dim = chex.Dimensions()
    dim["NT"] = r_t.shape
    chex.assert_shape([r_t, discount_t, v_tm1], dim["NT"])
    chex.assert_shape(v_t, dim["N"])

    gae_fn = functools.partial(
        rlax.truncated_generalized_advantage_estimation,
        lambda_=lambda_,
        stop_target_gradients=True,
    )
    return jax.vmap(gae_fn)(
        r_t=r_t,
        discount_t=discount_t,
        values=jnp.concatenate([v_tm1, jnp.expand_dims(v_t, 1)], 1),
    )


def compute_ppo_loss(*, prob_ratios_tm1: chex.Array, adv: chex.Array, epsilon: float) -> chex.Array:
    chex.assert_rank([prob_ratios_tm1, adv], 2)
    chex.assert_equal_shape([prob_ratios_tm1, adv])
    loss_fn = functools.partial(rlax.clipped_surrogate_pg_loss, epsilon=epsilon, use_stop_gradient=True)
    return jax.vmap(loss_fn)(prob_ratios_t=prob_ratios_tm1, adv_t=adv).mean()


def compute_entropy_loss(logits: chex.Array) -> chex.Array:
    chex.assert_rank(logits, 3)
    return jax.vmap(rlax.entropy_loss)(logits, jnp.ones(logits.shape[:2])).mean()


def compute_value_loss(*, td_target: chex.Array, v_tm1: chex.Array) -> chex.Array:
    chex.assert_equal_shape([td_target, v_tm1])
    return optax.huber_loss(td_target, v_tm1).mean()


def normalize(
    values: chex.Array,
    keep_mean: bool = False,
    eps: float = 1e-8,
) -> chex.Array:
    mean, var = jnp.mean(values), jnp.var(values)
    whitened = (values - mean) * jax.lax.rsqrt(var + eps)
    if keep_mean:
        whitened += mean
    return whitened


if __name__ == "__main__":
    main()
