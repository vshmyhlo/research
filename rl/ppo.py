import argparse
import functools
import math
from typing import Any, Iterator, Optional, Tuple

import chex
import distrax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
from jax.random import PRNGKey
from tensorboardX import SummaryWriter
from tqdm import tqdm


class Actor(nn.Module):
    dim: int
    num_actions: int

    @nn.compact
    def __call__(self, s: chex.Array) -> chex.Array:
        x = s
        x = nn.Dense(self.dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.num_actions)(x)
        return x


class Critic(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, s: chex.Array) -> chex.Array:
        x = s
        x = nn.Dense(self.dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        x = jnp.squeeze(x, -1)
        return x


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
        "--bs",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--discount",
        "-g",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--dim",
        "-d",
        type=int,
        required=True,
    )
    args = parser.parse_args()

    num_train_observations = 1_000_000
    update_epochs = 4
    num_train_steps = math.ceil(num_train_observations / (args.bs * args.horizon) * update_epochs)
    print(f"{num_train_observations=}, {num_train_steps=}")

    # init env
    env_name = "LunarLander-v3"
    # obs_space = gym.make(env_name).observation_space
    envs = gym.make_vec(
        env_name,
        num_envs=args.bs,
        vectorization_mode="async",
        wrappers=[
            functools.partial(gym.wrappers.TimeLimit, max_episode_steps=400),
            gym.wrappers.RecordEpisodeStatistics,
            # functools.partial(
            #     gym.wrappers.RescaleObservation,
            #     min_obs=obs_space.low,
            #     max_obs=obs_space.high,
            # ),
        ],
    )

    # init models
    actor = Actor(args.dim, envs.action_space[0].n)
    critic = Critic(args.dim)

    # init optimizer
    lr_schedule = optax.cosine_decay_schedule(2.5e-4, num_train_steps)
    opt = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(lr_schedule),
    )

    # init states
    s_tm1, _ = envs.reset(seed=42)
    rng = prng_sequence(PRNGKey(42))
    # TODO: fix this for optimizer and batch norm
    params = {
        "actor": {"params": actor.init(next(rng), s_tm1)["params"]},
        "critic": {"params": critic.init(next(rng), s_tm1)["params"]},
    }
    opt_state = opt.init(params)

    # reward ema
    r_ema = WhitenEMA(0.99)
    r_ema_state = r_ema.init()

    # make train step functino
    opt_step = make_opt_step(actor, critic, opt, args.discount)

    with SummaryWriter(f"./tf_logs/{args.run_id}") as tb_writer:
        train_step = 0
        observations_seen = 0
        pbar = tqdm()

        while train_step < num_train_steps:
            exps = []
            for _ in range(args.horizon):
                observations_seen += args.bs
                pbar.update(args.bs)

                logits_tm1 = actor.apply(params["actor"], s_tm1)
                dist_tm1 = distrax.Softmax(logits_tm1)
                a_tm1 = dist_tm1.sample(seed=next(rng))
                logprob_tm1 = dist_tm1.log_prob(a_tm1)

                s_t, r_t, terminations_t, truncations_t, infos_t = envs.step(np.array(a_tm1))
                done_t = terminations_t | truncations_t
                del terminations_t, truncations_t

                exps.append(
                    {
                        "s_tm1": s_tm1,
                        "a_tm1": a_tm1,
                        "logprob_tm1": logprob_tm1,
                        "r_t": r_t,
                        "done_t": done_t,
                    }
                )
                s_tm1 = s_t

                if jnp.any(done_t):
                    infos_t = jax.tree.map(lambda x: x[done_t], infos_t)
                    tb_writer.add_scalar(
                        "episode/reward", infos_t["episode"]["r"].mean(), global_step=observations_seen
                    )
                    tb_writer.add_scalar(
                        "episode/length", infos_t["episode"]["l"].mean(), global_step=observations_seen
                    )

            batch = jax.tree.map(lambda *x: jnp.stack(x, 1), *exps)
            batch["s_t"] = s_t
            # r_ema_state, batch["r_t"] = r_ema.update(r_ema_state, batch.pop("r_t"), keep_mean=True)

            for _ in range(update_epochs):
                params, opt_state, loss, aux = opt_step(params, opt_state, **batch)
                train_step = optax.tree_utils.tree_get_all_with_path(opt_state, "count")[0][-1]

            tb_writer.add_scalar("loss", loss, global_step=observations_seen)
            tb_writer.add_scalar("lr", lr_schedule(train_step), global_step=observations_seen)
            tb_writer.add_scalar("adv/abs", jnp.abs(aux["adv"]).mean(), global_step=observations_seen)
            tb_writer.add_scalar("adv/std", jnp.std(aux["adv"]), global_step=observations_seen)
            tb_writer.add_scalar("td_error/abs", jnp.abs(aux["td_error"]).mean(), global_step=observations_seen)
            tb_writer.add_scalar("td_error/std", jnp.std(aux["td_error"]), global_step=observations_seen)
            tb_writer.add_scalar("pg_loss", aux["pg_loss"], global_step=observations_seen)
            tb_writer.add_scalar("entropy_loss", aux["entropy_loss"], global_step=observations_seen)
            tb_writer.add_scalar("critic_loss", aux["critic_loss"], global_step=observations_seen)
            tb_writer.add_scalar("grad_norm/actor", aux["actor_grad_norm"], global_step=observations_seen)
            tb_writer.add_scalar("grad_norm/critic", aux["critic_grad_norm"], global_step=observations_seen)
            tb_writer.add_scalar(
                "prob_ratios/abs", jnp.abs(aux["prob_ratios_tm1"]).mean(), global_step=observations_seen
            )
            tb_writer.add_scalar("prob_ratios/std", jnp.std(aux["prob_ratios_tm1"]), global_step=observations_seen)

            if train_step % (100 * update_epochs) == 0:
                frames, fps = render_video(
                    env_name,
                    lambda s: distrax.Softmax(actor.apply(params["actor"], s)).sample(seed=next(rng)),
                )
                tb_writer.add_video("video", [frames], global_step=observations_seen, dataformats="NTHWC", fps=fps)

            tb_writer.flush()


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


def make_opt_step(actor: Actor, critic: Critic, opt: optax.GradientTransformation, discount: float):
    @jax.jit
    def compute_loss(params, s_tm1, a_tm1, logprob_old_tm1, r_t, s_t, done_t):
        batch_shape = r_t.shape

        # compute action scores
        logits_tm1: chex.Array = actor.apply(params["actor"], s_tm1)

        # compute value estimate
        s = jnp.concatenate([s_tm1, jnp.expand_dims(s_t, 1)], 1)
        v: chex.Array = critic.apply(params["critic"], s)
        v_tm1 = v[:, :-1]

        # discount factor
        discount_t = jnp.where(done_t, 0, jnp.full(batch_shape, discount))

        # compute advantage
        adv_t = jax.vmap(rlax.truncated_generalized_advantage_estimation)(
            r_t,
            discount_t=discount_t,
            lambda_=jnp.full(batch_shape[0], 0.95),
            values=jax.lax.stop_gradient(v),
        )

        logprob_tm1 = distrax.Softmax(logits_tm1).log_prob(a_tm1)
        prob_ratios_tm1 = jnp.exp(logprob_tm1 - jax.lax.stop_gradient(logprob_old_tm1))

        # compute loss
        pg_loss = compute_ppo_loss(prob_ratios_tm1=prob_ratios_tm1, adv_t=adv_t)
        entropy_loss = compute_entropy_loss(logits=logits_tm1)
        td_target = jax.lax.stop_gradient(adv_t + v_tm1)
        critic_loss = compute_value_loss(td_target=td_target, v_tm1=v_tm1)

        # total loss
        loss = pg_loss + 0.01 * entropy_loss + 0.5 * critic_loss

        return loss, {
            "td_error": td_target - v_tm1,
            "pg_loss": pg_loss,
            "entropy_loss": entropy_loss,
            "critic_loss": critic_loss,
            "adv": adv_t,
            "prob_ratios_tm1": prob_ratios_tm1,
        }

    @jax.jit
    def train_step(params, opt_state, s_tm1, a_tm1, logprob_tm1, r_t, s_t, done_t):
        (loss, aux), grads = jax.value_and_grad(compute_loss, has_aux=True)(
            params, s_tm1, a_tm1, logprob_tm1, r_t, s_t, done_t
        )
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        aux = {
            **aux,
            "actor_grad_norm": optax.global_norm(updates["actor"]),
            "critic_grad_norm": optax.global_norm(updates["critic"]),
        }
        return params, opt_state, loss, aux

    return train_step


def compute_ppo_loss(*, prob_ratios_tm1: chex.Array, adv_t: chex.Array) -> chex.Array:
    chex.assert_rank([prob_ratios_tm1, adv_t], 2)
    chex.assert_equal_shape([prob_ratios_tm1, adv_t])
    loss_fn = functools.partial(rlax.clipped_surrogate_pg_loss, epsilon=0.2, use_stop_gradient=True)
    return jax.vmap(loss_fn)(prob_ratios_t=prob_ratios_tm1, adv_t=adv_t).mean()


def compute_entropy_loss(*, logits: chex.Array) -> chex.Array:
    chex.assert_rank([logits], 3)
    b, t, _ = logits.shape
    return jax.vmap(rlax.entropy_loss)(logits, jnp.ones((b, t))).mean()


def compute_value_loss(*, td_target: chex.Array, v_tm1: chex.Array) -> chex.Array:
    chex.assert_rank([td_target, v_tm1], 2)
    chex.assert_equal_shape([td_target, v_tm1])
    return optax.huber_loss(td_target, v_tm1).mean()


def prng_sequence(key: chex.Array) -> Iterator[chex.Array]:
    while True:
        key, subkey = jax.random.split(key)
        yield subkey


def whiten(values: chex.Array, keep_mean: bool) -> chex.Array:
    mean, var = jnp.mean(values), jnp.var(values)
    whitened = (values - mean) * jax.lax.rsqrt(var + 1e-8)
    if keep_mean:
        whitened += mean
    return whitened


class WhitenEMA:
    def __init__(self, decay: float, debias: bool = True, accumulator_dtype: Optional[Any] = None):
        self.mean_ema = optax.ema(decay, debias=debias, accumulator_dtype=accumulator_dtype)
        self.var_ema = optax.ema(decay, debias=debias, accumulator_dtype=accumulator_dtype)

    def init(self):
        return {
            "mean": self.mean_ema.init(jnp.array(0, dtype=jnp.float32)),
            "var": self.var_ema.init(jnp.array(0, dtype=jnp.float32)),
        }

    def update(self, state, values: chex.Array, keep_mean: bool) -> Tuple[Any, chex.Array]:
        mean, var = jnp.mean(values), jnp.var(values)

        mean, mean_state = self.mean_ema.update(mean, state["mean"])
        var, var_state = self.var_ema.update(var, state["var"])

        whitened = (values - mean) * jax.lax.rsqrt(var + 1e-8)
        if keep_mean:
            whitened += mean

        return {"mean": mean_state, "var": var_state}, whitened


if __name__ == "__main__":
    main()
