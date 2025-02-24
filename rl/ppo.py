# from flax import nnx
import functools
from typing import Iterator

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

# Initialise the environment
# for _ in range(1000):
# this is where you would insert your policy
# action = env.action_space.sample()

# step (transition) through the environment with the action
# receiving the next observation, reward and if the episode has terminated or truncated

# If the episode has ended then we can reset to start a new episode
# if terminated or truncated:
#     observation, info = env.reset()


# env.close()

# class Trajectory:


class Actor(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        x = observations
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.num_actions)(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        x = observations
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = jnp.squeeze(nn.Dense(1)(x), -1)
        return x


def make_env(i: int, video_folder: str) -> gym.Env:
    env = gym.wrappers.RecordEpisodeStatistics(gym.make("LunarLander-v3", render_mode="rgb_array"))
    if i == 0:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda i: i % 100 == 0,
            # disable_logger=True,
        )
    return env


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--g", type=float, required=True)
    args = parser.parse_args()

    envs = gym.vector.AsyncVectorEnv(
        [functools.partial(make_env, i=i, video_folder=f"./tf_logs/{args.run_id}/videos") for i in range(args.bs)]
    )

    actor = Actor(envs.action_space[0].n)
    critic = Critic()
    opt = optax.adam(1e-4)

    s_tm1, _ = envs.reset(seed=42)

    rng = prng_sequence(PRNGKey(42))
    params = {
        "actor": actor.init(next(rng), s_tm1),
        "critic": critic.init(next(rng), s_tm1),
    }
    opt_state = opt.init(params)

    train_step = make_train_step(actor, critic, opt, args.g)

    with SummaryWriter(f"./tf_logs/{args.run_id}") as tb_writer:
        step = 0
        pbar = tqdm()
        while step < 1_000_000:
            exps = []
            for _ in range(args.h):
                step += args.bs

                logits = actor.apply(params["actor"], s_tm1)
                dist = distrax.Softmax(logits)
                a_tm1 = dist.sample(seed=next(rng))
                s_t, r_t, terminations_t, truncations_t, infos_t = envs.step(np.array(a_tm1))
                exps.append((s_tm1, a_tm1, r_t, s_t, terminations_t | truncations_t))
                s_tm1 = s_t

                if jnp.any(terminations_t):
                    infos_t = jax.tree.map(lambda x: x[terminations_t], infos_t)
                    tb_writer.add_scalar("eposide/reward", infos_t["episode"]["r"].mean(), global_step=step)
                    tb_writer.add_scalar("eposide/length", infos_t["episode"]["l"].mean(), global_step=step)

                pbar.update(args.bs)

            batch = jax.tree.map(lambda *x: jnp.stack(x, 1), *exps)
            params, opt_state, loss, () = train_step(params, opt_state, *batch)
            tb_writer.add_scalar("loss", loss, global_step=step)


def make_train_step(actor: Actor, critic: Critic, opt: optax.GradientTransformation, g: float):
    @jax.jit
    def compute_loss(params, s_tm1, a_tm1, r_t, s_t, done_t):
        shape = r_t.shape

        logits_tm1 = actor.apply(params["actor"], s_tm1)
        # logp_tm1 = distrax.Softmax(logits_tm1).log_prob(a_tm1)

        v_tm1 = critic.apply(params["critic"], s_tm1)
        v_t = critic.apply(params["critic"], s_t)

        discount_t = jnp.where(
            done_t,
            jnp.zeros(shape),
            jnp.full(shape, g),
        )

        v_target = jax.vmap(rlax.discounted_returns)(
            r_t,
            discount_t=discount_t,
            v_t=jax.lax.stop_gradient(v_t),
        )

        td_error = v_target - v_tm1

        v_tmp = jnp.concatenate([v_tm1, v_t[:, -1:]], 1)

        adv = jax.vmap(rlax.truncated_generalized_advantage_estimation)(
            r_t,
            discount_t=discount_t,
            lambda_=jnp.full(shape[0], 0.95),
            values=jax.lax.stop_gradient(v_tmp),
        )

        actor_loss = jax.vmap(rlax.policy_gradient_loss)(
            logits_tm1,
            a_tm1,
            adv,
            jnp.ones_like(td_error),
        ) + 0.01 * jax.vmap(rlax.entropy_loss)(logits_tm1, jnp.ones_like(td_error))

        critic_loss = jnp.abs(td_error).mean()
        loss = actor_loss.mean() + 0.5 * critic_loss.mean()

        return loss, ()

    @jax.jit
    def train_step(params, opt_state, s_tm1, a_tm1, r_t, s_t, done_t):
        (loss, aux), grads = jax.value_and_grad(compute_loss, has_aux=True)(params, s_tm1, a_tm1, r_t, s_t, done_t)

        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    return train_step


def prng_sequence(key: jax.Array) -> Iterator[jax.Array]:
    while True:
        key, subkey = jax.random.split(key)
        yield subkey


if __name__ == "__main__":
    main()
