import os

import chex
import click
import gym
import haiku as hk
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax
import rlax
from tensorboardX import SummaryWriter
from tqdm import tqdm

import rl
import rl.utils
import rl.wrappers
from rl.utils import TransitionList


class Meter:

    def __init__(self) -> None:
        self.values = []

    def update(self, x):
        self.values.append(x)

    def compute_and_reset(self):
        values = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, 0), *self.values)
        values = jax.tree_util.tree_map(lambda x: x.mean(), values)
        self.values = []
        return values

    def __len__(self):
        return len(self.values)

    def __bool__(self):
        return len(self) > 0


def build_agent(obs_space, act_space, d=32):

    def agent(obs, state):
        logits = hk.nets.MLP([d, act_space.n])(obs)
        v = jnp.squeeze(hk.nets.MLP([d, 1])(obs), -1)
        return logits, v, state

    return agent


@click.command()
@click.option('--run-id', type=str, required=True)
@click.option('--bs', 'batch_size', type=int, default=1)
@click.option('--ew', 'entropy_weight', type=float, default=1e-2)
@click.option('--h', 'horizon', type=int, default=32)
@click.option('--d', 'discount', type=float, default=0.98)
@click.option('--lr', 'lr', type=float, default=1e-2)
def main(run_id, **kwargs):
    seed = 42
    num_observations = 50000
    run_id = os.path.join(run_id, ','.join(f'{k}={v}' for k, v in kwargs.items()))

    rng = hk.PRNGSequence(seed)

    env = gym.vector.SyncVectorEnv([lambda: gym.make('CartPole-v1') for _ in range(kwargs['batch_size'])])
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(seed)
    print(env.single_action_space)
    print(env.single_observation_space)

    obs_tm1 = env.reset()
    agent_state_tm1 = None

    agent = hk.without_apply_rng(hk.transform(build_agent(env.single_observation_space, env.single_action_space)))
    agent_forward = jax.jit(agent.apply)
    params = agent.init(next(rng), obs_tm1, agent_state_tm1)

    opt = optax.chain(
        # optax.clip_by_global_norm(1),
        optax.adam(kwargs['lr'], b1=0.99),)
    opt_state = opt.init(params)

    @jax.jit
    def opt_step(params, opt_state, batch):

        def compute_loss(params, traj):
            logits_tm1, v_tm1, _ = agent_forward(params, traj['obs_tm1'], None)

            _, v_t, _ = agent_forward(params, traj['obs_t'][-1], None)

            dist = rlax.softmax()
            entropy_tm1 = dist.entropy(logits_tm1)
            log_prob_tm1 = dist.logprob(traj['a_tm1'], logits_tm1)

            v_target = rl.utils.n_step_bootstrapped_return(
                r_t=traj['r_t'],
                d_t=traj['d_t'],
                v_t=jax.lax.stop_gradient(v_t),
                discount=jnp.array(kwargs['discount']),
            )

            td_error = v_target - v_tm1

            critic_loss = 0.5 * (td_error**2)
            actor_loss = (-log_prob_tm1 * jax.lax.stop_gradient(td_error) - kwargs['entropy_weight'] * entropy_tm1)
            loss = actor_loss + critic_loss

            chex.assert_equal_shape(
                [log_prob_tm1, td_error, entropy_tm1, v_target, v_tm1, critic_loss, actor_loss, loss])

            aux = {
                'critic_loss': critic_loss,
                'actor_loss': actor_loss,
                'loss': loss,
                'v_target': v_target,
                'td_error': td_error,
                'v_tm1': v_tm1,
                'entropy_tm1': entropy_tm1
            }

            return loss.mean(), aux

        def compute_loss_batch(params, batch):
            loss, aux = jax.vmap(compute_loss, in_axes=(None, 0))(params, batch)
            return loss.mean(), aux

        grads, aux = jax.grad(compute_loss_batch, has_aux=True)(params, batch)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, aux, grads

    transitions = TransitionList()
    writer = SummaryWriter(f'./log/{run_id}')

    episodes_done = 0
    observations_seen = 0
    opt_steps = 0
    pbar = tqdm(total=num_observations, initial=0)

    meter = Meter()
    ep_meter = Meter()

    while observations_seen < num_observations:
        logits_tm1, _, agent_state_t = agent_forward(params, obs_tm1, agent_state_tm1)

        a_tm1 = rlax.softmax().sample(next(rng), logits_tm1)

        obs_t, r_t, d_t, info = env.step(np.array(a_tm1))

        observations_seen += len(obs_t)
        pbar.update(len(obs_t))

        transitions.append(
            obs_tm1=obs_tm1,
            a_tm1=a_tm1,
            r_t=r_t,
            obs_t=obs_t,
            d_t=d_t,
        )

        for i in jnp.where(d_t)[0]:
            episodes_done += 1
            info_at_end = jax.tree_util.tree_map(lambda x: x[i], info)
            assert info_at_end['_episode']
            ep_meter.update(info_at_end['episode'])
            writer.add_scalar('episode/i', episodes_done, global_step=observations_seen)

        obs_tm1, agent_state_tm1 = obs_t, agent_state_t

        if len(transitions) >= kwargs['horizon']:
            batch = transitions.build_batch()
            transitions = TransitionList()
            # pprint(jax.tree_util.tree_map(lambda x: x.shape, batch))
            params, opt_state, aux, grads = opt_step(params, opt_state, batch)
            opt_steps += 1
            meter.update(aux)

            if opt_steps % 10 == 0:
                if meter:
                    for k, v in meter.compute_and_reset().items():
                        writer.add_scalar(k, v.mean(), global_step=observations_seen)
                if ep_meter:
                    for k, v in ep_meter.compute_and_reset().items():
                        writer.add_scalar(f'episode/{k}', v.mean(), global_step=observations_seen)

                grads, _ = jax.flatten_util.ravel_pytree(grads)
                grad_norm = jnp.linalg.norm(grads, 2)
                writer.add_scalar('grad_norm', grad_norm, global_step=observations_seen)


if __name__ == '__main__':
    main()
