import click
import gym
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
from tensorboardX import SummaryWriter
from tqdm import tqdm

import rl
import rl.utils
import rl.wrappers


class TransitionList:

    def __init__(self) -> None:
        self.transitions = []

    def __len__(self):
        return len(self.transitions)

    def append(self, **kwargs):
        self.transitions.append(kwargs)

    def build_batch(self):
        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, 1), *self.transitions)


def build_agent(obs_space, act_space):

    def agent(obs, state):
        logits = hk.nets.MLP([32, act_space.n])(obs)
        v = jnp.squeeze(hk.nets.MLP([32, 1])(obs), -1)
        return logits, v, state

    return agent


@click.command()
@click.option('--run-id', type=str, required=True)
def main(run_id):
    batch_size = 32
    seed = 42
    num_observations = 50000
    entropy_weight = 1e-2
    horizon = 32

    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))

    env = gym.vector.SyncVectorEnv([lambda: gym.make('CartPole-v1') for _ in range(batch_size)])
    env = gym.wrappers.RecordEpisodeStatistics(env)
    print(env.single_action_space)
    print(env.single_observation_space)

    env.seed(42)

    obs_tm1 = env.reset()
    agent_state_tm1 = None

    agent = hk.without_apply_rng(hk.transform(build_agent(env.single_observation_space, env.single_action_space)))
    agent_forward = jax.jit(agent.apply)
    params = agent.init(next(rng), obs_tm1, agent_state_tm1)

    opt = optax.adam(1e-2)
    opt_state = opt.init(params)

    @jax.jit
    def opt_step(params, opt_state, batch):

        def compute_loss(params, batch):
            logits_tm1, v_tm1, _ = agent_forward(params, batch['obs_tm1'], None)

            _, v_t, _ = agent_forward(params, batch['obs_t'][:, -1], None)

            dist = rlax.softmax()
            entropy_tm1 = dist.entropy(logits_tm1)
            log_prob_tm1 = dist.logprob(batch['a_tm1'], logits_tm1)

            v_target = jax.vmap(rl.utils.n_step_bootstrapped_return, in_axes=0, out_axes=0)(
                r_t=batch['r_t'],
                d_t=batch['d_t'],
                v_t=jax.lax.stop_gradient(v_t),
                discount=jnp.full_like(v_t, 0.98),
            )

            td_error = v_target - v_tm1

            critic_loss = 0.5 * (td_error**2)
            actor_loss = (-log_prob_tm1 * jax.lax.stop_gradient(td_error) - entropy_weight * entropy_tm1)
            loss = actor_loss + critic_loss

            print(
                jax.tree_util.tree_map(lambda x: x.shape, [
                    v_target,
                    v_tm1,
                    td_error,
                    critic_loss,
                    log_prob_tm1,
                    entropy_tm1,
                    actor_loss,
                    loss,
                ]))

            loss = loss.mean()

            return loss, (critic_loss, actor_loss, v_target, td_error, v_tm1, entropy_tm1)

        grads, aux = jax.grad(compute_loss, has_aux=True)(params, batch)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, aux

    transitions = TransitionList()
    writer = SummaryWriter(f'./log/{run_id}')

    episodes_done = 0
    observations_seen = len(obs_tm1)
    pbar = tqdm(total=num_observations, initial=observations_seen)

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
            for k in info_at_end['episode']:
                writer.add_scalar(f'episode/{k}', info_at_end['episode'][k], global_step=observations_seen)
            writer.add_scalar('episode/i', episodes_done, global_step=observations_seen)

        obs_tm1, agent_state_tm1 = obs_t, agent_state_t

        if len(transitions) >= horizon:
            batch = transitions.build_batch()
            transitions = TransitionList()
            # pprint(jax.tree_util.tree_map(lambda x: x.shape, batch))
            params, opt_state, (critic_loss, actor_loss, v_target, td_error, v_tm1,
                                entropy_tm1), = opt_step(params, opt_state, batch)
            writer.add_scalar('critic_loss', critic_loss.mean(), global_step=observations_seen)
            writer.add_scalar('actor_loss', actor_loss.mean(), global_step=observations_seen)
            writer.add_scalar('v_target', v_target.mean(), global_step=observations_seen)
            writer.add_scalar('v_tm1', v_tm1.mean(), global_step=observations_seen)
            writer.add_scalar('td_error', td_error.mean(), global_step=observations_seen)
            writer.add_scalar('enropy_tm1', entropy_tm1.mean(), global_step=observations_seen)


if __name__ == '__main__':
    main()
