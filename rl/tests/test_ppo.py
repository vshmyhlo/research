import chex
import jax
import jax.numpy as jnp
import pytest
from ppo import ActorCritic, compute_gae

from utils import prng_sequence


@pytest.fixture
def keys():
    return prng_sequence(jax.random.PRNGKey(42))


def test_agent_apply(keys):
    agent = ActorCritic(32, 5)
    carry = agent.initialize_carry(next(keys), (8, 10))
    params = agent.init(next(keys), carry, jnp.zeros((8, 10)))
    s_tm1 = jnp.zeros((8, 10))

    carry, (logits_tm1, v_tm1) = agent.apply(
        params,
        carry,
        s_tm1,
    )

    jax.tree.map(lambda x: chex.assert_shape(x, (8, 32)), carry)
    chex.assert_shape(logits_tm1, (8, 5))
    chex.assert_shape(v_tm1, (8,))


def test_agent_scan(keys):
    agent = ActorCritic(32, 5)
    carry = agent.initialize_carry(next(keys), (8, 10))
    params = agent.init(next(keys), carry, jnp.zeros((8, 10)))
    s_tm1 = jnp.zeros((8, 100, 10))
    done_t = jnp.zeros((8, 100), dtype=jnp.bool)

    carry, (logits_tm1, v_tm1) = agent.apply(
        params,
        carry,
        s_tm1,
        done_t,
        method="scan",
    )

    jax.tree.map(lambda x: chex.assert_shape(x, (8, 32)), carry)
    chex.assert_shape(logits_tm1, (8, 100, 5))
    chex.assert_shape(v_tm1, (8, 100))


def test_compute_gae():
    gae = compute_gae(
        r_t=jnp.zeros((8, 10)),
        discount_t=jnp.zeros((8, 10)),
        v_tm1=jnp.zeros((8, 10)),
        v_t=jnp.zeros((8,)),
        lambda_=0.95,
    )
    chex.assert_shape(gae, (8, 10))
