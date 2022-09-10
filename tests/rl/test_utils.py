import haiku as hk
import jax
import jax.numpy as jnp
import rlax

from rl.utils import n_step_bootstrapped_return

n_step_bootstrapped_return = jax.jit(n_step_bootstrapped_return)

# TODO: test against rlax


def test_n_step_bootstrapped_return():
    reward_t = jnp.array([1.0, 2.0, 3.0])
    done_t = jnp.array([False, False, True])
    value_prime = jnp.array(4.0)

    actual = n_step_bootstrapped_return(reward_t, done_t, value_prime, discount=0.9)
    expected = jnp.array([5.23, 4.7, 3])

    assert jnp.allclose(actual, expected)


def test_n_step_bootstrapped_return_2():
    reward_t = jnp.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    done_t = jnp.array([False, False, True, False, False, False])
    value_prime = jnp.array(4.0)

    actual = n_step_bootstrapped_return(reward_t, done_t, value_prime, discount=0.9)
    expected = jnp.array([5.23, 4.7, 3, 8.146, 7.94, 6.6])

    assert jnp.allclose(actual, expected)


def test_n_step_bootstrapped_return_rlax():
    size = 100

    rng = hk.PRNGSequence(42)
    reward_t = jax.random.normal(next(rng), shape=[size])
    done_t = jax.random.uniform(next(rng), shape=[size]) < 0.2
    value_prime = jax.random.normal(next(rng), shape=())

    actual = n_step_bootstrapped_return(reward_t, done_t, value_prime, discount=0.9)
    discount = jnp.where(done_t, 0, 0.9)
    expected = rlax.n_step_bootstrapped_returns(reward_t, discount, jnp.full_like(reward_t, value_prime), size)

    assert jnp.allclose(actual, expected)
