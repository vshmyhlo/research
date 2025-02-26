import jax.numpy as jnp
import rlax


def test_discounted_returns_1():
    r_t = jnp.array(
        [1, 2, 4, 8, 2, 4, 8],
        dtype=jnp.float32,
    )
    discount_t = jnp.array(
        [0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5],
        dtype=jnp.float32,
    )
    actual = rlax.discounted_returns(
        r_t,
        discount_t=discount_t,
        v_t=jnp.full_like(r_t, 16),
    )
    expected = jnp.array(
        [4, 6, 8, 8, 8, 12, 16],
    )

    assert jnp.array_equal(actual, expected)


def test_discounted_returns_2():
    r_t = jnp.array(
        [1, 2, 4, 8],
        dtype=jnp.float32,
    )
    discount_t = jnp.array(
        [0.5, 0.5, 0.5, 0.5],
        dtype=jnp.float32,
    )
    actual = rlax.discounted_returns(
        r_t,
        discount_t=discount_t,
        v_t=jnp.zeros_like(r_t),
    )
    expected = jnp.array(
        [4, 6, 8, 8],
    )

    assert jnp.array_equal(actual, expected)


def test_lambda_returns_1():
    r_t = jnp.array(
        [1, 2, 4, 8, 2, 4, 8],
        dtype=jnp.float32,
    )
    discount_t = jnp.array(
        [0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5],
        dtype=jnp.float32,
    )
    discounted = rlax.discounted_returns(
        r_t,
        discount_t=discount_t,
        v_t=jnp.zeros_like(r_t),
    )
    lambda_ = rlax.lambda_returns(
        r_t,
        discount_t=discount_t,
        v_t=jnp.zeros_like(r_t),
        lambda_=1.0,
    )

    assert jnp.array_equal(lambda_, discounted)


def test_gae_1():
    r_t = jnp.array(
        [1, 2, 4, 8, 2, 4, 8],
        dtype=jnp.float32,
    )
    v = jnp.array(
        [1, 2, 3, 4, 1, 2, 3, 4],
        dtype=jnp.float32,
    )

    discount_t = jnp.array(
        [0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5],
        dtype=jnp.float32,
    )
    gae = rlax.truncated_generalized_advantage_estimation(
        r_t,
        discount_t=discount_t,
        lambda_=0.95,
        values=v,
    )

    v_tm1 = v[:-1]
    v_t = v[1:]

    lambda_ = rlax.lambda_returns(
        r_t,
        discount_t=discount_t,
        v_t=v_t,
        lambda_=0.95,
    )

    assert jnp.allclose(lambda_, gae + v_tm1)
