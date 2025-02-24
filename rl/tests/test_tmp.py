import jax.numpy as jnp
import rlax


def test_discounted_returns():
    r_t = jnp.array(
        [1, 2, 4, 8, 1, 2, 4],
        dtype=jnp.float32,
    )
    discount_t = jnp.array(
        [0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5],
        dtype=jnp.float32,
    )
    actual = rlax.discounted_returns(
        r_t,
        discount_t=discount_t,
        v_t=jnp.full_like(r_t, 8),
    )
    expected = jnp.array(
        [4, 6, 8, 8, 4, 6, 8],
    )

    assert jnp.array_equal(actual, expected)
