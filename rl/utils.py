import jax
import jax.numpy as jnp


# TODO: jit unrolls the loop
def n_step_bootstrapped_return(
    r_t,
    d_t,
    v_t,
    discount,
):
    # assert shape_matches(reward_t, done_t, dim=2)
    # assert shape_matches(value_prime, dim=1)

    mask_t = (~d_t).astype(jnp.float32)
    return_ = v_t
    return_t = jnp.zeros_like(r_t)

    for t in reversed(range(r_t.shape[0])):
        return_ = r_t[t] + mask_t[t] * discount * return_
        return_t = return_t.at[t].set(return_)

    return return_t
