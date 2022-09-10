import chex
import jax
import jax.numpy as jnp


class TransitionList:

    def __init__(self) -> None:
        self.transitions = []

    def __len__(self):
        return len(self.transitions)

    def append(self, **kwargs):
        self.transitions.append(kwargs)

    def build_batch(self):
        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, 1), *self.transitions)


# TODO: jit unrolls the loop
def n_step_bootstrapped_return(
    r_t,
    d_t,
    v_t,
    discount,
):

    chex.assert_equal_shape([r_t, d_t])
    chex.assert_rank([r_t, d_t], 1)
    chex.assert_equal_shape([v_t, discount])
    chex.assert_rank([v_t, discount], 0)

    mask_t = (~d_t).astype(jnp.float32)
    return_ = v_t
    return_t = jnp.zeros_like(r_t)

    for t in reversed(range(r_t.shape[0])):
        return_ = r_t[t] + mask_t[t] * discount * return_
        return_t = return_t.at[t].set(return_)

    return return_t
