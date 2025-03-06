from typing import Any, Generator, Optional, Tuple

import chex
import flax
import flax.struct
import jax
import jax.numpy as jnp
import optax


class NormalizeEMA:
    def __init__(self, decay: float, accumulator_dtype: Optional[Any] = None):
        self.mean_ema = optax.ema(decay, debias=True, accumulator_dtype=accumulator_dtype)
        self.var_ema = optax.ema(decay, debias=True, accumulator_dtype=accumulator_dtype)

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


@flax.struct.dataclass
class NormStdEmaState:
    var: optax.OptState


class NormStdEma:
    def __init__(self, decay: float):
        self.var_ema = optax.ema(decay, debias=True)

    def init(self) -> NormStdEmaState:
        return NormStdEmaState(var=self.var_ema.init(jnp.zeros(())))

    def update(self, state: NormStdEmaState, values: chex.Array) -> Tuple[NormStdEmaState, chex.Array]:
        mean, var = jnp.mean(values), jnp.var(values)
        var, var_state = self.var_ema.update(var, state.var)
        values = ((values - mean) * jax.lax.rsqrt(var + 1e-8)) + mean
        return NormStdEmaState(var=var_state), values


def prng_sequence(key: chex.PRNGKey) -> Generator[chex.PRNGKey, None, None]:
    while True:
        key, subkey = jax.random.split(key)
        yield subkey
