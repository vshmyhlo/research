from typing import Any, Generator, Optional, SupportsFloat, Tuple

import chex
import flax
import flax.linen as nn
import flax.struct
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.typing import (
    PRNGKey as PRNGKey,
)
from flax.typing import (
    Shape as Shape,
)
from gymnasium.core import ActType, ObsType
from gymnasium.vector.vector_env import ArrayType, VectorWrapper
from gymnasium.wrappers.utils import RunningMeanStd


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


def lerp(x: chex.Array, y: chex.Array, a: float) -> chex.Array:
    return (1 - a) * x + a * y


class RunningNorm(nn.Module):
    momentum: float = 0.99
    epsilon: float = 1e-5

    @nn.compact
    def __call__(
        self,
        x,
        update_running_average: bool = False,
    ):
        feature_shape = x.shape[-1:]

        ra_mean = self.variable(
            "batch_stats",
            "mean",
            lambda s: jnp.zeros(s, jnp.float32),
            feature_shape,
        )
        ra_var = self.variable(
            "batch_stats",
            "var",
            lambda s: jnp.ones(s, jnp.float32),
            feature_shape,
        )

        if update_running_average:
            reduction_axes = tuple(i for i in range(x.ndim - 1))
            batch_mean = jnp.mean(x, axis=reduction_axes)
            batch_var = jnp.var(x, axis=reduction_axes)
            chex.assert_equal_shape([batch_mean, batch_var, ra_mean.value, ra_var.value])

            if not self.is_initializing():
                ra_mean.value = self.momentum * ra_mean.value + (1 - self.momentum) * batch_mean
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * batch_var

        mean = ra_mean.value
        var = ra_var.value

        return (x - mean) * jax.lax.rsqrt(var + self.epsilon)


class NormalizeReward(VectorWrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.vector.VectorEnv,
        epsilon: float = 1e-8,
    ):
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        VectorWrapper.__init__(self, env)

        self.reward_rms = RunningMeanStd(shape=())
        self.epsilon = epsilon
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        """Property to freeze/continue the running mean calculation of the reward statistics."""
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        """Sets the property to freeze/continue the running mean calculation of the reward statistics."""
        self._update_running_mean = setting

    def step(self, actions: ActType) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Steps through the environment, normalizing the reward returned."""
        obs, reward, terminated, truncated, info = super().step(actions)
        return obs, self.normalize(reward), terminated, truncated, info

    def normalize(self, reward: SupportsFloat):
        """Normalizes the rewards with the running mean rewards and their variance."""
        if self._update_running_mean:
            self.reward_rms.update(reward)
        # mean = np.mean(reward, axis=0)
        mean = self.reward_rms.mean
        return (reward - mean) / np.sqrt(self.reward_rms.var + self.epsilon) + mean
