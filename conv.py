class ConvFGVMap(nn.Module):
    features: int
    kernel_size: int | Sequence[int]
    strides: None | int | Sequence[int] = 1
    padding: str = "SAME"
    feature_group_count: int = 1
    use_bias: bool = True
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Initializer = nn.initializers.lecun_normal()
    bias_init: Initializer = nn.initializers.zeros_init()
    promote_dtype: PromoteDtypeFn = promote_dtype

    @nn.compact
    def __call__(self, inputs: Array, mask: jax.Array) -> Array:
        chex.assert_rank(inputs, 3)

        kernel_size: Sequence[int]
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size,)
        else:
            kernel_size = tuple(self.kernel_size)

        def maybe_broadcast(
            x: int | Sequence[int] | None,
        ) -> tuple[int, ...]:
            if x is None:
                # backward compatibility with using None as sentinel for
                # broadcast 1
                x = 1
            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return tuple(x)

        strides = maybe_broadcast(self.strides)
        in_features = jnp.shape(inputs)[-1]

        assert in_features % self.feature_group_count == 0
        kernel_shape = kernel_size + (
            in_features // self.feature_group_count,
            self.features,
        )

        kernel = self.param("kernel", self.kernel_init, kernel_shape, self.param_dtype)

        if self.use_bias:
            bias_shape = (self.features,)
            bias = self.param("bias", self.bias_init, bias_shape, self.param_dtype)
        else:
            bias = None

        inputs, kernel, bias = self.promote_dtype(inputs, kernel, bias, dtype=self.dtype)
        assert inputs is not None
        assert kernel is not None

        def conv(inputs: jax.Array, kernel: jax.Array) -> jax.Array:
            return jax.lax.conv_general_dilated(
                inputs,
                kernel,
                strides,
                self.padding,
                dimension_numbers=("NTC", "TIO", "NTC"),
                precision=self.precision,
            )

        mask = ein.rearrange(mask, "b t -> b t 1")
        inputs = jnp.where(mask, inputs, 0.0)
        inputs = ein.rearrange(inputs, "b t (g d) -> b t g d", g=self.feature_group_count)
        kernel = ein.rearrange(kernel, "k d (g o) -> k d g o", g=self.feature_group_count)
        y = jax.vmap(conv, in_axes=(2, 2), out_axes=2)(inputs, kernel)
        y = ein.rearrange(y, "b t g d -> b t (g d)")
        y = jnp.where(mask, y, 0.0)

        if self.use_bias:
            bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)  # type: ignore
            y += bias

        return y
