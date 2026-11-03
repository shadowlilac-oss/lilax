"""Causal convolution for the Wan autoencoder."""

from typing import Optional, Sequence, Tuple, Union

from flax import nnx
import jax
import jax.numpy as jnp

from lilax._backend.m3.core.sharding import Sharding
from ..utils_wan import constrain, param_metadata

_SPATIAL_NAME = "vae_spatial_activation"
_SPATIAL_AXES = ("activation_batch", None, None, "activation_length", None)


def _canonicalize_tuple(x: Union[int, Sequence[int]], rank: int, name: str) -> Tuple[int, ...]:
  if isinstance(x, int):
    return (x,) * rank
  elif isinstance(x, Sequence) and len(x) == rank:
    return tuple(x)
  raise ValueError(f"Argument '{name}' must be an integer or a sequence of {rank} integers. Got {x}")


class WanCausalConv3d(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, in_channels: int, out_channels: int,
               kernel_size: Union[int, Tuple[int, int, int]], stride: Union[int, Tuple[int, int, int]] = 1,
               padding: Union[int, Tuple[int, int, int]] = 0, use_bias: bool = True,
               sharding: Sharding | None = None, dtype: jnp.dtype = jnp.float32,
               weights_dtype: jnp.dtype = jnp.float32, precision: jax.lax.Precision = None):
    self.kernel_size = _canonicalize_tuple(kernel_size, 3, "kernel_size")
    self.stride = _canonicalize_tuple(stride, 3, "stride")
    padding_tuple = _canonicalize_tuple(padding, 3, "padding")
    self._causal_padding = (
        (0, 0), (2 * padding_tuple[0], 0), (padding_tuple[1], padding_tuple[1]),
        (padding_tuple[2], padding_tuple[2]), (0, 0),
    )
    self._depth_padding_before = self._causal_padding[1][0]
    self.sharding = sharding
    self.conv = nnx.Conv(
        in_features=in_channels, out_features=out_channels, kernel_size=self.kernel_size, strides=self.stride,
        use_bias=use_bias, padding="VALID", rngs=rngs, kernel_init=nnx.initializers.xavier_uniform(),
        kernel_metadata=param_metadata(sharding, "causal_conv_kernel", (None, None, None, None, None)),
        dtype=dtype, param_dtype=weights_dtype, precision=precision,
    )

  def __call__(self, x: jax.Array, cache_x: Optional[jax.Array] = None, idx=-1) -> jax.Array:
    x = constrain(self.sharding, x, _SPATIAL_NAME, _SPATIAL_AXES)
    current_padding = list(self._causal_padding)
    padding_needed = self._depth_padding_before
    if cache_x is not None and padding_needed > 0:
      assert cache_x.shape[0] == x.shape[0] and cache_x.shape[2:] == x.shape[2:]
      cache_len = cache_x.shape[1]
      x = jnp.concatenate([cache_x, x], axis=1)
      padding_needed -= cache_len
      if padding_needed < 0:
        x = x[:, -padding_needed:, ...]
        current_padding[1] = (0, 0)
      else:
        current_padding[1] = (padding_needed, 0)
    padding_to_apply = tuple(current_padding)
    if any(p > 0 for dim_pads in padding_to_apply for p in dim_pads):
      x_padded = jnp.pad(x, padding_to_apply, mode="constant", constant_values=0.0)
    else:
      x_padded = x
    return self.conv(x_padded)


