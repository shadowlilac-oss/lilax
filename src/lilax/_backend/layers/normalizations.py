"""Normalization Layers."""

from typing import Any

from flax import nnx
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from lilax._backend.common.common_types import Array, DType, ShardMode, is_fp8_dtype
from lilax._backend.layers.initializers import Initializer
from lilax._backend.utils import max_logging
from lilax._backend.device_memory import memory_kinds
from lilax._backend.distributed.sharding import truncate_out_sharding


def align_scale_with_normalized_axis(scale: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Reshards a 1-D per-feature scale to match the sharding of the axis it scales."""
  activation_axis = jax.typeof(y).sharding.spec.partitions[-1]
  scale_spec = jax.typeof(scale).sharding.spec
  if scale_spec.partitions[-1] == activation_axis:
    return scale
  return jax.sharding.reshard(
      scale,
      jax.sharding.PartitionSpec(activation_axis, unreduced=scale_spec.unreduced, reduced=scale_spec.reduced),
  )


class RMSNorm(nnx.Module):
  """RMS normalization."""

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-6,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      shard_mode: ShardMode = ShardMode.AUTO,
      kernel_axes: tuple[None | str, ...] = (),
      scale_init: Initializer = jax.nn.initializers.ones,
      parameter_memory_host_offload: bool = False,
      scale_offset: float = 0.0,
      with_scale: bool = True,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = dtype if is_fp8_dtype(weight_dtype) else weight_dtype
    self.shard_mode = shard_mode
    self.kernel_axes = kernel_axes
    self.scale_init = scale_init
    self.parameter_memory_host_offload = parameter_memory_host_offload
    self.scale_offset = scale_offset
    self.with_scale = with_scale
    if self.with_scale:
      self.scale = nnx.Param(scale_init(rngs.params(), (num_features,), self.weight_dtype), out_sharding=kernel_axes)
    else:
      self.scale = None

  def __call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)

    if self.shard_mode != ShardMode.EXPLICIT:
      out_sharding = None
    if out_sharding is not None:
      out_sharding = truncate_out_sharding(out_sharding, y.ndim)
    if not self.with_scale:
      if out_sharding is not None:
        y = jax.lax.with_sharding_constraint(y, out_sharding)
      return y

    scale = self.scale.get_value()
    if self.parameter_memory_host_offload:
      max_logging.log("normalizations.py: Moving scale parameter to device")
      scale = jax.device_put(scale, memory_kinds.device_space())

    scale_fp32 = jnp.asarray(scale, jnp.float32)
    effective_scale = scale_fp32 + self.scale_offset
    if self.shard_mode == ShardMode.EXPLICIT:
      effective_scale = align_scale_with_normalized_axis(effective_scale, y)

    if self.scale_offset != 0.0:
      normed_fp32 = x * lax.rsqrt(mean2 + self.epsilon)
      y = jnp.einsum("...k,k->...k", normed_fp32, effective_scale, out_sharding=out_sharding)
      return jnp.asarray(y, self.dtype)

    effective_scale = jnp.asarray(effective_scale, self.dtype)
    return jnp.einsum("...k,k->...k", y, effective_scale, out_sharding=out_sharding)


class GlobalRMSNorm(RMSNorm):
  """Applies RMSNorm over the last two dimensions (Heads * HeadDim)."""

  def __call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None) -> jnp.ndarray:
    input_shape = x.shape
    flattened_shape = input_shape[:-2] + (input_shape[-2] * input_shape[-1],)
    x_flat = x.reshape(flattened_shape)
    y_flat = super().__call__(x_flat, out_sharding)
    return y_flat.reshape(input_shape)


def Qwen3NextRMSNorm(
    num_features: int,
    epsilon: float = 1e-6,
    dtype: DType = None,
    weight_dtype: DType = None,
    shard_mode=None,
    kernel_axes=None,
    parameter_memory_host_offload=None,
    *,
    rngs: nnx.Rngs,
):
  """Used for input and post attention layernorms in Qwen3NextDecoderLayer."""
  return nnx.data(
      RMSNorm(
          num_features=num_features,
          epsilon=epsilon,
          dtype=dtype,
          weight_dtype=weight_dtype,
          shard_mode=shard_mode if shard_mode is not None else ShardMode.AUTO,
          kernel_axes=kernel_axes if kernel_axes is not None else (),
          scale_init=jax.nn.initializers.zeros,
          scale_offset=1.0,
          parameter_memory_host_offload=bool(parameter_memory_host_offload),
          rngs=rngs,
      )
  )


class Qwen3NextRMSNormGated(nnx.Module):
  """Applies RMS normalization and a gated SiLU activation."""

  def __init__(
      self,
      num_features: int,
      epsilon: float,
      dtype: DType,
      weight_dtype: DType,
      shard_mode: ShardMode = ShardMode.AUTO,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.rms_norm = nnx.data(
        RMSNorm(
            num_features=num_features,
            epsilon=self.epsilon,
            dtype=jnp.float32,
            weight_dtype=weight_dtype,
            shard_mode=shard_mode,
            scale_init=nnx.initializers.ones,
            rngs=rngs,
        )
    )

  def __call__(self, hidden_states: Array, gate: Array, out_sharding: NamedSharding | None = None) -> Array:
    """Applies RMSNorm and then a SiLU gate."""
    normalized_states = self.rms_norm(hidden_states, out_sharding=out_sharding)
    gated_states = normalized_states * jax.nn.silu(gate.astype(jnp.float32))
    return gated_states.astype(self.dtype)


def l2norm(x: Array, dim: int = -1, eps: float = 1e-6) -> Array:
  """Normalizes a vector to have a length of 1."""
  inv_norm = jax.lax.rsqrt((x * x).sum(axis=dim, keepdims=True) + jnp.array(eps, dtype=x.dtype))
  return x * inv_norm
