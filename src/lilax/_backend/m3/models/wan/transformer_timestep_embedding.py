"""Learned timestep projection for the Wan transformer."""

from typing import Any, Optional

from flax import nnx
import jax
import jax.numpy as jnp

from lilax._backend.m3.core.sharding import Sharding
from .utils_wan import get_activation, param_metadata, safe_getattr


class TimestepEmbedding(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, in_channels: int, time_embed_dim: int = 32, act_fn: str = "silu",
               out_dim: int = None, post_act_fn: Optional[str] = None, cond_proj_dim: int = None,
               sample_proj_bias=True, dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32,
               precision: jax.lax.Precision = None, sharding_specs: Optional[Any] = None,
               sharding: Sharding | None = None):
    linear_1_kernel = safe_getattr(sharding_specs, "emb_linear_1_kernel", ("embed", "mlp"))
    linear_1_bias = safe_getattr(sharding_specs, "emb_linear_1_bias", ("mlp",))
    linear_2_kernel = safe_getattr(sharding_specs, "emb_linear_2_kernel", ("mlp", "embed"))
    linear_2_bias = safe_getattr(sharding_specs, "emb_linear_2_bias", ("embed",))
    self.linear_1 = nnx.Linear(rngs=rngs, in_features=in_channels, out_features=time_embed_dim,
                               use_bias=sample_proj_bias, dtype=dtype, param_dtype=weights_dtype, precision=precision,
                               kernel_init=nnx.initializers.xavier_uniform(),
                               kernel_metadata=param_metadata(sharding, "time_embed_linear_1_kernel", linear_1_kernel),
                               bias_init=nnx.initializers.zeros,
                               bias_metadata=param_metadata(sharding, "time_embed_linear_1_bias", linear_1_bias))
    if cond_proj_dim is not None:
      self.cond_proj = nnx.Linear(rngs=rngs)
    else:
      self.cond_proj = None
    self.act = get_activation(act_fn)
    time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
    self.linear_2 = nnx.Linear(rngs=rngs, in_features=time_embed_dim, out_features=time_embed_dim_out,
                               use_bias=sample_proj_bias, dtype=dtype, param_dtype=weights_dtype, precision=precision,
                               kernel_init=nnx.initializers.xavier_uniform(),
                               kernel_metadata=param_metadata(sharding, "time_embed_linear_2_kernel", linear_2_kernel),
                               bias_init=nnx.initializers.zeros,
                               bias_metadata=param_metadata(sharding, "time_embed_linear_2_bias", linear_2_bias))
    self.post_act = None if post_act_fn is None else get_activation(post_act_fn)

  def __call__(self, sample, condition=None):
    if condition is not None:
      sample = sample + self.cond_proj(condition)
    sample = self.linear_1(sample)
    if self.act is not None:
      sample = self.act(sample)
    sample = self.linear_2(sample)
    if self.post_act is not None:
      sample = self.post_act(sample)
    return sample
