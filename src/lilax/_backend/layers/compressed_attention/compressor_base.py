"""Shared projections and normalization for DeepSeek-V4 long-range attention compressors."""


from typing import Any, Optional

import jax.numpy as jnp

from flax import nnx

from lilax._backend.common.common_types import MODEL_MODE_TRAIN
from lilax._backend.layers.initializers import nd_dense_init, NdInitializer
from lilax._backend.layers.linears import DenseGeneral
from lilax._backend.layers.normalizations import RMSNorm
from lilax._backend.layers.quantization.aqt import AqtQuantization as Quant


class BaseDeepseekCompressor(nnx.Module):
  """Shared base class for DeepSeek-V4 long-range attention compressors.

  This module encapsulates the shared infrastructure for both the Heavily Compressed
  Attention (HCA) and Compressed Sparse Attention (CSA) paradigms introduced in DeepSeek-V4.

  Responsibilities:
    1. Initializes and holds the shared Linear projections (kv_proj, gate_proj) used to
       map embeddings into the compressed representation space.
    2. Owns the KV RMSNorm instance applied to the aggregated representations.
    3. Manages common hyperparameter properties (compress_rate, head_dim, dtype).
  """

  def __init__(
      self,
      config: Any,
      compress_ratio: int,
      rotary_embedding: Any,
      proj_multiplier: int,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      quant: Optional[Quant] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.config = config
    self.compress_rate = compress_ratio
    self.head_dim = config.head_dim
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype
    self.model_mode = model_mode
    self.rngs = rngs

    proj_dim = proj_multiplier * self.head_dim

    self.kv_proj = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=proj_dim,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("embed_attn", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=quant,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=self.rngs,
    )

    self.gate_proj = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=proj_dim,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("embed_attn", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=quant,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=self.rngs,
    )

    self.position_bias = nnx.Param(jnp.zeros((self.compress_rate, proj_dim), dtype=self.weight_dtype))

    self.kv_norm = RMSNorm(
        num_features=self.head_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        epsilon=self.config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    self.rotary_emb = rotary_embedding
