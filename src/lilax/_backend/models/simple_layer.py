"""Simple decoder layers for testing and debugging purposes."""

from typing import Optional

from jax import numpy as jnp
from jax.sharding import Mesh

from flax import nnx
from lilax._backend.common.common_types import Config, ShardMode
from lilax._backend.layers.quantization.aqt import AqtQuantization
from lilax._backend.utils.sharding import create_sharding


class SimpleDecoderLayer(nnx.Module):
  """Decoder layer consisting of a single [embed, embed] weight matrix."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: Optional[AqtQuantization] = None,
  ) -> None:
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.rngs = rngs
    self.quant = quant

    init_fn = nnx.with_partitioning(nnx.initializers.lecun_normal(), sharding=("embed", "mlp"), mesh=self.mesh)
    self.weights = nnx.Param(init_fn(self.rngs.params(), (self.config.emb_dim, self.config.emb_dim)))

    activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")
    self.out_sharding = (
        create_sharding(self.mesh, activation_axis_names) if config.shard_mode == ShardMode.EXPLICIT else None
    )

  def __call__(
      self, inputs: jnp.ndarray, positions, segmentation, deterministic, model_mode,
      previous_chunk=None, page_state=None, slot=None, kv_cache=None, attention_metadata=None,
  ):
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    if self.config.scan_layers:
      return jnp.dot(inputs, self.weights.astype(inputs.dtype), out_sharding=self.out_sharding), None
    return jnp.dot(inputs, self.weights.astype(inputs.dtype), out_sharding=self.out_sharding)


class SimpleMlpDecoderLayer(nnx.Module):
  """Decoder layer consisting of [embed,mlp] followed by an [mlp,embed] matmul."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: Optional[AqtQuantization] = None,
  ) -> None:
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.rngs = rngs
    self.quant = quant

    init_ff1_fn = nnx.with_partitioning(nnx.initializers.lecun_normal(), sharding=("embed", "mlp"), mesh=self.mesh)
    self.ff_1 = nnx.Param(init_ff1_fn(self.rngs.params(), (self.config.emb_dim, self.config.mlp_dim)))

    init_ff2_fn = nnx.with_partitioning(nnx.initializers.lecun_normal(), sharding=("mlp", "embed"), mesh=self.mesh)
    self.ff_2 = nnx.Param(init_ff2_fn(self.rngs.params(), (self.config.mlp_dim, self.config.emb_dim)))

    activation_axes_names = ("activation_batch", "activation_norm_length", "activation_embed")
    self.activation_sharding = (
        create_sharding(mesh, activation_axes_names) if config.shard_mode == ShardMode.EXPLICIT else None
    )
    mlp_axes_names = ("activation_batch", "activation_norm_length", "activation_mlp")
    self.mlp_sharding = create_sharding(mesh, mlp_axes_names) if config.shard_mode == ShardMode.EXPLICIT else None

  def __call__(
      self, inputs: jnp.ndarray, positions, segmentation, deterministic, model_mode,
      previous_chunk=None, page_state=None, slot=0, kv_cache=None, attention_metadata=None,
  ):
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    intermediate = jnp.dot(inputs, self.ff_1.astype(inputs.dtype), out_sharding=self.mlp_sharding)
    output = jnp.dot(intermediate, self.ff_2.astype(inputs.dtype), out_sharding=self.activation_sharding)
    if self.config.scan_layers:
      return output, None
    return output
