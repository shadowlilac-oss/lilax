"""Qwen2 family of model decoder layers."""

import functools
from typing import Any

from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from flax import nnx

from lilax._backend.common.common_types import Config
from lilax._backend.layers.quantization.configure import configure_kv_quant
from lilax._backend.layers.normalizations import RMSNorm
from lilax._backend.layers.quantization.aqt import AqtQuantization as Quant
from lilax._backend.layers.attentions import Attention
from lilax._backend.layers.linears import MlpBlock
from lilax._backend.utils import model_mode_shapes
from lilax._backend.distributed.sharding import create_sharding, get_logical_axis_rules, maybe_shard_with_logical


class AttentionWithNorm(nnx.Module):
  """Base class with shared common components: self-attention block with normalization."""

  def __init__(self, config: Config, mesh: Mesh, model_mode: str, quant: None | Quant, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    batch_size, seq_len = model_mode_shapes.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)
    self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")
    self.mlp_activation_axis_names = ("activation_batch", "activation_norm_length", "activation_mlp")
    self.out_sharding = create_sharding(mesh, self.activation_axis_names, rules=get_logical_axis_rules())
    self.mlp_intermediate_sharding = create_sharding(mesh, self.mlp_activation_axis_names, rules=get_logical_axis_rules())
    self._maybe_shard_with_logical = functools.partial(
        maybe_shard_with_logical, mesh=mesh, shard_mode=config.shard_mode, debug_sharding=config.debug_sharding,
    )
    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim, dtype=config.dtype, weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode, kernel_axes=("norm",), epsilon=config.normalization_layer_epsilon, rngs=rngs,
    )
    query_pre_attn_scalar = config.head_dim**-0.5
    self.self_attention = Attention(
        config=config, num_query_heads=config.num_query_heads, num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim, max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length, attention_kernel=config.attention,
        inputs_q_shape=dummy_inputs_shape, inputs_kv_shape=dummy_inputs_shape, mesh=mesh,
        dtype=config.dtype, weight_dtype=config.weight_dtype, dropout_rate=config.dropout_rate,
        float32_qk_product=config.float32_qk_product, float32_logits=config.float32_logits, quant=quant,
        kv_quant=configure_kv_quant(config), use_ragged_attention=config.use_ragged_attention,
        ragged_block_size=config.ragged_block_size, use_qk_norm=config.use_qk_norm,
        use_bias_in_projections=config.attention_bias, query_pre_attn_scalar=query_pre_attn_scalar,
        model_mode=model_mode, use_mrope=config.use_mrope, mrope_section=config.mrope_section, rngs=rngs,
    )
    self.post_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim, dtype=config.dtype, weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode, kernel_axes=("norm",), epsilon=config.normalization_layer_epsilon, rngs=rngs,
    )

  def apply_attention_with_norm(
      self, inputs: jnp.ndarray, decoder_segment_ids: None | jnp.ndarray, decoder_positions: None | jnp.ndarray,
      deterministic: bool, model_mode: str, kv_cache: None | jnp.ndarray = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    """Applies self-attention with pre and post-layer normalization."""
    inputs = self._maybe_shard_with_logical(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    lnx = self.pre_self_attention_layer_norm(inputs, out_sharding=self.out_sharding)
    lnx = self._maybe_shard_with_logical(lnx, self.activation_axis_names)
    attention_lnx, kv_cache = self.self_attention(
        lnx, lnx, decoder_positions, decoder_segment_ids=decoder_segment_ids, deterministic=deterministic,
        model_mode=model_mode, out_sharding=self.out_sharding, kv_cache=kv_cache, attention_metadata=attention_metadata,
    )
    attention_lnx = self._maybe_shard_with_logical(attention_lnx, self.activation_axis_names)
    intermediate_inputs = inputs + attention_lnx
    hidden_states = self.post_self_attention_layer_norm(intermediate_inputs, out_sharding=self.out_sharding)
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)
    return hidden_states, intermediate_inputs, kv_cache


class Qwen2DecoderLayer(AttentionWithNorm):
  """Qwen2 Transformer decoder layer (dense)."""

  def __init__(self, config: Config, mesh: Mesh, model_mode: str, quant: None | Quant, rngs: nnx.Rngs):
    super().__init__(config, mesh, model_mode, quant, rngs)
    self.mlp = MlpBlock(
        in_features=config.emb_dim, intermediate_dim=config.mlp_dim, activations=config.mlp_activations,
        intermediate_dropout_rate=config.dropout_rate, dtype=config.dtype, weight_dtype=config.weight_dtype,
        config=config, mesh=mesh, quant=quant, model_mode=model_mode, rngs=rngs,
    )

  def __call__(
      self, inputs: jnp.ndarray, decoder_segment_ids: None | jnp.ndarray, decoder_positions: None | jnp.ndarray,
      deterministic: bool, model_mode: str, previous_chunk=None, slot: None | int = None,
      kv_cache: None | jnp.ndarray = None, attention_metadata: None | dict[str, Any] = None,
  ):
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    hidden_states, intermediate_inputs, kv_cache = self.apply_attention_with_norm(
        inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode,
        kv_cache=kv_cache, attention_metadata=attention_metadata,
    )
    mlp_lnx = self.mlp(
        hidden_states, deterministic=deterministic, intermediate_sharding=self.mlp_intermediate_sharding,
        out_sharding=self.out_sharding,
    )
    mlp_lnx = self._maybe_shard_with_logical(mlp_lnx, self.activation_axis_names)
    layer_output = intermediate_inputs + mlp_lnx
    layer_output = self._maybe_shard_with_logical(layer_output, self.activation_axis_names)
    if self.config.scan_layers:
      return layer_output, None
    else:
      return layer_output, kv_cache
