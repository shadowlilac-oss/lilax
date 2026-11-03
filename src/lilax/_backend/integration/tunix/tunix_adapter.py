from typing import Any, Optional, Tuple

import jax.numpy as jnp
from flax import nnx
from jax import Array

from lilax._backend.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
from lilax._backend.integration.tunix.utils import VllmWeightMapping
from lilax._backend.models.models import Transformer


def _segment_ids_from_attention_mask(attention_mask: Array, input_tokens: Array) -> Array:
  attention_mask = jnp.asarray(attention_mask)
  batch, seq_len = input_tokens.shape
  if attention_mask.shape != (batch, seq_len, seq_len):
    raise ValueError(
        f"attention_mask has shape {attention_mask.shape}, expected {(batch, seq_len, seq_len)} "
        f"for input_tokens of shape {input_tokens.shape}. The last query row is read as Tunix's "
        "per-token non-pad mask, which requires that shape."
    )
  return attention_mask[:, -1, :].astype(jnp.int32)


class TunixMaxTextAdapter(nnx.Module):
  def __init__(
      self, base_model: Transformer, use_standalone_mappings: bool = True,
      use_no_op_mappings: bool = False, pad_id: Optional[int] = None,
  ):
    super().__init__()
    self.base = base_model
    self._vllm_weight_mapping = VllmWeightMapping(
        self.base.config.model_name, HF_MODEL_CONFIGS[self.base.config.model_name].to_dict(), use_standalone_mappings
    )
    self.use_no_op_mappings = use_no_op_mappings
    self._pad_id = pad_id

  def __call__(
      self, input_tokens: Array, positions: Array, cache: Optional[Any], attention_mask: Optional[Array],
      decoder_segment_ids: Optional[Array] = None, output_hidden_states: bool = False,
      forced_routed_experts: Optional[Array] = None, segment_ids: Optional[Array] = None,
  ) -> Tuple[Array, None]:
    if segment_ids is not None:
      decoder_segment_ids = segment_ids
    if decoder_segment_ids is None and attention_mask is not None:
      decoder_segment_ids = _segment_ids_from_attention_mask(attention_mask, input_tokens)
    if decoder_segment_ids is None and self._pad_id is not None:
      decoder_segment_ids = (input_tokens != self._pad_id).astype(jnp.int32)
    logits = self.base(
        decoder_input_tokens=input_tokens, decoder_positions=positions, decoder_segment_ids=decoder_segment_ids,
        forced_routed_experts=forced_routed_experts,
    )
    return logits, None

  def to_hf_mappings(self):
    if self.use_no_op_mappings:
      return {}
    return self._vllm_weight_mapping.to_hf_mapping()

  def to_hf_transpose_keys(self):
    if self.use_no_op_mappings:
      return {}
    return self._vllm_weight_mapping.to_hf_transpose_keys()

  def to_hf_hook_fns(self):
    if self.use_no_op_mappings:
      return {}
    return self._vllm_weight_mapping.to_hf_hook_fns()

  def lora_to_hf_mappings(self):
    if self.use_no_op_mappings:
      return {}
    return self._vllm_weight_mapping.lora_to_hf_mappings()
