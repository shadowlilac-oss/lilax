"""Validation of native decoder-block-specific layer constraints."""

from lilax._backend.common.common_types import DecoderBlockType


def validate_decoder_block_constraints(config):
  if config.decoder_block == DecoderBlockType.LLAMA4:
    if config.capacity_factor >= 0:
      raise ValueError(
          "Llama4 decoder has not been tested with capacity_factor >= 0 -- please set that value to -1 for now!"
      )
    if config.num_experts_per_tok > 1:
      raise ValueError("Only top-1 routing is supported for Llama4 for now!")
    if config.base_num_decoder_layers % config.interleave_moe_layer_step != 0:
      raise ValueError(
          f"The number of decoder layers ({config.base_num_decoder_layers}) must be divisible by interleave moe layer step "
          f"({config.interleave_moe_layer_step})"
      )
  if config.decoder_block == DecoderBlockType.ENVY:
    if config.base_num_decoder_layers % config.interleave_moe_layer_step != 0:
      raise ValueError(
          f"The number of decoder layers ({config.base_num_decoder_layers}) must be divisible by interleave moe layer step "
          f"({config.interleave_moe_layer_step})"
      )
  if config.decoder_block in (DecoderBlockType.QWEN3_NEXT, DecoderBlockType.QWEN3_5):
    if int(config.gdn_num_value_heads) % int(config.gdn_num_key_heads) != 0:
      raise ValueError("gdn_num_value_heads must be divisible by gdn_num_key_heads")
    rotary_dim = int(config.head_dim * config.partial_rotary_factor)
    if rotary_dim % 2 != 0:
      raise ValueError(f"Calculated rotary dimension ({rotary_dim}) must be a multiple of 2.")
    gdn_context_parallel_size = config.ici_context_parallelism * config.dcn_context_parallelism
    if gdn_context_parallel_size > 1 and config.context_parallel_load_balance:
      raise ValueError(
          "GatedDeltaNet context parallelism requires context_parallel_load_balance=False. The GatedDeltaNet "
          "layers carry a recurrence, so device order is sequence order: device i composes the state left by "
          "device i-1. DUAL_CHUNK_SWAP hands device 0 the first and last chunks, device 1 the second and "
          "second-to-last, and so on, which composes the segments out of order. Softmax attention tolerates the "
          "reorder because it rebuilds the causal mask from positions; a recurrence cannot. The run still trains "
          "and the loss still falls, so set this explicitly rather than relying on the failure being visible."
      )
  else:
    if config.partial_rotary_factor is not None and config.partial_rotary_factor != 1.0:
      raise ValueError("`partial_rotary_factor` is only effective when `decoder_block` is set to 'qwen3_next'.")
