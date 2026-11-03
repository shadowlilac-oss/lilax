from lilax.configuration.domain_types import RematLocation


def validate_shard_embed_moe_on_fsdp(self):
  if self.shard_embed_moe_on_fsdp and (
      self.quantization == "" or not self.weight_quantization_calibration_method.startswith("fixed")
  ):
    raise ValueError(
        "shard_embed_moe_on_fsdp requires quantization to be specified and "
        "weight_quantization_calibration_method to be fixed (static scaling mode)."
    )
  return self


def validate_moe_routing_options(self):
  if self.num_moe_emb_chunks > 0:
    if not self.use_gmm_v2 or not self.use_ring_of_experts:
      raise ValueError(
          "num_moe_emb_chunks > 0 requires use_gmm_v2=True and use_ring_of_experts=True. "
          f"Got use_gmm_v2={self.use_gmm_v2}, use_ring_of_experts={self.use_ring_of_experts}."
      )
    if self.moe_x_sorted != RematLocation.REMAT:
      raise ValueError(
          f"moe_x_sorted={RematLocation(self.moe_x_sorted).value} is not supported with num_moe_emb_chunks > 0; "
          "use moe_x_sorted=remat."
      )
  if self.moe_quantize_token_all_gather:
    if not self.sparse_matmul:
      raise ValueError("moe_quantize_token_all_gather=True requires sparse_matmul=True.")
    if not self.use_ring_of_experts:
      raise ValueError("moe_quantize_token_all_gather=True requires use_ring_of_experts=True.")
    if not self.use_gmm_v2:
      raise ValueError("moe_quantize_token_all_gather=True requires use_gmm_v2=True.")
    if self.num_moe_emb_chunks > 0:
      raise ValueError("moe_quantize_token_all_gather=True does not support num_moe_emb_chunks > 0.")
    if self.quantization == "" or not self.act_quantization_calibration_method.lower().startswith("fixed"):
      raise ValueError(
          "moe_quantize_token_all_gather=True requires quantization to be specified and "
          "act_quantization_calibration_method to be fixed (static scaling mode)."
      )
  if self.moe_topk_before_ep_all_gather and not (self.sparse_matmul and self.use_ring_of_experts):
    raise ValueError("moe_topk_before_ep_all_gather=True requires sparse_matmul=True and use_ring_of_experts=True.")
  if self.moe_quantize_combine_bwd_method:
    method = self.moe_quantize_combine_bwd_method
    if not self.use_ring_of_experts:
      raise ValueError(f"moe_quantize_combine_bwd_method='{method}' requires use_ring_of_experts=True.")
    if not (method == "rowwise" or method.startswith("fixed")):
      raise ValueError(
          f"Unsupported moe_quantize_combine_bwd_method: '{method}'. Supported options: '', 'rowwise', 'fixed,<bound>'."
      )
  return self


def validate_moe_retry_options(self):
  if self.retry_dropless_first_steps < 0:
    raise ValueError(f"retry_dropless_first_steps must be >= 0 (got {self.retry_dropless_first_steps}).")
  if self.retry_dropless_first_steps > 0 and self.moe_dropless_fallback != "step":
    raise ValueError("retry_dropless_first_steps > 0 requires moe_dropless_fallback='step'.")
  if self.first_phase_ragged_buffer_factor < 0:
    raise ValueError(
        f"first_phase_ragged_buffer_factor must be 0 (off) or > 0 (got {self.first_phase_ragged_buffer_factor})."
    )
  if self.first_phase_ragged_buffer_factor > 0:
    if self.retry_dropless_first_steps <= 0:
      raise ValueError("first_phase_ragged_buffer_factor > 0 requires retry_dropless_first_steps > 0.")
    if self.moe_dropless_fallback != "step":
      raise ValueError("first_phase_ragged_buffer_factor > 0 requires moe_dropless_fallback='step'.")
    if self.ragged_buffer_factor <= 0:
      raise ValueError("first_phase_ragged_buffer_factor > 0 requires ragged_buffer_factor > 0.")
    if self.first_phase_ragged_buffer_factor < self.ragged_buffer_factor:
      raise ValueError(
          f"first_phase_ragged_buffer_factor ({self.first_phase_ragged_buffer_factor}) must be >= "
          f"ragged_buffer_factor ({self.ragged_buffer_factor})."
      )
    if self.te_moe_block:
      raise ValueError("first_phase_ragged_buffer_factor > 0 is not supported with te_moe_block=True.")
  if self.log_required_ragged_buffer_factor:
    if self.te_moe_block:
      raise ValueError("log_required_ragged_buffer_factor=True is not supported with te_moe_block=True.")
    if not (self.use_ring_of_experts and self.use_ragged_sort):
      raise ValueError(
          "log_required_ragged_buffer_factor=True requires use_ring_of_experts=True and use_ragged_sort=True."
      )
    if self.num_moe_emb_chunks > 0:
      raise ValueError("log_required_ragged_buffer_factor=True does not support num_moe_emb_chunks > 0.")
  return self
