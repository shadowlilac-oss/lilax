"""Validation for MoE routing capacity and fallback settings."""


def validate_moe_dropless_fallback(self):
  """Validates prerequisites for the step-level ('step') and in-layer ('layer') dropless fallback."""
  mode = self.moe_dropless_fallback
  if mode is None:
    return
  prefix = f"moe_dropless_fallback='{mode}'"
  if self.num_experts <= 1:
    raise ValueError(f"{prefix} requires num_experts > 1.")
  if self.ragged_buffer_factor <= 0:
    raise ValueError(f"{prefix} requires ragged_buffer_factor > 0.0.")
  if mode == "step" and not self.use_ring_of_experts:
    raise ValueError(f"{prefix} is currently only supported with use_ring_of_experts=True.")
  if not self.use_ragged_sort:
    raise ValueError(f"{prefix} requires use_ragged_sort=True.")
  if self.num_moe_emb_chunks > 0:
    raise ValueError(f"{prefix} does not support num_moe_emb_chunks > 0.")
  if mode == "layer" and self.use_random_routing:
    raise ValueError(f"{prefix} does not support use_random_routing=True.")
  if mode == "step":
    if self.enable_diloco:
      raise ValueError(f"{prefix} is not supported with enable_diloco=True.")
    if self.compiled_trainstep_file:
      raise ValueError(f"{prefix} is not supported with compiled_trainstep_file.")
    if self.optimizer_memory_host_offload or self.parameter_memory_host_offload:
      raise ValueError(
          f"{prefix} is not supported with optimizer_memory_host_offload or parameter_memory_host_offload."
      )


def validate_retry_dropless_first_steps_and_first_phase_buffer(self):
  """Validates retry_dropless_first_steps, first_phase_ragged_buffer_factor and the REQUIRED_RBF probe."""
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


def validate_moe_log_max_load_ratio(self):
  """Validates that moe_log_max_load_ratio is used with the ring-of-experts ragged path."""
  if self.moe_log_max_load_ratio and not (
      self.sparse_matmul and not self.te_moe_block and self.use_ring_of_experts and self.use_ragged_sort
  ):
    raise ValueError(
        "moe_log_max_load_ratio=True requires sparse_matmul=True, te_moe_block=False,"
        " use_ring_of_experts=True and use_ragged_sort=True."
    )


def validate_moe_topk_before_ep_all_gather(self):
  """Validates that moe_topk_before_ep_all_gather is used with ring of experts."""
  if self.moe_topk_before_ep_all_gather and not (self.sparse_matmul and self.use_ring_of_experts):
    raise ValueError("moe_topk_before_ep_all_gather=True requires sparse_matmul=True and use_ring_of_experts=True.")


def validate_ragged_buffer_factor(self):
  """Validates that ragged_buffer_factor and eval_ragged_buffer_factor are used with supported settings."""
  if self.te_moe_block:
    if 0 < self.ragged_buffer_factor < 1.0 or self.eval_ragged_buffer_factor > 0:
      raise ValueError(
          "te_moe_block=True requires ragged_buffer_factor >= 1.0, or <= 0 for worst-case capacity, "
          "and does not support eval_ragged_buffer_factor > 0."
      )
    return

  if self.ragged_buffer_factor <= 0 and self.eval_ragged_buffer_factor <= 0:
    return

  if self.use_ring_of_experts and not self.use_ragged_sort:
    raise ValueError(
        "Ragged buffer factor is currently only supported with:\n"
        "  1. Ragged A2A approach (use_ring_of_experts=False)\n"
        "  2. Ragged sort with ring of experts (use_ring_of_experts=True AND use_ragged_sort=True)"
    )
