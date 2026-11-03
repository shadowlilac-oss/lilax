from lilax.configuration.domain_types import DecoderBlockType, RopeType
from lilax.configuration.domains.model_moe import RematLocation, TEGroupedGemmQuantizationType


def validate_expert_options(self) -> "MaxTextConfig":
  if self.enable_diloco and self.use_tokamax_gmm:
    raise ValueError(
        "use_tokamax_gmm=True is not compatible with enable_diloco=True due to a known "
        "incompatibility between tokamax's GroupSizes vmap_rule and JAX's scan batching. "
        "Please set use_tokamax_gmm=False."
    )
  if self.defer_small_all_reduces:
    if not (self.use_ring_of_experts and self.sparse_matmul):
      raise ValueError(
          "defer_small_all_reduces requires use_ring_of_experts=True and sparse_matmul=True (the deferral is"
          " implemented on the ring-of-experts sparse_matmul path only)."
      )
    if self.moe_dropless_fallback == "layer":
      raise ValueError(
          "defer_small_all_reduces does not support moe_dropless_fallback='layer' (the in-layer fallback's two"
          " branches would emit per-chunk counts of different chunkings)."
      )
  if self.use_lineage:
    if not self.scan_layers:
      raise ValueError("use_lineage=True requires scan_layers=True.")
    if self.decoder_block != DecoderBlockType.DEEPSEEK:
      raise ValueError(f"use_lineage=True requires decoder_block='deepseek', got decoder_block={self.decoder_block!r}.")
    if self.attention_type != "mla":
      raise ValueError(f"use_lineage=True requires attention_type='mla', got attention_type={self.attention_type!r}.")
    if self.rope_type != RopeType.YARN:
      raise ValueError(f"use_lineage=True requires rope_type='yarn', got rope_type={self.rope_type!r}.")
    if self.capacity_factor <= 0:
      raise ValueError(f"use_lineage=True requires capacity_factor > 0, got capacity_factor={self.capacity_factor}.")
  if self.num_experts > 1:
    if self.decoder_block == DecoderBlockType.GPT_OSS and not self.sparse_matmul and self.capacity_factor != -1:
      raise ValueError("GPT-OSS MoE only supports dropless (capacity_factor=-1) with dense matmul.")
    if (
        self.routed_bias and self.routed_bias_update_rate > 0.0
        and self.decoder_block not in (DecoderBlockType.DEEPSEEK, DecoderBlockType.DEEPSEEK4)
    ):
      raise ValueError("Loss-free load balancing is only supported for the DeepSeek decoder block.")
    if self.te_moe_block and not self.sparse_matmul:
      raise ValueError("te_moe_block=True requires sparse_matmul=True.")
    if self.te_moe_quantize_before_fsdp_all_gather:
      if not self.te_moe_block or self.te_gmm_quantization != TEGroupedGemmQuantizationType.TE_MXFP8:
        raise ValueError(
            "te_moe_quantize_before_fsdp_all_gather=True requires te_moe_block=True and te_gmm_quantization=te_mxfp8."
        )
      if self.use_2d_fsdp_sharding:
        raise ValueError("te_moe_quantize_before_fsdp_all_gather=True does not support 2D FSDP weight sharding.")
    if self.te_moe_block and not self.prefuse_moe_weights:
      raise ValueError("te_moe_block=True requires prefuse_moe_weights=True.")
    if self.te_moe_block and self.routed_bias_update_rate > 0.0:
      raise ValueError("te_moe_block=True does not currently support routed_bias_update_rate > 0.")
    if self.te_moe_block and self.norm_topk_prob:
      raise ValueError("te_moe_block=True does not currently support norm_topk_prob=True.")
    if self.te_moe_block and self.use_random_routing:
      raise ValueError("te_moe_block=True does not support use_random_routing=True.")
    if self.te_moe_block and self.decoder_block == DecoderBlockType.LLAMA4:
      raise ValueError("te_moe_block=True does not currently support Llama4 routing semantics.")
    if self.te_moe_block and not self.te_gmm_quantization:
      raise ValueError(
          "te_gmm_quantization must be specified when te_moe_block=True. "
          "te_gmm_quantization=te_no_quant is supported for BF16."
      )
    if self.model_name.startswith("deepseek4") and self.first_num_hash_layers > 0 and self.use_ring_of_experts:
      raise ValueError("DeepSeek V4 hash routing is currently not supported with ring of experts.")
    if self.te_moe_block:
      if 0 < self.ragged_buffer_factor < 1.0 or self.eval_ragged_buffer_factor > 0:
        raise ValueError(
            "te_moe_block=True requires ragged_buffer_factor >= 1.0, or <= 0 for worst-case capacity, "
            "and does not support eval_ragged_buffer_factor > 0."
        )
    elif (self.ragged_buffer_factor > 0 or self.eval_ragged_buffer_factor > 0) and (
        self.use_ring_of_experts and not self.use_ragged_sort
    ):
      raise ValueError(
          "Ragged buffer factor is currently only supported with:\n"
          "  1. Ragged A2A approach (use_ring_of_experts=False)\n"
          "  2. Ragged sort with ring of experts (use_ring_of_experts=True AND use_ragged_sort=True)"
      )
    if self.moe_dropless_fallback is not None:
      mode = self.moe_dropless_fallback
      prefix = f"moe_dropless_fallback='{mode}'"
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
    if self.moe_log_max_load_ratio and not (
        self.sparse_matmul and not self.te_moe_block and self.use_ring_of_experts and self.use_ragged_sort
    ):
      raise ValueError(
          "moe_log_max_load_ratio=True requires sparse_matmul=True, te_moe_block=False,"
          " use_ring_of_experts=True and use_ragged_sort=True."
      )
    if self.moe_tc_ragged_3d_dispatch and not (self.moe_tc_ragged_sort and self.moe_tc_ragged_3d_gmm):
      raise ValueError("moe_tc_ragged_3d_dispatch=True requires moe_tc_ragged_sort=True and moe_tc_ragged_3d_gmm=True.")
    if self.moe_tc_routing_checkpoint != RematLocation.REMAT and not (
        self.moe_tc_ragged_sort and self.moe_tc_ragged_weights_on_activation
    ):
      raise ValueError(
          f"moe_tc_routing_checkpoint={RematLocation(self.moe_tc_routing_checkpoint).value} requires"
          " moe_tc_ragged_sort=True and moe_tc_ragged_weights_on_activation=True; use moe_tc_routing_checkpoint=remat."
      )
    if self.moe_tc_window_routing and not self.moe_tc_ragged_sort:
      raise ValueError("moe_tc_window_routing=True requires moe_tc_ragged_sort=True.")
    if self.moe_tc_ragged_sort:
      if not self.use_ring_of_experts:
        raise ValueError("moe_tc_ragged_sort=True requires use_ring_of_experts=True.")
      if not self.use_ragged_sort:
        raise ValueError("moe_tc_ragged_sort=True requires use_ragged_sort=True.")
      if self.ragged_buffer_factor <= 0:
        raise ValueError("moe_tc_ragged_sort=True requires ragged_buffer_factor > 0.0.")
      if self.moe_tc_ragged_3d_gmm:
        if not (self.use_tokamax_gmm and self.use_gmm_v2):
          raise ValueError("moe_tc_ragged_3d_gmm=True requires use_tokamax_gmm=True and use_gmm_v2=True.")
        unsupported = ["mlp_bias"] if self.mlp_bias else []
        if self.ici_tensor_parallelism > 1 or self.dcn_tensor_parallelism > 1:
          unsupported.append("tensor parallelism")
        if self.num_moe_emb_chunks > 0:
          unsupported.append("num_moe_emb_chunks > 0")
        if unsupported:
          raise ValueError(f"moe_tc_ragged_3d_gmm=True does not support: {', '.join(unsupported)}.")
  if self.warmup_skip_train_dropless and not self.warmup_programs_in_init:
    raise ValueError("warmup_skip_train_dropless=True requires warmup_programs_in_init=True.")
  return self
