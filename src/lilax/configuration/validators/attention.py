import copy

from lilax.configuration.domain_types import DecoderBlockType


def validate_attention_options(self) -> "MaxTextConfig":
  if self.global_parameter_scale == 0:
    raise ValueError("Global parameter scale cannot be zero.")
  if self.global_parameter_scale < 0:
    raise ValueError("math domain error")
  if self.global_parameter_scale & (self.global_parameter_scale - 1):
    raise ValueError(
        "Global parameter scale should be a power of two. If you want finer grained control of the model sizes "
        "then you can explicitly set base_embed_dim, base_num_query_heads, base_num_kv_heads, "
        "base_mlp_dim, base_num_decoder_layers and/or head_dim."
    )
  base_scale, remainder = divmod(self.global_parameter_scale.bit_length() - 1, 3)
  head_scale = base_scale + int(remainder > 0)
  self.emb_dim = (2 ** (base_scale + int(remainder > 1))) * self.base_emb_dim
  self.num_query_heads = (2 ** head_scale) * self.base_num_query_heads
  self.num_kv_heads = (2 ** head_scale) * self.base_num_kv_heads
  self.mlp_dim = (2 ** head_scale) * self.base_mlp_dim
  self.moe_mlp_dim = (2 ** head_scale) * self.base_moe_mlp_dim
  self.num_decoder_layers = (2 ** base_scale) * self.base_num_decoder_layers
  if self.pipeline_parallel_layers == -1:
    self.pipeline_parallel_layers = (
        self.num_decoder_layers - self.first_num_dense_layers
        if self.decoder_block == DecoderBlockType.DEEPSEEK else self.num_decoder_layers
    )
  self.using_pipeline_parallelism = self.ici_pipeline_parallelism > 1 or self.dcn_pipeline_parallelism > 1
  if self.using_pipeline_parallelism:
    num_stages = int(self.ici_pipeline_parallelism * self.dcn_pipeline_parallelism)
    if self.num_pipeline_repeats == -1:
      repeats, remainder = divmod(self.pipeline_parallel_layers, num_stages * self.num_layers_per_pipeline_stage)
      assert not remainder, (
          f"The number of layers per stage ({self.num_layers_per_pipeline_stage}) times the number of stages "
          f"({num_stages}) must divide the number of pipeline_parallel_layers which defaults to decoder layers "
          f"({self.pipeline_parallel_layers}) "
      )
      self.num_pipeline_repeats = repeats
    if self.pipeline_fsdp_ag_per_repeat:
      assert self.num_pipeline_repeats > 1, "Pipeline weight prefetching only supports circular pipeline."
      assert self.num_layers_per_pipeline_stage == 1, (
          "Pipeline weight prefetching currently only supports one layer per pipeline stage."
      )
      assert not self.pipeline_delay_activation_forwarding, "Pipeline weight prefetching does not support pipeline delay."
      assert not self.scan_layers_per_stage, "Pipeline weight prefetching currently does not support scan."
      qwix_qt_types = ("int4", "int8", "fp4", "fp4_e2m1", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_full")
      assert not self.quantization or (self.use_qwix_quantization and self.quantization in qwix_qt_types), (
          "Pipeline weight prefetching (pipeline_fsdp_ag_per_repeat) only supports Qwix quantized training: "
          f"use_qwix_quantization=True and quantization in {qwix_qt_types}."
      )
    else:
      bsw_options = ("pipeline_bsw_phase_split", "pipeline_bsw_in_compute_dtype")
      enabled = [name for name in bsw_options if getattr(self, name)]
      assert not enabled, f"{enabled} only apply to pipeline weight prefetching: set pipeline_fsdp_ag_per_repeat=True."
    assert num_stages * self.num_pipeline_repeats * self.num_layers_per_pipeline_stage == self.pipeline_parallel_layers, (
        f"The product of pipeline stages ({num_stages}), repeats ({self.num_pipeline_repeats}), and layers "
        f"per stage ({self.num_layers_per_pipeline_stage}) must be equal to pipeline_parallel_layers "
        f"which defaults to decoder layers ({self.pipeline_parallel_layers})"
    )
    if self.num_pipeline_microbatches == -1:
      self.num_pipeline_microbatches = 2 * num_stages if self.pipeline_delay_activation_forwarding else num_stages
    assert self.num_pipeline_microbatches > 0, "num_pipeline_microbatches must be positive"
    assert self.num_pipeline_microbatches % num_stages == 0, (
        f"The number of microbatches ({self.num_pipeline_microbatches}) must be divisible by the number of "
        f"stages ({num_stages})"
    )
    if self.pipeline_delay_activation_forwarding:
      assert self.num_pipeline_microbatches >= 2 * num_stages, (
          f"Delayed activation forwarding requires at least 2 * num_stages microbatches, but {num_stages} stages "
          f"are used with {self.num_pipeline_microbatches} microbatches"
      )
    self._pipeline_input_layout = copy.deepcopy({
        "logical_axis_rules": self.logical_axis_rules,
        "mesh_axes": self.mesh_axes,
        "data_sharding": self.data_sharding,
    })
    for rule in self.logical_axis_rules:
      if rule and rule[0] == "activation_embed_and_logits_batch":
        rule[1] = ["stage", "data", "fsdp", "fsdp_transpose", "expert"]
        break
    if "stage" in self.mesh_axes:
      stage_idx = self.mesh_axes.index("stage")
      data_idx = self.mesh_axes.index("data")
      if stage_idx > data_idx:
        self.mesh_axes[stage_idx], self.mesh_axes[data_idx] = self.mesh_axes[data_idx], self.mesh_axes[stage_idx]
    if (
        isinstance(self.data_sharding, list) and self.data_sharding and self.data_sharding[0]
        and "stage" in self.data_sharding[0]
    ):
      self.data_sharding[0].remove("stage")
      self.data_sharding[0].insert(0, "stage")
    if self.quantization and self.quantization in ("fp8", "nanoo_fp8", "fp8_gpu", "te_fp8_delayedscaling"):
      self.logical_axis_rules.append(["aqt_amax_history", ("stage",)])
  num_query_heads = self.num_query_heads
  if self.engram_layers:
    if not self.hf_access_token or not self.tokenizer_path:
      raise ValueError("Engram requires both 'hf_access_token' and 'tokenizer_path' to load the Hugging Face tokenizer.")
    if len(self.engram_vocab_bases) != self.engram_max_ngram_size - 1:
      raise ValueError(
          f"Engram vocab size mismatch: expected {self.engram_max_ngram_size - 1} (max_ngram_size - 1), "
          f"but got {self.engram_vocab_bases}."
      )
  if self.num_experts > 1:
    if self.moe_mlp_dim <= 0:
      raise ValueError("moe_mlp_dim must be positive for MoE models (num_experts > 1)")
    if (
        self.interleave_moe_layer_step == 1 and self.first_num_dense_layers == 0
        and self.inhomogeneous_layer_cycle_interval == 1 and self.base_mlp_dim != self.base_moe_mlp_dim
    ):
      if self.decoder_block == DecoderBlockType.QWEN3_MOE:
        self.base_mlp_dim = self.base_moe_mlp_dim
        self.mlp_dim = (2 ** head_scale) * self.base_mlp_dim
      elif self.decoder_block != DecoderBlockType.GEMMA4:
        raise ValueError(
            "For a fully MoE model, base_mlp_dim must equal base_moe_mlp_dim. "
            f"Got base_mlp_dim={self.base_mlp_dim}, base_moe_mlp_dim={self.base_moe_mlp_dim}."
        )
  if self.enable_streaming_diloco:
    if not self.scan_layers:
      raise ValueError("enable_streaming_diloco=True requires scan_layers=True.")
    if self.num_diloco_fragments is not None and self.num_diloco_fragments > 1:
      num_transformer_fragments = self.num_diloco_fragments - 1
      if self.num_decoder_layers % num_transformer_fragments != 0:
        raise ValueError(
            f"The number of decoder layers ({self.num_decoder_layers}) must be divisible by "
            f"(num_diloco_fragments - 1) ({num_transformer_fragments}) when enable_streaming_diloco is True."
        )
    if self.diloco_bucketize_non_scanned and self.shard_optimizer_over_data:
      raise ValueError("diloco_bucketize_non_scanned does not support shard_optimizer_over_data (Zero-1).")
  if self.context_parallel_strategy.lower() not in ("all_gather", "ring", "ulysses", "usp"):
    raise ValueError("context_parallel_strategy must be one of 'all_gather', 'ring', 'ulysses', or 'usp'.")
  if self.sa_qk_diag_grid < 2 or (self.sa_qk_diag_grid & (self.sa_qk_diag_grid - 1)) != 0:
    raise ValueError(f"sa_qk_diag_grid must be a power of 2 and >= 2, got {self.sa_qk_diag_grid}.")
  if self.attention_type == "compressed" and self.dq_reduction_steps not in (0, 3):
    raise ValueError(
        f"attention_type='compressed' requires dq_reduction_steps to be 0 or 3, got {self.dq_reduction_steps}."
    )
  if self.moba and self.attention not in ("dot_product"):
    raise ValueError("MoBA is only supported with dot_product attention.")
  if self.decoder_block == DecoderBlockType.DEEPSEEK4 and not (
      self.attention == "dot_product" or (self.attention == "flash" and self.use_tokamax_splash)
  ):
    raise ValueError(
        "DeepSeek4 is only supported with `dot_product` attention or `flash` attention "
        "with `use_tokamax_splash=True`."
    )
  if self.model_name in ("gemma4-e2b", "gemma4-e4b") and self.scan_layers:
    raise ValueError(
        f"{self.model_name} requires scan_layers=False (per-layer KV sharing is incompatible with scanned layers)."
    )
  if self.decoder_block == DecoderBlockType.LLAMA4:
    if self.capacity_factor >= 0:
      raise ValueError(
          "Llama4 decoder has not been tested with capacity_factor >= 0 -- please set that value to -1 for now!"
      )
    if self.num_experts_per_tok > 1:
      raise ValueError("Only top-1 routing is supported for Llama4 for now!")
  if self.decoder_block in (DecoderBlockType.LLAMA4, DecoderBlockType.ENVY):
    if self.base_num_decoder_layers % self.interleave_moe_layer_step != 0:
      raise ValueError(
          f"The number of decoder layers ({self.base_num_decoder_layers}) must be divisible by interleave moe layer step "
          f"({self.interleave_moe_layer_step})"
      )
  if self.decoder_block in (DecoderBlockType.QWEN3_NEXT, DecoderBlockType.QWEN3_5):
    if int(self.gdn_num_value_heads) % int(self.gdn_num_key_heads) != 0:
      raise ValueError("gdn_num_value_heads must be divisible by gdn_num_key_heads")
    rotary_dim = int(self.head_dim * self.partial_rotary_factor)
    if rotary_dim % 2 != 0:
      raise ValueError(f"Calculated rotary dimension ({rotary_dim}) must be a multiple of 2.")
    if self.ici_context_parallelism * self.dcn_context_parallelism > 1 and self.context_parallel_load_balance:
      raise ValueError(
          "GatedDeltaNet context parallelism requires context_parallel_load_balance=False. The GatedDeltaNet "
          "layers carry a recurrence, so device order is sequence order: device i composes the state left by "
          "device i-1. DUAL_CHUNK_SWAP hands device 0 the first and last chunks, device 1 the second and "
          "second-to-last, and so on, which composes the segments out of order. Softmax attention tolerates the "
          "reorder because it rebuilds the causal mask from positions; a recurrence cannot. The run still trains "
          "and the loss still falls, so set this explicitly rather than relying on the failure being visible."
      )
  elif self.partial_rotary_factor is not None and self.partial_rotary_factor != 1.0:
    raise ValueError("`partial_rotary_factor` is only effective when `decoder_block` is set to 'qwen3_next'.")
  if self.model_name == "glm5.3-flash" and not self.use_m3_model:
    raise ValueError("glm5.3-flash requires use_m3_model=True; legacy decoder_block is not supported for GLM-5.3-Flash.")
  if self.decoder_block == DecoderBlockType.QWEN3_CUSTOM_MOE and self.moe_expert_input_dim != self.attention_output_dim:
    raise ValueError(
        f"For qwen3_custom_moe, moe_expert_input_dim ({self.moe_expert_input_dim}) "
        f"must be equal to attention_output_dim ({self.attention_output_dim})"
    )
  if self.te_mhc and self.hardware not in ("gpu", "gpu_multiprocess"):
    raise ValueError("`te_mhc=True` requires NVIDIA GPU hardware.")
  for name in ("mla_qk_head_chunk_size", "csa_qk_head_chunk_size"):
    chunk_size = getattr(self, name)
    if chunk_size > 0 and (chunk_size > num_query_heads or num_query_heads % chunk_size != 0):
      raise ValueError(
          f"`{name}` ({chunk_size}) must cleanly divide exactly into `num_query_heads` ({num_query_heads})."
      )
    if chunk_size > 0 and self.use_indexer and (
        chunk_size > self.indexer_n_heads or self.indexer_n_heads % chunk_size != 0
    ):
      raise ValueError(
          f"`{name}` ({chunk_size}) must cleanly divide exactly into `indexer_n_heads` ({self.indexer_n_heads})."
      )
  return self
