from lilax.configuration.domains.execution_mesh import CustomRule, ReorderStrategy


def validate_context_parallel_options(self) -> "MaxTextConfig":
  strategy = self.context_parallel_strategy.lower()
  self.context_parallel_strategy = strategy
  if self.ulysses_context_sharding != "context_usp_ulysses":
    raise ValueError(f"Assigned ulysses_context_sharding {self.ulysses_context_sharding} is not supported.")
  if self.custom_mesh_and_rule == CustomRule.DEFAULT and self.context_sharding not in ("context", "expert"):
    raise ValueError(f"Assigned context_sharding f{self.context_sharding} is not supported.")
  if (
      self.per_device_batch_size > 0
      and (self.per_device_batch_size * self.max_target_length) % self.num_vocab_tiling != 0
  ):
    raise ValueError("Per device batch size times sequence length should be divisible by the number of vocab tiles.")
  if self.custom_mesh_and_rule == CustomRule.DEFAULT and self.context_sharding in ("context", "expert"):
    context_size = getattr(self, f"ici_{self.context_sharding}_parallelism") * getattr(
        self, f"dcn_{self.context_sharding}_parallelism"
    )
    if self.attention_type == "compressed" and any(ratio > 0 for ratio in self.compress_ratios) and context_size > 1:
      raise ValueError(
          f"Context parallelism (context_parallel_size={context_size}) is not supported with "
          "attention_type='compressed' and a non-zero compress_ratio. Set every entry of compress_ratios "
          "to 0, or disable context parallelism."
      )
    if strategy == "ring" and "gpu" not in self.hardware and "tpu" not in self.hardware and context_size > 1:
      raise ValueError(
          "Ring context parallelism strategy (context_parallel_strategy='ring') is only supported on GPUs "
          "or TPU with attention=flash and use_tokamax_splash=True."
      )
    if strategy == "ring" and "gpu" not in self.hardware and "tpu" in self.hardware:
      if context_size <= 1:
        raise ValueError("TPU Tokamax ring attention requires context_parallel_size > 1.")
      if self.context_sharding != "context":
        raise ValueError("TPU Tokamax ring attention requires context_sharding='context'.")
      if self.max_target_length % (context_size * context_size) != 0:
        raise ValueError(
            "TPU Tokamax ring attention requires max_target_length to be divisible by context_parallel_size squared."
        )
      if self.context_parallel_load_balance and context_size % 2 != 0:
        raise ValueError("TPU Tokamax ring load balancing requires an even context_parallel_size.")
    if (
        context_size > 1 and "gpu" not in self.hardware and self.context_parallel_load_balance
        and self.context_parallel_reorder_strategy == ReorderStrategy.STRIPED
    ):
      raise ValueError(
          "STRIPED reorder strategy requires Transformer Engine and is only supported on GPUs. "
          f"Got hardware={self.hardware!r}."
      )
  if self.attention_type == "compressed" and self.dq_reduction_steps not in (0, 3):
    raise ValueError(
        f"attention_type='compressed' requires dq_reduction_steps to be 0 or 3, got {self.dq_reduction_steps}."
    )
  if strategy not in ("ring", "usp") and self.ring_scan_unroll != 1:
    raise ValueError(
        f"ring_scan_unroll={self.ring_scan_unroll} was specified, but is only supported when "
        "context_parallel_strategy='ring' or 'usp'."
    )
  if strategy != "usp" and (
      self.ici_context_usp_ulysses_parallelism != 1 or self.dcn_context_usp_ulysses_parallelism != 1
  ):
    raise ValueError(
        "ici/dcn_context_usp_ulysses_parallelism was specified, but is only supported when "
        "context_parallel_strategy='usp'."
    )
  if strategy == "ring" and "gpu" not in self.hardware and "tpu" in self.hardware:
    if self.use_indexer and self.dq_reduction_steps != 0:
      raise ValueError("TPU Tokamax ring attention with sparse indexer mask only supports dq_reduction_steps=0.")
    if self.dq_reduction_steps not in (0, 3):
      raise ValueError("TPU Tokamax ring attention requires dq_reduction_steps to be 0 or 3.")
    if self.attention != "flash":
      raise ValueError("TPU ring context parallelism requires attention=flash.")
    if not self.use_tokamax_splash:
      raise ValueError("TPU ring context parallelism requires use_tokamax_splash=True.")
    if self.use_jax_splash:
      raise ValueError("TPU ring context parallelism requires use_jax_splash=False.")
    if self.attention_type not in ("global", "mla"):
      raise ValueError("TPU Tokamax ring attention supports only attention_type='global' or 'mla'.")
    if self.attention_type == "mla" and self.packing:
      raise ValueError("TPU Tokamax ring attention with MLA does not support packing yet.")
    if self.attention_type == "mla" and self.use_batch_split_schedule:
      raise ValueError("TPU Tokamax ring attention with MLA does not support the DeepSeek batch-split schedule.")
    if self.context_parallel_load_balance and self.mtp_num_layers > 0:
      raise ValueError("TPU Tokamax ring attention with context_parallel_load_balance=True does not support MTP.")
    if self.use_ragged_attention:
      raise ValueError("TPU Tokamax ring attention does not support ragged attention.")
    if self.attention_sink:
      raise ValueError("TPU Tokamax ring attention does not support attention sinks.")
    if self.use_chunked_prefill:
      raise ValueError("TPU Tokamax ring attention does not support chunked prefill yet.")
    if self.moba:
      raise ValueError("TPU Tokamax ring attention does not support MoBA.")
    if self.use_multimodal:
      raise ValueError("TPU Tokamax ring attention does not support multimodal attention.")
    if self.use_qk_clip:
      raise ValueError("TPU Tokamax ring attention does not support QK-Clip statistics yet.")
    if self.enable_dropout and self.dropout_rate > 0.0:
      raise ValueError("TPU Tokamax ring attention does not support dropout yet.")
  if strategy == "ulysses":
    if self.hardware != "tpu":
      raise ValueError("Ulysses context parallelism (context_parallel_strategy='ulysses') is only supported on TPU.")
    if self.custom_mesh_and_rule == CustomRule.DEFAULT and self.context_sharding != "context":
      raise ValueError("TPU Ulysses attention requires context_sharding='context'.")
    context_size = self.ici_context_parallelism * self.dcn_context_parallelism
    if self.ici_context_parallelism <= 0 or self.dcn_context_parallelism <= 0:
      raise ValueError(
          "TPU Ulysses attention requires explicit positive ici/dcn context parallelism values; "
          "inferred (-1) sizes are not supported."
      )
    if context_size <= 1:
      raise ValueError("TPU Ulysses attention requires context_parallel_size > 1.")
    if self.dcn_context_parallelism != 1:
      raise ValueError("TPU Ulysses attention does not support dcn context parallelism yet.")
    if self.attention != "flash":
      raise ValueError("TPU Ulysses attention requires attention=flash.")
    if not self.use_tokamax_splash:
      raise ValueError("TPU Ulysses attention requires use_tokamax_splash=True.")
    if self.use_jax_splash:
      raise ValueError("TPU Ulysses attention requires use_jax_splash=False.")
    if self.use_indexer:
      raise ValueError("TPU Ulysses attention does not support sparse indexer masks.")
    if self.attention_type != "global":
      raise ValueError("TPU Ulysses attention is initially supported only for global causal attention.")
    if self.context_parallel_load_balance:
      raise ValueError(
          "TPU Ulysses attention requires context_parallel_load_balance=False: after the all-to-all every device "
          "attends over the full sequence in natural order, so the load-balancing reorder is unnecessary and "
          "would corrupt the causal mask."
      )
    if self.use_ragged_attention:
      raise ValueError("TPU Ulysses attention does not support ragged attention.")
    if self.attention_sink:
      raise ValueError("TPU Ulysses attention does not support attention sinks.")
    if self.use_chunked_prefill:
      raise ValueError("TPU Ulysses attention does not support chunked prefill yet.")
    if self.use_multimodal:
      raise ValueError("TPU Ulysses attention does not support multimodal attention.")
    if self.enable_dropout and self.dropout_rate > 0.0:
      raise ValueError("TPU Ulysses attention does not support dropout yet.")
    if self.dq_reduction_steps not in (0, 3):
      raise ValueError("TPU Ulysses attention requires dq_reduction_steps to be 0 or 3.")
    if self.use_qk_clip:
      raise ValueError("TPU Ulysses attention does not support QK-Clip statistics yet.")
    if self.max_target_length % context_size != 0:
      raise ValueError(
          "TPU Ulysses attention requires max_target_length "
          f"({self.max_target_length}) to be divisible by context_parallel_size ({context_size})."
      )
    base_scale, remainder = divmod(self.global_parameter_scale.bit_length() - 1, 3)
    head_scale = base_scale + int(remainder > 0)
    num_query_heads = (2 ** head_scale) * self.base_num_query_heads
    num_kv_heads = (2 ** head_scale) * self.base_num_kv_heads
    if num_query_heads % context_size != 0:
      raise ValueError(
          "TPU Ulysses attention requires num_query_heads "
          f"({num_query_heads}) to be divisible by context_parallel_size ({context_size})."
      )
    if num_kv_heads == 1:
      raise ValueError("TPU Ulysses attention does not support MQA with context_parallel_size > 1.")
    if num_kv_heads % context_size != 0:
      raise ValueError(
          "TPU Ulysses attention requires num_kv_heads "
          f"({num_kv_heads}) to be divisible by context_parallel_size ({context_size})."
      )
  if strategy == "usp":
    if self.hardware != "tpu":
      raise ValueError("USP context parallelism (context_parallel_strategy='usp') is only supported on TPU.")
    if self.custom_mesh_and_rule == CustomRule.DEFAULT and self.context_sharding != "context":
      raise ValueError("TPU USP attention requires context_sharding='context'.")
    if self.custom_mesh_and_rule == CustomRule.DEFAULT and self.custom_mesh_and_rule_for_eval == CustomRule.DEFAULT:
      usp_sequence_axes = (self.context_sharding, self.ulysses_context_sharding)
      for axis in usp_sequence_axes:
        if axis not in self.mesh_axes:
          raise ValueError(f"TPU USP attention requires mesh axis '{axis}' in mesh_axes.")
      axes = next(
          (rule[1] for rule in self.logical_axis_rules if rule and len(rule) >= 2 and rule[0] == "activation_length"),
          None,
      )
      if isinstance(axes, str):
        axes = (axes,)
      elif isinstance(axes, (list, tuple)):
        axes = tuple(axis for axis in axes if isinstance(axis, str))
      else:
        axes = ()
      if axes != usp_sequence_axes:
        raise ValueError(
            f"TPU USP attention requires activation_length to map to {usp_sequence_axes} in logical_axis_rules."
        )
    ring_size = self.ici_context_parallelism
    ulysses_size = self.ici_context_usp_ulysses_parallelism
    if (
        ring_size <= 0 or ulysses_size <= 0
        or self.dcn_context_parallelism <= 0 or self.dcn_context_usp_ulysses_parallelism <= 0
    ):
      raise ValueError(
          "TPU USP attention requires explicit positive ici/dcn context parallelism values; "
          "inferred (-1) sizes are not supported."
      )
    if ring_size <= 1:
      raise ValueError("TPU USP attention requires ici_context_parallelism > 1 for the ring dimension.")
    if ulysses_size <= 1:
      raise ValueError("TPU USP attention requires ici_context_usp_ulysses_parallelism > 1 for the Ulysses dimension.")
    if self.dcn_context_parallelism != 1 or self.dcn_context_usp_ulysses_parallelism != 1:
      raise ValueError("TPU USP attention does not support dcn context parallelism yet.")
    if self.attention != "flash":
      raise ValueError("TPU USP attention requires attention=flash.")
    if not self.use_tokamax_splash:
      raise ValueError("TPU USP attention requires use_tokamax_splash=True.")
    if self.use_jax_splash:
      raise ValueError("TPU USP attention requires use_jax_splash=False.")
    if self.use_indexer:
      raise ValueError("TPU USP attention does not support sparse indexer masks.")
    if self.attention_type != "global":
      raise ValueError("TPU USP attention is initially supported only for global causal attention.")
    if self.context_parallel_load_balance and ring_size % 2 != 0:
      raise ValueError("TPU USP attention with context_parallel_load_balance=True requires an even ici_context_parallelism.")
    if self.use_ragged_attention:
      raise ValueError("TPU USP attention does not support ragged attention.")
    if self.attention_sink:
      raise ValueError("TPU USP attention does not support attention sinks.")
    if self.use_chunked_prefill:
      raise ValueError("TPU USP attention does not support chunked prefill yet.")
    if self.use_multimodal:
      raise ValueError("TPU USP attention does not support multimodal attention.")
    if self.enable_dropout and self.dropout_rate > 0.0:
      raise ValueError("TPU USP attention does not support dropout yet.")
    if self.dq_reduction_steps not in (0, 3):
      raise ValueError("TPU USP attention requires dq_reduction_steps to be 0 or 3.")
    if self.use_qk_clip:
      raise ValueError("TPU USP attention does not support QK-Clip statistics yet.")
    if self.mtp_num_layers > 0:
      raise ValueError("TPU USP attention does not support multi-token prediction (mtp_num_layers > 0) yet.")
    if self.sa_bwd_dkv_megacore:
      raise ValueError("TPU USP attention does not support sa_bwd_dkv_megacore yet.")
    if self.max_target_length % (ring_size * ulysses_size) != 0:
      raise ValueError(
          "TPU USP attention requires max_target_length "
          f"({self.max_target_length}) to be divisible by the total context parallelism ({ring_size * ulysses_size})."
      )
    if self.max_target_length % (ring_size * ring_size) != 0:
      raise ValueError("TPU USP attention requires max_target_length to be divisible by ici_context_parallelism squared.")
    base_scale, remainder = divmod(self.global_parameter_scale.bit_length() - 1, 3)
    head_scale = base_scale + int(remainder > 0)
    num_query_heads = (2 ** head_scale) * self.base_num_query_heads
    num_kv_heads = (2 ** head_scale) * self.base_num_kv_heads
    if num_query_heads % ulysses_size != 0:
      raise ValueError(
          "TPU USP attention requires num_query_heads "
          f"({num_query_heads}) to be divisible by ici_context_usp_ulysses_parallelism ({ulysses_size})."
      )
    if num_kv_heads == 1:
      raise ValueError("TPU USP attention does not support MQA with ici_context_usp_ulysses_parallelism > 1.")
    if num_kv_heads % ulysses_size != 0:
      raise ValueError(
          "TPU USP attention requires num_kv_heads "
          f"({num_kv_heads}) to be divisible by ici_context_usp_ulysses_parallelism ({ulysses_size})."
      )
  return self
