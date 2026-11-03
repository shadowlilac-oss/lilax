"""Validation of native context parallel configuration."""

from lilax._backend.common.common_types import AttentionType, ReorderStrategy


def validate_context_parallelism(config, context_parallel_size) -> None:
  if config.attention_type == AttentionType.COMPRESSED.value:
    # `compress_ratios` is per-layer and a ratio of 0 downgrades that layer to local sliding
    # attention (see CompressedAttention.__init__), so context parallelism stays legal when no
    # layer is actually compressed. `compress_ratios` defaults to [] for every other model.
    if any(ratio > 0 for ratio in config.compress_ratios) and context_parallel_size > 1:
      raise ValueError(
          f"Context parallelism (context_parallel_size={context_parallel_size}) is not supported with "
          "attention_type='compressed' and a non-zero compress_ratio. Set every entry of compress_ratios "
          "to 0, or disable context parallelism."
      )
    # The Tokamax Splash backward kernel only accepts dq_reduction_steps of 3 or None; 0 means
    # "unset" here and is mapped to the kernel default downstream.
    if config.dq_reduction_steps not in (0, 3):
      raise ValueError(
          f"attention_type='compressed' requires dq_reduction_steps to be 0 or 3, got {config.dq_reduction_steps}."
      )
  context_parallel_strategy = config.context_parallel_strategy.lower()
  if context_parallel_strategy not in ("all_gather", "ring", "ulysses", "usp"):
    raise ValueError("context_parallel_strategy must be one of 'all_gather', 'ring', 'ulysses', or 'usp'.")
  config.context_parallel_strategy = context_parallel_strategy
  if (
      context_parallel_strategy == "ring"
      and "gpu" not in config.hardware
      and "tpu" not in config.hardware
      and context_parallel_size > 1
  ):
    raise ValueError(
        "Ring context parallelism strategy (context_parallel_strategy='ring') is only supported on GPUs "
        "or TPU with attention=flash and use_tokamax_splash=True."
    )
  if context_parallel_strategy == "ring" and "gpu" not in config.hardware and "tpu" in config.hardware:
    if context_parallel_size <= 1:
      raise ValueError("TPU Tokamax ring attention requires context_parallel_size > 1.")
    if config.context_sharding != "context":
      raise ValueError("TPU Tokamax ring attention requires context_sharding='context'.")
    if config.use_indexer and config.dq_reduction_steps != 0:
      raise ValueError("TPU Tokamax ring attention with sparse indexer mask only supports dq_reduction_steps=0.")
    if config.dq_reduction_steps not in (0, 3):
      raise ValueError("TPU Tokamax ring attention requires dq_reduction_steps to be 0 or 3.")
    if config.max_target_length % (context_parallel_size * context_parallel_size) != 0:
      raise ValueError(
          "TPU Tokamax ring attention requires max_target_length to be divisible by context_parallel_size squared."
      )
    if config.attention != "flash":
      raise ValueError("TPU ring context parallelism requires attention=flash.")
    if not config.use_tokamax_splash:
      raise ValueError("TPU ring context parallelism requires use_tokamax_splash=True.")
    if config.use_jax_splash:
      raise ValueError("TPU ring context parallelism requires use_jax_splash=False.")
    if config.attention_type not in ("global", "mla"):
      raise ValueError("TPU Tokamax ring attention supports only attention_type='global' or 'mla'.")
    if config.attention_type == "mla":
      if config.packing:
        raise ValueError("TPU Tokamax ring attention with MLA does not support packing yet.")
      if config.use_batch_split_schedule:
        raise ValueError("TPU Tokamax ring attention with MLA does not support the DeepSeek batch-split schedule.")
    if config.context_parallel_load_balance:
      if context_parallel_size % 2 != 0:
        raise ValueError("TPU Tokamax ring load balancing requires an even context_parallel_size.")
      if config.mtp_num_layers > 0:
        raise ValueError("TPU Tokamax ring attention with context_parallel_load_balance=True does not support MTP.")
    if config.use_ragged_attention:
      raise ValueError("TPU Tokamax ring attention does not support ragged attention.")
    if config.attention_sink:
      raise ValueError("TPU Tokamax ring attention does not support attention sinks.")
    if config.use_chunked_prefill:
      raise ValueError("TPU Tokamax ring attention does not support chunked prefill yet.")
    if config.moba:
      raise ValueError("TPU Tokamax ring attention does not support MoBA.")
    if config.use_multimodal:
      raise ValueError("TPU Tokamax ring attention does not support multimodal attention.")
    if config.use_qk_clip:
      raise ValueError("TPU Tokamax ring attention does not support QK-Clip statistics yet.")
    if config.enable_dropout and config.dropout_rate > 0.0:
      raise ValueError("TPU Tokamax ring attention does not support dropout yet.")
  if context_parallel_strategy not in ("ring", "usp") and config.ring_scan_unroll != 1:
    raise ValueError(
        f"ring_scan_unroll={config.ring_scan_unroll} was specified, but is only supported when "
        "context_parallel_strategy='ring' or 'usp'."
    )
  if context_parallel_strategy == "ulysses":
    if config.hardware != "tpu":
      raise ValueError("Ulysses context parallelism (context_parallel_strategy='ulysses') is only supported on TPU.")
    if config.context_sharding != "context":
      raise ValueError("TPU Ulysses attention requires context_sharding='context'.")
    ici_context_parallel_size = config.ici_context_parallelism
    dcn_context_parallel_size = config.dcn_context_parallelism
    if ici_context_parallel_size <= 0 or dcn_context_parallel_size <= 0:
      raise ValueError(
          "TPU Ulysses attention requires explicit positive ici/dcn context parallelism values; "
          "inferred (-1) sizes are not supported."
      )
    if context_parallel_size <= 1:
      raise ValueError("TPU Ulysses attention requires context_parallel_size > 1.")
    if dcn_context_parallel_size != 1:
      raise ValueError("TPU Ulysses attention does not support dcn context parallelism yet.")
    if config.attention != "flash":
      raise ValueError("TPU Ulysses attention requires attention=flash.")
    if not config.use_tokamax_splash:
      raise ValueError("TPU Ulysses attention requires use_tokamax_splash=True.")
    if config.use_jax_splash:
      raise ValueError("TPU Ulysses attention requires use_jax_splash=False.")
    if config.use_indexer:
      raise ValueError("TPU Ulysses attention does not support sparse indexer masks.")
    if config.attention_type != "global":
      raise ValueError("TPU Ulysses attention is initially supported only for global causal attention.")
    if config.context_parallel_load_balance:
      raise ValueError(
          "TPU Ulysses attention requires context_parallel_load_balance=False: after the all-to-all every device "
          "attends over the full sequence in natural order, so the load-balancing reorder is unnecessary and "
          "would corrupt the causal mask."
      )
    if config.use_ragged_attention:
      raise ValueError("TPU Ulysses attention does not support ragged attention.")
    if config.attention_sink:
      raise ValueError("TPU Ulysses attention does not support attention sinks.")
    if config.use_chunked_prefill:
      raise ValueError("TPU Ulysses attention does not support chunked prefill yet.")
    if config.use_multimodal:
      raise ValueError("TPU Ulysses attention does not support multimodal attention.")
    if config.enable_dropout and config.dropout_rate > 0.0:
      raise ValueError("TPU Ulysses attention does not support dropout yet.")
    if config.dq_reduction_steps not in (0, 3):
      raise ValueError("TPU Ulysses attention requires dq_reduction_steps to be 0 or 3.")
    if config.use_qk_clip:
      raise ValueError("TPU Ulysses attention does not support QK-Clip statistics yet.")
    if config.max_target_length % context_parallel_size != 0:
      raise ValueError(
          "TPU Ulysses attention requires max_target_length "
          f"({config.max_target_length}) to be divisible by context_parallel_size ({context_parallel_size})."
      )
    if config.num_query_heads % context_parallel_size != 0:
      raise ValueError(
          "TPU Ulysses attention requires num_query_heads "
          f"({config.num_query_heads}) to be divisible by context_parallel_size ({context_parallel_size})."
      )
    if config.num_kv_heads == 1:
      raise ValueError("TPU Ulysses attention does not support MQA with context_parallel_size > 1.")
    if config.num_kv_heads % context_parallel_size != 0:
      raise ValueError(
          "TPU Ulysses attention requires num_kv_heads "
          f"({config.num_kv_heads}) to be divisible by context_parallel_size ({context_parallel_size})."
      )
  config._validate_usp_context_parallelism()
  # STRIPED reorder strategy is a Transformer Engine feature and is GPU-only.
  # AUTO is resolved in training because test code paths may load the same
  # config but use a different reorder path.
  if (
      context_parallel_size > 1
      and "gpu" not in config.hardware
      and config.context_parallel_load_balance
      and config.context_parallel_reorder_strategy == ReorderStrategy.STRIPED
  ):
    raise ValueError(
        "STRIPED reorder strategy requires Transformer Engine and is only supported on GPUs. "
        f"Got hardware={config.hardware!r}."
    )
