"""Validation of native explicit sharding support."""

from lilax._backend.common.common_types import DecoderBlockType, ShardMode


def validate_explicit_sharding(config):
  if config.shard_mode == ShardMode.EXPLICIT:
    supported_decoders = {
        "simple", "simple_mlp", "llama2", "llama4", "deepseek", "gpt3", "gpt_oss", "mistral", "mixtral",
        "qwen2", "qwen3", "qwen3_moe", "qwen3_custom_moe", "qwen3_5", "qwen3_next", "gemma", "gemma2",
        "gemma3", "gemma4", "gemma4_small",
    }
    if config.decoder_block.value not in supported_decoders:
      raise ValueError(
          f"Decoder '{config.decoder_block.value}' is not supported with 'explicit' sharding. "
          f"Supported options are: {sorted(supported_decoders)}."
      )
    if config.use_multimodal:
      raise ValueError(
          "'explicit' sharding is not supported with `use_multimodal`; the vision and audio encoders "
          "have not been onboarded to explicit sharding yet."
      )
    # Both hybrid decoders share the same GatedDeltaNet sublayers, so the same gaps.
    if config.decoder_block in (DecoderBlockType.QWEN3_5, DecoderBlockType.QWEN3_NEXT):
      decoder_name = config.decoder_block.value
      if config.num_experts > 1 and not config.sparse_matmul:
        raise ValueError(
            f"'explicit' sharding with the '{decoder_name}' decoder requires"
            " `sparse_matmul=True`; the dense matmul MoE path has not been"
            " onboarded to explicit sharding yet."
        )
      gdn_context_parallel_size = (
          config.ici_context_parallelism
          * config.dcn_context_parallelism
          * config.ici_context_usp_ulysses_parallelism
          * config.dcn_context_usp_ulysses_parallelism
      )
      if gdn_context_parallel_size > 1:
        raise ValueError(
            f"'explicit' sharding with the '{decoder_name}' decoder does not"
            " support context parallelism yet. The GatedDeltaNet short"
            " convolution left-pads the sequence by `gdn_conv_kernel_dim -"
            " 1` and slices the result back, which explicit sharding cannot"
            " express on a sharded sequence axis. Use `shard_mode=auto` when"
            " `ici_context_parallelism` or"
            " `ici_context_usp_ulysses_parallelism` is set."
        )
