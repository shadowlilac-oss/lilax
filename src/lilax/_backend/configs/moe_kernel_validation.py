"""Validation of native mixture-of-experts GMM kernel and weight-layout options."""

from lilax._backend.common.common_types import DecoderBlockType
from lilax._backend.configs.precision_types import DType


def validate_gmm_v2(config):
  if config.use_gmm_v2:
    if not config.use_tokamax_gmm:
      raise ValueError("GMM v2 requires `use_tokamax_gmm=True`.")
    if config.use_batch_split_schedule:
      raise ValueError("GMM v2 is not supported with a batch split schedule.")


def validate_moe_kernel_options(config):
  """Validates MoE token chunking, backward accumulation and prequantization, and flat FSDP weights."""
  if config.num_moe_token_chunks > 1:
    if config.max_target_length % config.num_moe_token_chunks != 0:
      raise ValueError(
          f"num_moe_token_chunks={config.num_moe_token_chunks} must evenly divide "
          f"max_target_length={config.max_target_length}."
      )
  if config.moe_accumulate_chunk_wgrad and not (
      config.use_tokamax_gmm and config.use_gmm_v2 and config.num_moe_emb_chunks == 0 and not config.prefuse_moe_weights
  ):
    raise ValueError(
        "moe_accumulate_chunk_wgrad=True requires use_tokamax_gmm=True, use_gmm_v2=True, num_moe_emb_chunks=0"
        " and prefuse_moe_weights=False."
    )
  if config.moe_accumulate_wi_dlhs and not (
      config.use_tokamax_gmm and config.use_gmm_v2 and not config.prefuse_moe_weights
  ):
    raise ValueError(
        "moe_accumulate_wi_dlhs=True requires use_tokamax_gmm=True, use_gmm_v2=True and prefuse_moe_weights=False."
    )
  if config.moe_bwd_prequant_before_unsort and not (
      config.moe_tc_ragged_3d_dispatch and config.moe_tc_ragged_weights_on_activation
  ):
    raise ValueError(
        "moe_bwd_prequant_before_unsort=True requires moe_tc_ragged_3d_dispatch=True and"
        " moe_tc_ragged_weights_on_activation=True."
    )
  if config.moe_combine_bwd_direct_qarray and not config.moe_bwd_prequant_before_unsort:
    raise ValueError("moe_combine_bwd_direct_qarray=True requires moe_bwd_prequant_before_unsort=True.")
  if config.moe_bwd_prequant_before_unsort and config.quantization:
    # The fused unsort applies the weight / act / bwd scales to the token-order cotangent, so they must be
    # per-tensor; the config's qwix rule keeps channelwise axes, so only fixed calibration gives that. With
    # moe_combine_bwd_direct_qarray, each EP shard also scales its cotangent before the all-gather, so a dynamic
    # (shard-local) scale would differ across shards.
    dynamic = [
        f"{name}_quantization_calibration_method={method!r}"
        for name in ("weight", "act", "bwd")
        if not (method := getattr(config, f"{name}_quantization_calibration_method")).lower().startswith("fixed")
    ]
    if dynamic:
      raise ValueError(
          "moe_bwd_prequant_before_unsort=True with quantization requires fixed weight, act and bwd calibration,"
          f" got {', '.join(dynamic)}."
      )
  if config.moe_combine_bwd_direct_qarray and config.moe_quantize_combine_bwd_method:
    raise ValueError(
        f"moe_quantize_combine_bwd_method={config.moe_quantize_combine_bwd_method!r} has no effect with"
        " moe_combine_bwd_direct_qarray=True: the fused EP combine quantizes the cotangent with the gmm bwd rule."
        " Unset moe_quantize_combine_bwd_method."
    )
  if config.moe_3d_dlhs_partial_sum and not (config.moe_accumulate_wi_dlhs and config.moe_tc_ragged_3d_gmm):
    raise ValueError("moe_3d_dlhs_partial_sum=True requires moe_accumulate_wi_dlhs=True and moe_tc_ragged_3d_gmm=True.")
  if config.moe_flat_fsdp_weights:
    if not (
        config.sparse_matmul
        and (config.shard_embed_moe_on_fsdp or not config.quantization)
        and ((config.use_tokamax_gmm and config.use_gmm_v2) or config.prefetch_fsdp_weights)
        and not config.prefuse_moe_weights
    ):
      raise ValueError(
          "moe_flat_fsdp_weights=True requires sparse_matmul=True, (shard_embed_moe_on_fsdp=True or unquantized),"
          " ((use_tokamax_gmm=True and use_gmm_v2=True) or prefetch_fsdp_weights=True) and prefuse_moe_weights=False."
      )
    # These either pick their own 3D expert weight layout or carry per-expert params that assume it.
    if (
        config.shard_exp_on_fsdp
        or config.use_2d_fsdp_sharding
        or config.use_batch_split_schedule
        or config.weight_dtype in (DType.FLOAT8_E4M3FN, DType.FLOAT8_E5M2)
        or config.decoder_block == DecoderBlockType.GEMMA4
    ):
      raise ValueError(
          "moe_flat_fsdp_weights=True is incompatible with shard_exp_on_fsdp, use_2d_fsdp_sharding,"
          " use_batch_split_schedule, fp8 weight_dtype and the gemma4 decoder block."
      )
