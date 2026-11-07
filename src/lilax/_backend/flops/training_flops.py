"""Total per-device training TFLOPs for a train step."""

from lilax._backend.flops import decoder_flops, engram_flops, ffn_flops, flops, vision_flops
from lilax._backend.common.common_types import DecoderBlockType


def calculate_tflops_training_per_device(config, log=True):
  """Calculate training TFLOP."""
  if config.use_m3_model:
    from lilax._backend.m3 import models as m3_models

    if config.model_name in m3_models.MODEL_FLOPS_REGISTRY:
      return m3_models.calculate_tflops_per_device(config, log=log)
  is_ffn_flops_already_total = False
  if config.num_experts > 1:
    if config.decoder_block in (
        DecoderBlockType.DEEPSEEK, DecoderBlockType.LLAMA4, DecoderBlockType.QWEN3_NEXT,
        DecoderBlockType.QWEN3_5, DecoderBlockType.GEMMA4, DecoderBlockType.DEEPSEEK4,
    ):
      total_ffn_flops = ffn_flops.calculate_routed_and_shared_ffn_tflops_per_device(config)
      is_ffn_flops_already_total = True
    elif config.decoder_block == DecoderBlockType.QWEN3_CUSTOM_MOE:
      in_dim = config.moe_expert_input_dim
      gate_flops = 2 * config.per_device_batch_size * config.max_target_length * in_dim * config.num_experts
      total_ffn_flops = (
          gate_flops
          + ffn_flops.calculate_ffn_mamtul_tflops_per_device(config, config.moe_mlp_dim, in_dim=in_dim)
          * config.num_experts_per_tok
      )
    else:
      gate_flops = 2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.num_experts
      total_ffn_flops = (
          gate_flops
          + ffn_flops.calculate_ffn_mamtul_tflops_per_device(config, config.moe_mlp_dim) * config.num_experts_per_tok
      )
  else:
    total_ffn_flops = ffn_flops.calculate_ffn_mamtul_tflops_per_device(config, config.mlp_dim)
  total_ffn_flops_all_layers = (
      total_ffn_flops if is_ffn_flops_already_total else total_ffn_flops * config.num_decoder_layers
  )
  if config.attention_type == "mla":
    qkv_flops, causal_attention_flops, projection_flops = flops.calculate_mla_tflops_per_device(config)
  else:
    kv_multiplier = 1 if config.share_kv_projections else 2
    qkv_flops = (
        2 * config.per_device_batch_size * config.max_target_length * config.emb_dim
        * (config.num_query_heads + kv_multiplier * config.num_kv_heads) * config.head_dim
    )
    noncausal_attention_flops = (
        4 * config.per_device_batch_size * config.max_target_length**2 * config.num_query_heads * config.head_dim
    )
    projection_flops = (
        2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.num_query_heads * config.head_dim
    )
    causal_attention_flops = noncausal_attention_flops / 2
  embedding_flops = 2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.vocab_size
  if config.decoder_block == DecoderBlockType.GEMMA2:
    attention_tflops, learnable_weight_tflops = decoder_flops.calculate_gemma2_tflops_training_per_device(
        config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops
    )
  elif config.decoder_block == DecoderBlockType.GEMMA3:
    attention_tflops, learnable_weight_tflops = decoder_flops.calculate_mixed_attention_model_tflops_training_per_device(
        config, total_ffn_flops_all_layers, qkv_flops, projection_flops, embedding_flops, attention_pattern_length=6
    )
  elif config.decoder_block == DecoderBlockType.GPT_OSS:
    attention_tflops, learnable_weight_tflops = decoder_flops.calculate_mixed_attention_model_tflops_training_per_device(
        config, total_ffn_flops_all_layers, qkv_flops, projection_flops, embedding_flops, attention_pattern_length=2
    )
  elif config.decoder_block == DecoderBlockType.LLAMA4:
    attention_tflops = decoder_flops.calculate_llama4_attention_tflops(config)
    learnable_weight_tflops = (
        (total_ffn_flops_all_layers + (qkv_flops + projection_flops) * config.num_decoder_layers + embedding_flops)
        * 3 / 10**12
    )
  elif config.decoder_block == DecoderBlockType.GEMMA4:
    attention_tflops, learnable_weight_tflops = decoder_flops.calculate_gemma4_tflops_training_per_device(
        config, total_ffn_flops_all_layers, embedding_flops, attention_pattern_length=6
    )
  elif config.decoder_block == DecoderBlockType.GEMMA4_SMALL:
    attention_tflops, learnable_weight_tflops = decoder_flops.calculate_gemma4_small_tflops_training_per_device(
        config, embedding_flops
    )
  elif config.decoder_block == DecoderBlockType.DEEPSEEK4:
    attention_tflops, learnable_weight_tflops = decoder_flops.calculate_deepseek4_tflops_training_per_device(
        config, total_ffn_flops_all_layers, embedding_flops
    )
  elif config.decoder_block == DecoderBlockType.DEEPSEEK:
    learnable_weight_tflops = (
        (total_ffn_flops_all_layers + (qkv_flops + projection_flops) * config.num_decoder_layers + embedding_flops)
        * 3 / 10**12
    )
    attention_tflops = causal_attention_flops * config.num_decoder_layers * 3 / 10**12
  elif config.decoder_block == DecoderBlockType.QWEN3_CUSTOM_MOE:
    qwen3_custom_proj_flops = (
        2 * config.per_device_batch_size * config.max_target_length * config.attention_output_dim
        * config.num_query_heads * config.head_dim
    )
    layer_up_proj_flops = (
        2 * config.per_device_batch_size * config.max_target_length * config.attention_output_dim * config.emb_dim
    )
    per_layer_flops = qkv_flops + qwen3_custom_proj_flops + layer_up_proj_flops
    total_weight_flops = total_ffn_flops_all_layers + per_layer_flops * config.num_decoder_layers + embedding_flops
    learnable_weight_tflops = total_weight_flops * 3 / 10**12
    attention_tflops = causal_attention_flops * config.num_decoder_layers * 3 / 10**12
  elif config.decoder_block in (DecoderBlockType.QWEN3_NEXT, DecoderBlockType.QWEN3_5):
    gdn_weight_flops_per_layer, gdn_attn_flops_per_layer = flops.calculate_gated_delta_net_flops_per_device(config)
    cycle_interval = config.inhomogeneous_layer_cycle_interval
    num_full_attn_layers = config.num_decoder_layers // cycle_interval
    num_linear_attn_layers = config.num_decoder_layers - num_full_attn_layers
    total_weights = (
        total_ffn_flops_all_layers + embedding_flops + (qkv_flops + projection_flops) * num_full_attn_layers
        + gdn_weight_flops_per_layer * num_linear_attn_layers
    )
    learnable_weight_tflops = total_weights * 3 / 10**12
    total_attn = (causal_attention_flops * num_full_attn_layers) + (gdn_attn_flops_per_layer * num_linear_attn_layers)
    attention_tflops = total_attn * 3 / 10**12
  else:
    learnable_weight_tflops = (
        (total_ffn_flops_all_layers + (qkv_flops + projection_flops) * config.num_decoder_layers + embedding_flops)
        * 3 / 10**12
    )
    attention_tflops = causal_attention_flops * config.num_decoder_layers * 3 / 10**12

  mtp_num_layers = getattr(config, "mtp_num_layers", 0)
  if mtp_num_layers > 0:
    if config.num_experts > 1:
      gate_flops = 2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.num_experts
      if config.decoder_block in (
          DecoderBlockType.DEEPSEEK, DecoderBlockType.DEEPSEEK4, DecoderBlockType.LLAMA4,
          DecoderBlockType.QWEN3_NEXT, DecoderBlockType.GEMMA4,
      ):
        shared_flops = (
            ffn_flops.calculate_ffn_mamtul_tflops_per_device(config, ffn_flops.get_shared_expert_mlp_dim(config))
            * config.shared_experts
        )
        routed_flops = (
            ffn_flops.calculate_ffn_mamtul_tflops_per_device(config, config.moe_mlp_dim) * config.num_experts_per_tok
        )
        last_layer_ffn_flops = gate_flops + shared_flops + routed_flops
      else:
        routed_flops = (
            ffn_flops.calculate_ffn_mamtul_tflops_per_device(config, config.moe_mlp_dim) * config.num_experts_per_tok
        )
        last_layer_ffn_flops = gate_flops + routed_flops
    else:
      last_layer_ffn_flops = ffn_flops.calculate_ffn_mamtul_tflops_per_device(config, config.mlp_dim)
    mtp_weight_flops = (last_layer_ffn_flops + qkv_flops + projection_flops) * mtp_num_layers
    mtp_attn_flops = causal_attention_flops * mtp_num_layers
    learnable_weight_tflops += mtp_weight_flops * 3 / 10**12
    attention_tflops += mtp_attn_flops * 3 / 10**12
  if config.engram_layers:
    engram_learnable_tflops, engram_attention_tflops = engram_flops.calculate_engram_tflops(config)
    learnable_weight_tflops += engram_learnable_tflops
    attention_tflops += engram_attention_tflops
  if config.use_multimodal:
    mm_total_tflops, mm_learnable_weight_tflops, mm_attention_tflops = vision_flops.calculate_vision_encoder_tflops(config)
    if log and mm_total_tflops > 0:
      print(
          f"{config.model_name} vision layers per train step:\n",
          f"Total TFLOPs: {mm_total_tflops:.2f} \n",
          f"split as {100 * mm_learnable_weight_tflops/mm_total_tflops:.2f}% learnable weight flops",
          f"and {100 * mm_attention_tflops/mm_total_tflops:.2f}% attention flops;\n",
          f"learnable weight {mm_learnable_weight_tflops:.2f} TFLOPs, attention {mm_attention_tflops:.2f} TFLOPs",
      )
    learnable_weight_tflops += mm_learnable_weight_tflops
    attention_tflops += mm_attention_tflops
  learnable_weight_tflops = learnable_weight_tflops * config.gradient_accumulation_steps
  attention_tflops = attention_tflops * config.gradient_accumulation_steps
  if config.use_dpo:
    reference_model_tflops = learnable_weight_tflops / 3
    reference_model_attention_tflops = attention_tflops / 3
    attention_tflops = attention_tflops + reference_model_attention_tflops
  else:
    reference_model_tflops = 0
  total_tflops = learnable_weight_tflops + attention_tflops + reference_model_tflops
  if log:
    print(
        "Per train step:\n",
        f"Total TFLOPs: {total_tflops:.2f} \n",
        f"split as {100 * learnable_weight_tflops/total_tflops:.2f}% learnable weight flops",
        f"and {100 * attention_tflops/total_tflops:.2f}% attention flops",
    )
  return total_tflops, learnable_weight_tflops, attention_tflops
