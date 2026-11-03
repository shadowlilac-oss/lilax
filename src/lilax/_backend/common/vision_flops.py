"""FLOPs accounting for multimodal vision encoders."""

from lilax._backend.utils import max_logging


def calculate_gemma3_vision_layers_tflops_per_device(config):
  B = config.per_device_batch_size
  C = config.num_channels_for_vit
  H = W = config.image_size_for_vit
  embed_dim = config.emb_dim
  patch_size = 14
  hidden_dim = 1152
  intermediate_dim = 4304
  num_layers = 27
  vision_exit_pooling_window = 4
  num_patches_h = H // patch_size
  num_patches_w = W // patch_size
  seq_len = num_patches_h * num_patches_w
  patch_embed_flops = 2 * B * seq_len * (C * patch_size * patch_size) * hidden_dim
  qkv_flops_per_layer = 3 * (2 * B * seq_len * hidden_dim * hidden_dim)
  attn_flops_per_layer = 4 * B * seq_len * seq_len * hidden_dim
  projection_flops_per_layer = 2 * B * seq_len * hidden_dim * hidden_dim
  mlp_flops_per_layer = 2 * (2 * B * seq_len * hidden_dim * intermediate_dim)
  total_attn_flops = attn_flops_per_layer * num_layers
  encoder_flops = (+qkv_flops_per_layer + projection_flops_per_layer + mlp_flops_per_layer) * num_layers
  seq_len_after_pooling = (num_patches_h // vision_exit_pooling_window) * (num_patches_w // vision_exit_pooling_window)
  vision_embedder_flops = 2 * B * seq_len_after_pooling * hidden_dim * embed_dim
  learnable_weight_flops = patch_embed_flops + encoder_flops + vision_embedder_flops
  if config.freeze_vision_encoder_params:
    learnable_weight_flops += 2 * vision_embedder_flops
  else:
    learnable_weight_flops *= 3
    total_attn_flops *= 3
  learnable_weight_tflops = learnable_weight_flops / 1e12
  total_attn_tflops = total_attn_flops / 1e12
  total_tflops = learnable_weight_tflops + total_attn_tflops
  return total_tflops, learnable_weight_tflops, total_attn_tflops


def calculate_llama4_vision_layers_tflops_per_device(config):
  B = config.per_device_batch_size
  C = config.num_channels_for_vit
  H = W = config.tile_size_for_vit
  patch_size = config.patch_size_for_vit
  hidden_dim = config.hidden_size_for_vit
  intermediate_dim = config.intermediate_size_for_vit
  num_layers = config.num_hidden_layers_for_vit
  pixel_shuffle_fc1_out_dim = config.projector_input_dim_for_vit
  pixel_shuffle_fc2_out_dim = config.projector_output_dim_for_vit
  base_emb_dim = config.base_emb_dim
  pixel_shuffle_ratio = config.pixel_shuffle_ratio_for_vit
  num_patches = (H // patch_size) * (W // patch_size)
  pixel_shuffle_tokens = num_patches * pixel_shuffle_ratio**2
  patch_embed_flops = 2 * B * num_patches * (C * patch_size * patch_size) * hidden_dim
  seq_len = num_patches + 1
  qkv_flops_per_layer = 3 * (2 * B * seq_len * hidden_dim * hidden_dim)
  attn_flops_per_layer = 4 * B * seq_len * seq_len * hidden_dim
  projection_flops_per_layer = 2 * B * seq_len * hidden_dim * hidden_dim
  mlp_flops_per_layer = 2 * (2 * B * seq_len * hidden_dim * intermediate_dim)
  total_attn_flops = attn_flops_per_layer * num_layers
  vision_encoder_flops = (+qkv_flops_per_layer + projection_flops_per_layer + mlp_flops_per_layer) * num_layers
  pixel_shuffle_fc1_flops = 2 * B * pixel_shuffle_tokens * intermediate_dim * pixel_shuffle_fc1_out_dim
  pixel_shuffle_fc2_flops = 2 * B * pixel_shuffle_tokens * pixel_shuffle_fc1_out_dim * pixel_shuffle_fc2_out_dim
  pixel_shuffle_total_flops = pixel_shuffle_fc1_flops + pixel_shuffle_fc2_flops
  projector_flops = 2 * B * pixel_shuffle_tokens * pixel_shuffle_fc1_out_dim * base_emb_dim
  learnable_weight_flops = patch_embed_flops + vision_encoder_flops + pixel_shuffle_total_flops + projector_flops
  if config.freeze_vision_encoder_params:
    learnable_weight_flops += 2 * projector_flops
  else:
    learnable_weight_flops *= 3
    total_attn_flops *= 3
  learnable_weight_tflops = learnable_weight_flops / 1e12
  total_attn_tflops = total_attn_flops / 1e12
  total_tflops = learnable_weight_tflops + total_attn_tflops
  return total_tflops, learnable_weight_tflops, total_attn_tflops


def calculate_vision_encoder_tflops(config):
  if config.model_name.startswith("gemma3"):
    mm_total_tflops, mm_learnable_weight_tflops, mm_attention_tflops = calculate_gemma3_vision_layers_tflops_per_device(
        config
    )
  elif config.model_name.startswith("llama4"):
    mm_total_tflops, mm_learnable_weight_tflops, mm_attention_tflops = calculate_llama4_vision_layers_tflops_per_device(
        config
    )
  else:
    max_logging.log(
        f"Vision encoder TFLOPs calculation not implemented for model {config.model_name}, counting as 0 for now."
    )
    mm_total_tflops = mm_learnable_weight_tflops = mm_attention_tflops = 0
  return mm_total_tflops, mm_learnable_weight_tflops, mm_attention_tflops
