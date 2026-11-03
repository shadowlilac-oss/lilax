"""Native mesh and training-state utilities."""

import functools
import os

from flax import nnx, traverse_util
from flax.core.meta import get_partition_spec
from flax.core.spmd import logical_axis_rules as axis_rules
from flax.training.train_state import TrainState
import jax
from jax.experimental import mesh_utils
from jax.experimental.serialize_executable import deserialize_and_load
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding
import numpy as np
import optax
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
import orbax.checkpoint.experimental.emergency.replicator_checkpoint_manager as emergency_replicator_checkpoint_manager

from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.common.common_types import AttentionType, DecoderBlockType, ReorderStrategy, ShardMode
from lilax._backend.experimental.lineage import lineage_adapter
from lilax._backend.models import gemma4_small
from lilax._backend.multimodal import processor as mm_processor
from lilax._backend.optimizers.schedules import create_learning_rate_schedule
from lilax._backend.trainers.diloco import diloco
from lilax._backend.trainers.diloco import utils as diloco_utils
from lilax._backend.utils import elastic_utils, gcs_utils, max_logging, max_utils, maxtext_utils_nnx, sharding

OVERWRITE_WITH_GRADIENT = "_overwrite_with_gradient"


def get_functional_train_with_signature(
    train_step, data_sharding, state_mesh_shardings, model, config, params_shardings=None
):
  """Get the shardings (both state and data) for `train_step`."""
  functional_train = functools.partial(train_step, model, config, state_mesh_shardings, params_shardings)
  functional_train.__name__ = "train_step"
  in_shardings = (state_mesh_shardings, data_sharding)
  out_shardings = (state_mesh_shardings, None)
  static_argnums = ()
  donate_argnums = 0
  return functional_train, in_shardings, out_shardings, static_argnums, donate_argnums


def get_functional_eval_with_signature(eval_step, data_sharding, state_mesh_shardings, model, config):
  """Get the shardings (both state and data) for `eval_step`."""
  functional_eval = functools.partial(eval_step, model, config)
  functional_eval.__name__ = "eval_step"
  in_shardings = (state_mesh_shardings, data_sharding)
  out_shardings = None
  static_argnums = ()
  donate_argnums = ()
  return functional_eval, in_shardings, out_shardings, static_argnums, donate_argnums


def shard_reorder_causal_load_balanced(
    batch, cp_size, shard_mode, reorder_strategy=ReorderStrategy.DUAL_CHUNK_SWAP, hardware="tpu"
):
  """Shard the output of the reordered sequence."""
  reordered = max_utils.reorder_causal_load_balanced(batch, cp_size, reorder_strategy, hardware)
  for _, value in batch.items():
    if isinstance(value, jax.Array):
      reordered = sharding.maybe_shard_with_name(reordered, value.sharding, shard_mode)
      break
  return reordered


def get_reorder_callable(cp_size, shard_mode, reorder_strategy=ReorderStrategy.DUAL_CHUNK_SWAP, hardware="tpu"):
  """Creates a callable that can be used with map() to reorder batches."""
  return functools.partial(
      shard_reorder_causal_load_balanced,
      cp_size=cp_size,
      shard_mode=shard_mode,
      reorder_strategy=reorder_strategy,
      hardware=hardware,
  )


def get_shaped_batch(config, batch_sharding=None, is_eval=False):
  """Return the shape of the batch as produced by the data iterator."""
  global_batch_size = config.global_batch_size_to_load_eval if is_eval else config.global_batch_size_to_load
  if config.enable_diloco:
    batch_shape = (
        config.num_diloco_replicas,
        global_batch_size // config.num_diloco_replicas,
        config.max_target_length,
    )
  else:
    batch_shape = (global_batch_size, config.max_target_length)
  shaped_batch = {}
  shaped_batch["inputs"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=batch_sharding)
  position_shape = batch_shape + (3,) if config.use_mrope and config.use_multimodal else batch_shape
  shaped_batch["inputs_position"] = jax.ShapeDtypeStruct(position_shape, jnp.int32, sharding=batch_sharding)
  shaped_batch["inputs_segmentation"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=batch_sharding)
  shaped_batch["targets"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=batch_sharding)
  shaped_batch["targets_position"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=batch_sharding)
  shaped_batch["targets_segmentation"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=batch_sharding)
  if getattr(config, "training_objective", "causal_lm") == "block_diffusion":
    shaped_batch["corruption_mask"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=batch_sharding)
    shaped_batch["targets_loss_mask"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=batch_sharding)
  if config.use_multimodal:
    batch_size = config.micro_batch_size_to_eval_on if is_eval else config.micro_batch_size_to_train_on
    is_video = getattr(config, "video_max_grid_t", None) is not None
    if is_video:
      max_t = config.video_max_grid_t
      max_h = config.video_max_grid_h
      max_w = config.video_max_grid_w
      tps = config.temporal_patch_size_for_vit
      patch = config.patch_size_for_vit
      channels = config.num_channels_for_vit
      video_shape = (batch_size, channels, max_t * tps, max_h * patch, max_w * patch)
      video_mask_shape = (batch_size, 1, max_t * tps, max_h * patch, max_w * patch)
      shaped_batch["images"] = jax.ShapeDtypeStruct(video_shape, jnp.float32, sharding=batch_sharding)
      shaped_batch["image_masks"] = jax.ShapeDtypeStruct(video_mask_shape, jnp.int32, sharding=batch_sharding)
      shaped_batch["video_grid_thw"] = jax.ShapeDtypeStruct((batch_size, 3), jnp.int32, sharding=batch_sharding)
    else:
      image_shape = mm_processor.get_dummy_image_shape_for_init(config.model_name, batch_size=batch_size)
      shaped_batch["images"] = jax.ShapeDtypeStruct(image_shape, jnp.int32, sharding=batch_sharding)
      if "llama4" in config.model_name:
        shaped_batch["image_masks"] = jax.ShapeDtypeStruct(image_shape[:2], jnp.int32, sharding=batch_sharding)
  if config.use_audio:
    audio_shape = mm_processor.get_dummy_audio_shape_for_init(config)
    shaped_batch["audios"] = jax.ShapeDtypeStruct(audio_shape, jnp.float32, sharding=batch_sharding)
  return shaped_batch


def should_prevent_cse_in_remat(config):
  if config.scan_layers:
    return False
  if config.gradient_accumulation_steps > 1 and config.hardware in ("gpu", "gpu_multiprocess"):
    return False
  return True


def get_save_and_offload_names(config) -> tuple[list[str], list[str]]:
  if config.remat_policy == "qkv_proj_offloaded":
    return [], ["query_proj", "value_proj", "key_proj", "kv_proj"]
  if config.remat_policy == "minimal_offloaded":
    return [], [
        "query_proj", "value_proj", "key_proj", "kv_proj", "qkv_proj", "out_proj", "mlpwi_0", "mlpwi_1",
        "mlpwi", "mlpwo",
    ]
  if config.remat_policy == "custom":
    return list(config.tensors_on_device or []), list(config.tensors_to_offload or [])
  return [], []


def load_compiled(config, partial_train, state, execution_devices, data_sharding=None):
  """Load a serialized compiled train step function."""
  def load_serialized_compiled(save_name):
    with open(save_name, "rb") as file:
      return file.read()

  def get_train_input_output_trees(func, input_args, input_kwargs):
    _, in_tree_recreated = jax.tree_util.tree_flatten((input_args, input_kwargs))
    out_shaped = jax.eval_shape(func, *input_args, **input_kwargs)
    _, out_tree_recreated = jax.tree_util.tree_flatten(out_shaped)
    return in_tree_recreated, out_tree_recreated

  serialized_compiled = load_serialized_compiled(config.compiled_trainstep_file)
  shaped_batch = get_shaped_batch(config, batch_sharding=data_sharding)
  shaped_input_args = (state, shaped_batch)
  shaped_input_kwargs = {}
  in_tree, out_tree = get_train_input_output_trees(partial_train, shaped_input_args, shaped_input_kwargs)
  p_train_step = deserialize_and_load(serialized_compiled, in_tree, out_tree, execution_devices=execution_devices)
  return p_train_step


def apply_gradient_clipping(raw_grads, state, clipping_threshold):
  gradient_clip_transformation = optax.clip_by_global_norm(clipping_threshold)
  if OVERWRITE_WITH_GRADIENT in raw_grads:
    fp8_stats = raw_grads.pop(OVERWRITE_WITH_GRADIENT)
    grads, _ = gradient_clip_transformation.update(raw_grads, state, None)
    grads[OVERWRITE_WITH_GRADIENT] = fp8_stats
    raw_grads[OVERWRITE_WITH_GRADIENT] = fp8_stats
  else:
    grads, _ = gradient_clip_transformation.update(raw_grads, state, None)
  return grads


def calculate_tokens_training_per_device(config):
  """Calculate training Tokens per device"""
  return config.max_target_length * config.per_device_batch_size * config.gradient_accumulation_steps


def calculate_gemma2_tflops_training_per_device(config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops):
  window = min(config.sliding_window_size, config.max_target_length)
  global_causal_flops = (
      2 * config.per_device_batch_size * config.max_target_length**2 * config.num_query_heads * config.head_dim
  )
  local_causal_flops = (
      4 * config.per_device_batch_size * (config.max_target_length * window - 0.5 * window**2)
      * config.num_query_heads * config.head_dim
  )
  attention_tflops = (global_causal_flops + local_causal_flops) * config.num_decoder_layers * 3 / 10**12
  learnable_weight_tflops = (
      ((total_ffn_flops + qkv_flops + projection_flops) * config.num_decoder_layers * 2 + embedding_flops) * 3 / 10**12
  )
  return attention_tflops, learnable_weight_tflops


def calculate_mixed_attention_model_tflops_training_per_device(
    config, total_ffn_flops_all_layers, qkv_flops, projection_flops, embedding_flops, attention_pattern_length
):
  num_layers = config.num_decoder_layers
  num_global_layers = num_layers // attention_pattern_length
  num_local_layers = num_layers - num_global_layers
  global_causal_flops_per_layer = (
      2 * config.per_device_batch_size * config.max_target_length**2 * config.num_query_heads * config.head_dim
  )
  window = min(config.sliding_window_size, config.max_target_length)
  local_causal_flops_per_layer = (
      4 * config.per_device_batch_size * (config.max_target_length * window - 0.5 * window**2)
      * config.num_query_heads * config.head_dim
  )
  causal_attention_flops = (
      num_global_layers * global_causal_flops_per_layer + num_local_layers * local_causal_flops_per_layer
  )
  attention_tflops = causal_attention_flops * 3 / 10**12
  total_learnable_flops = total_ffn_flops_all_layers
  total_learnable_flops += (qkv_flops + projection_flops) * num_layers + embedding_flops
  learnable_weight_tflops = total_learnable_flops * 3 / 10**12
  return attention_tflops, learnable_weight_tflops


def calculate_gemma4_tflops_training_per_device(
    config, total_ffn_flops_all_layers, embedding_flops, attention_pattern_length
):
  num_layers = config.num_decoder_layers
  num_global_layers = num_layers // attention_pattern_length
  num_local_layers = num_layers - num_global_layers
  kv_multiplier = 1 if config.share_kv_projections else 2
  global_head_dim = config.global_head_dim or config.head_dim
  global_num_kv_heads = config.global_num_kv_heads or config.num_kv_heads
  global_causal_flops_per_layer = (
      2 * config.per_device_batch_size * config.max_target_length**2 * config.num_query_heads * global_head_dim
  )
  window = min(config.sliding_window_size, config.max_target_length)
  local_causal_flops_per_layer = (
      4 * config.per_device_batch_size * (config.max_target_length * window - 0.5 * window**2)
      * config.num_query_heads * config.head_dim
  )
  causal_attention_flops = (
      num_global_layers * global_causal_flops_per_layer + num_local_layers * local_causal_flops_per_layer
  )
  attention_tflops = causal_attention_flops * 3 / 10**12
  global_qkv_flops_per_layer = (
      2 * config.per_device_batch_size * config.max_target_length * config.emb_dim
      * (config.num_query_heads + kv_multiplier * global_num_kv_heads) * global_head_dim
  )
  global_projection_flops_per_layer = (
      2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.num_query_heads * global_head_dim
  )
  local_qkv_flops_per_layer = (
      2 * config.per_device_batch_size * config.max_target_length * config.emb_dim
      * (config.num_query_heads + 2 * config.num_kv_heads) * config.head_dim
  )
  local_projection_flops_per_layer = (
      2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.num_query_heads * config.head_dim
  )
  total_learnable_flops = total_ffn_flops_all_layers
  total_learnable_flops += (
      (local_qkv_flops_per_layer + local_projection_flops_per_layer) * num_local_layers
      + (global_qkv_flops_per_layer + global_projection_flops_per_layer) * num_global_layers + embedding_flops
  )
  learnable_weight_tflops = total_learnable_flops * 3 / 10**12
  return attention_tflops, learnable_weight_tflops


def _calculate_chunked_attention_flops_per_layer(config, seq_len, chunk_size):
  num_chunks = seq_len // chunk_size
  rem_chunk_size = seq_len % chunk_size
  chunked_complexity = (num_chunks * chunk_size**2) + (rem_chunk_size**2)
  return 4 * config.per_device_batch_size * chunked_complexity * config.num_query_heads * config.head_dim


def calculate_llama4_attention_tflops(config):
  num_layers = config.num_decoder_layers
  seq_len = config.max_target_length
  chunk_size = config.chunk_attn_window_size
  num_global_layers = num_layers // config.nope_layer_interval
  num_chunked_layers = num_layers - num_global_layers
  global_attention_flops_per_layer = (
      4 * config.per_device_batch_size * seq_len**2 * config.num_query_heads * config.head_dim
  )
  chunked_attention_flops_per_layer = _calculate_chunked_attention_flops_per_layer(config, seq_len, chunk_size)
  noncausal_attention_flops = (num_global_layers * global_attention_flops_per_layer) + (
      num_chunked_layers * chunked_attention_flops_per_layer
  )
  causal_attention_flops = noncausal_attention_flops / 2
  attention_tflops = causal_attention_flops * 3 / 10**12
  return attention_tflops


def calculate_gemma4_small_tflops_training_per_device(config, embedding_flops):
  b = config.per_device_batch_size
  s = config.max_target_length
  num_layers = config.num_decoder_layers
  num_kv_shared = config.num_kv_shared_layers
  layer_types = gemma4_small.build_layer_types(num_layers, config.model_name)
  first_shared = gemma4_small.first_kv_shared_layer_idx(num_layers, num_kv_shared)
  global_head_dim = config.global_head_dim or config.head_dim
  global_num_kv_heads = config.global_num_kv_heads or config.num_kv_heads
  num_activations = len(config.mlp_activations)
  window = min(config.sliding_window_size, s)

  def attention_flops_for(attn_type):
    if attn_type == AttentionType.GLOBAL:
      return 2 * b * s * s * config.num_query_heads * global_head_dim
    return 4 * b * (s * window - 0.5 * window**2) * config.num_query_heads * config.head_dim

  def qo_flops_for(attn_type):
    head_dim = global_head_dim if attn_type == AttentionType.GLOBAL else config.head_dim
    q = 2 * b * s * config.emb_dim * config.num_query_heads * head_dim
    o = 2 * b * s * config.emb_dim * config.num_query_heads * head_dim
    return q + o

  def kv_flops_for(attn_type):
    if attn_type == AttentionType.GLOBAL:
      kv_heads, head_dim = global_num_kv_heads, global_head_dim
    else:
      kv_heads, head_dim = config.num_kv_heads, config.head_dim
    return 2 * b * s * config.emb_dim * (2 * kv_heads) * head_dim

  attention_flops = 0
  projection_flops = 0
  ffn_per_layer = 2 * b * s * config.mlp_dim * config.emb_dim * num_activations + 2 * b * s * config.mlp_dim * config.emb_dim
  ffn_flops = 0
  for lyr_idx, attn_type in enumerate(layer_types):
    attention_flops += attention_flops_for(attn_type)
    projection_flops += qo_flops_for(attn_type)
    is_shared = num_kv_shared > 0 and lyr_idx >= first_shared
    if not is_shared:
      projection_flops += kv_flops_for(attn_type)
    if is_shared and config.use_double_wide_mlp:
      ffn_flops += 2 * ffn_per_layer
    else:
      ffn_flops += ffn_per_layer
  ple_flops = 0
  ple_dim = config.hidden_size_per_layer_input
  if ple_dim > 0:
    ple_flops += 2 * b * s * config.emb_dim * (num_layers * ple_dim)
    ple_flops += num_layers * (2 * b * s * config.emb_dim * ple_dim + 2 * b * s * ple_dim * config.emb_dim)
  attention_tflops = attention_flops * 3 / 10**12
  learnable_weight_tflops = (projection_flops + ffn_flops + ple_flops + embedding_flops) * 3 / 10**12
  return attention_tflops, learnable_weight_tflops


def calculate_ffn_mamtul_tflops_per_device(config, mlp_dim, in_dim=None):
  if in_dim is None:
    in_dim = config.emb_dim
  ffn1_flops = (
      2 * config.per_device_batch_size * config.max_target_length * mlp_dim * in_dim * len(config.mlp_activations)
  )
  ffn2_flops = 2 * config.per_device_batch_size * config.max_target_length * mlp_dim * in_dim
  return ffn1_flops + ffn2_flops


def get_shared_expert_mlp_dim(config):
  return config.mlp_dim if config.decoder_block == DecoderBlockType.GEMMA4 else config.moe_mlp_dim


def get_dense_moe_layers(config):
  if config.decoder_block in (DecoderBlockType.DEEPSEEK, DecoderBlockType.MING):
    num_dense_layers = config.first_num_dense_layers
    num_moe_layers = config.num_decoder_layers - config.first_num_dense_layers
    return num_dense_layers, num_moe_layers
  elif config.decoder_block == DecoderBlockType.LLAMA4:
    num_moe_layers = config.num_decoder_layers // config.interleave_moe_layer_step
    num_dense_layers = config.num_decoder_layers - num_moe_layers
    return num_dense_layers, num_moe_layers
  elif config.decoder_block in (DecoderBlockType.QWEN3_NEXT, DecoderBlockType.DEEPSEEK4):
    return 0, config.num_decoder_layers
  elif config.decoder_block == DecoderBlockType.DEFAULT:
    raise ValueError("Unsupported decoder block for dense/MoE layer calculation")

  num_experts = getattr(config, "num_experts", 0)
  if num_experts > 1:
    num_moe_layers = config.num_decoder_layers
    num_dense_layers = 0
  else:
    num_moe_layers = 0
    num_dense_layers = config.num_decoder_layers
  return num_dense_layers, num_moe_layers


def calculate_routed_and_shared_ffn_tflops_per_device(config):
  gate_flops = 2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.num_experts
  num_dense_layers, num_moe_layers = get_dense_moe_layers(config)
  dense_ffn_flops = calculate_ffn_mamtul_tflops_per_device(config, config.mlp_dim) * num_dense_layers
  shared_experts_flops = (
      calculate_ffn_mamtul_tflops_per_device(config, get_shared_expert_mlp_dim(config)) * config.shared_experts
  )
  routed_experts_flops = calculate_ffn_mamtul_tflops_per_device(config, config.moe_mlp_dim) * config.num_experts_per_tok
  moe_ffn_flops = (gate_flops + shared_experts_flops + routed_experts_flops) * num_moe_layers
  return dense_ffn_flops + moe_ffn_flops


def calculate_indexer_mask_ratio(indexer_topk, max_target_length):
  T = float(max_target_length)
  K = float(indexer_topk)
  ratio = K / T
  mask_multiplier = ratio - (0.5 * ratio**2)
  return mask_multiplier


def calculate_indexer_tflops_per_device(config):
  batch_len = config.per_device_batch_size * config.max_target_length
  q_flops = 2 * batch_len * config.q_lora_rank * config.indexer_n_heads * config.indexer_head_dim
  k_flops = 2 * batch_len * config.emb_dim * config.indexer_head_dim
  head_weight_flops = 2 * batch_len * config.emb_dim * config.indexer_n_heads
  proj_flops = q_flops + k_flops + head_weight_flops
  qk_product_flops = 2 * batch_len * config.max_target_length * config.indexer_n_heads * config.indexer_head_dim
  head_reduction_flops = 2 * batch_len * config.max_target_length * config.indexer_n_heads
  scoring_flops = (qk_product_flops + head_reduction_flops) / 2
  return proj_flops, scoring_flops


def calculate_mla_tflops_per_device(config):
  batch_len = config.per_device_batch_size * config.max_target_length
  qk_head_dim_sum = config.qk_nope_head_dim + config.qk_rope_head_dim
  if config.q_lora_rank == 0:
    q_flops = 2 * batch_len * config.emb_dim * config.num_query_heads * qk_head_dim_sum
  else:
    q_flops = (
        2 * batch_len
        * (config.emb_dim * config.q_lora_rank + config.q_lora_rank * config.num_query_heads * qk_head_dim_sum)
    )
  kv_flops = (
      2 * batch_len
      * (config.emb_dim * (config.kv_lora_rank + config.qk_rope_head_dim)
         + config.kv_lora_rank * config.num_query_heads * (config.qk_nope_head_dim + config.v_head_dim))
  )
  qkv_flops = q_flops + kv_flops
  if config.use_indexer and config.max_target_length > config.indexer_topk:
    indexer_proj_flops, indexer_scoring_flops = calculate_indexer_tflops_per_device(config)
    qkv_flops += indexer_proj_flops
    multiplier = calculate_indexer_mask_ratio(config.indexer_topk, config.max_target_length)
    attention_flops = (
        2 * batch_len * config.max_target_length * config.num_query_heads
        * (qk_head_dim_sum + config.v_head_dim) * multiplier
    )
    attention_flops += indexer_scoring_flops
  else:
    attention_flops = (
        2 * batch_len * config.max_target_length * config.num_query_heads * (qk_head_dim_sum + config.v_head_dim)
    )
    attention_flops = attention_flops / 2
  projection_flops = 2 * batch_len * config.emb_dim * config.num_query_heads * config.v_head_dim
  return qkv_flops, attention_flops, projection_flops


def calculate_gated_delta_net_flops_per_device(config):
  B = config.per_device_batch_size
  S = config.max_target_length
  E = config.emb_dim
  H_k = config.gdn_num_key_heads
  H_v = config.gdn_num_value_heads
  D_k = config.gdn_key_head_dim
  D_v = config.gdn_value_head_dim
  K_conv = config.gdn_conv_kernel_dim
  K_dim = H_k * D_k
  V_dim = H_v * D_v
  flops_qkvz_ba = 2 * B * S * E * (2 * K_dim + 2 * V_dim + 2 * H_v)
  flops_out = 2 * B * S * E * V_dim
  flops_projections = flops_qkvz_ba + flops_out
  flops_conv = 2 * B * S * K_conv * (2 * K_dim + V_dim)
  flops_core_per_token = H_v * (D_k * D_v) * 8
  flops_core = B * S * flops_core_per_token
  gdn_weight_flops = flops_projections + flops_conv
  gdn_attn_flops = flops_core
  return gdn_weight_flops, gdn_attn_flops


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


def calculate_engram_tflops(config):
  B = config.per_device_batch_size
  S = config.max_target_length
  G = config.mhc_expansion_rate
  D = config.emb_dim
  k = config.engram_kernel_size
  num_ngram_orders = config.engram_max_ngram_size - 1
  engram_dim = config.engram_num_heads * config.engram_head_dim * num_ngram_orders
  key_flops = 2 * (B * S) * engram_dim * (G * D)
  value_flops = 2 * (B * S) * engram_dim * D
  attention_flops = 2 * (B * S) * G * D
  total_channels = G * D
  conv_flops = 2 * (B * S) * k * total_channels
  num_layers = len(config.engram_layers)
  learnable_tflops = num_layers * (key_flops + value_flops + conv_flops) * 3 / 1e12
  attention_tflops = num_layers * attention_flops * 3 / 1e12
  return learnable_tflops, attention_tflops


def calculate_mhc_flops_per_layer(config):
  batch_size = config.per_device_batch_size
  seq_len = config.max_target_length
  k_mhc = getattr(config, "mhc_expansion_rate", 1)
  emb_dim = config.emb_dim
  pre_alpha_flops = 2 * batch_size * seq_len * (k_mhc * emb_dim) * k_mhc
  pre_contract_flops = 2 * batch_size * seq_len * k_mhc * emb_dim
  post_alpha_flops = 2 * batch_size * seq_len * (k_mhc * emb_dim) * k_mhc
  post_outer_flops = batch_size * seq_len * k_mhc * emb_dim
  res_alpha_flops = 2 * batch_size * seq_len * (k_mhc * emb_dim) * (k_mhc**2)
  res_contract_flops = 2 * batch_size * seq_len * (k_mhc**2) * emb_dim
  return pre_alpha_flops + pre_contract_flops + post_alpha_flops + post_outer_flops + res_alpha_flops + res_contract_flops


def calculate_deepseek4_tflops_training_per_device(config, total_ffn_flops_all_layers, embedding_flops):
  batch_size = config.per_device_batch_size
  seq_len = config.max_target_length
  sliding_window_size = config.sliding_window_size
  num_layers = config.num_decoder_layers
  HCA_RATIO = 128
  CSA_RATIO = 4
  c_ratios = getattr(config, "compress_ratios", [])
  if not c_ratios:
    num_sliding_layers, num_hca_layers, num_csa_layers = 2, 20, 21
  else:
    num_sliding_layers = c_ratios.count(0)
    num_hca_layers = c_ratios.count(HCA_RATIO)
    num_csa_layers = c_ratios.count(CSA_RATIO)
  wq_flops = (
      2 * batch_size * seq_len
      * ((config.emb_dim * config.q_lora_rank) + (config.q_lora_rank * config.num_query_heads * config.head_dim))
  )
  wkv_flops = 2 * batch_size * seq_len * config.emb_dim * (config.num_kv_heads * config.head_dim)
  grp_feat = (config.num_query_heads * config.head_dim) // config.o_groups
  o_flops = (2 * batch_size * seq_len * config.o_groups * grp_feat * config.o_lora_rank) + (
      2 * batch_size * seq_len * (config.o_groups * config.o_lora_rank) * config.emb_dim
  )
  base_attention_weights = wq_flops + wkv_flops + o_flops
  if getattr(config, "mhc_expansion_rate", 1) > 1:
    mhc_flops_per_layer = calculate_mhc_flops_per_layer(config)
  else:
    mhc_flops_per_layer = 0
  prefix_attn = (
      4 * batch_size * (seq_len * sliding_window_size - 0.5 * sliding_window_size**2)
      * config.num_query_heads * config.head_dim
  )
  hca_pooler = 4 * batch_size * seq_len * config.emb_dim * config.head_dim
  hca_attn = prefix_attn + 2 * batch_size * seq_len * (seq_len / HCA_RATIO) * config.num_query_heads * config.head_dim
  idx_proj = (
      2 * batch_size * seq_len
      * ((config.q_lora_rank * config.indexer_n_heads * config.indexer_head_dim) + (config.emb_dim * config.indexer_n_heads))
  )
  idx_score = batch_size * seq_len * (seq_len / CSA_RATIO) * config.indexer_n_heads * config.indexer_head_dim
  idx_reduce = batch_size * seq_len * (seq_len / CSA_RATIO) * config.indexer_n_heads
  idx_kv_gate_pool = 4 * batch_size * seq_len * config.emb_dim * (2 * config.indexer_head_dim)
  csa_indexer = idx_proj + idx_score + idx_reduce + idx_kv_gate_pool
  csa_pooler = 4 * batch_size * seq_len * config.emb_dim * (2 * config.head_dim)
  csa_k = min(config.indexer_topk, seq_len // CSA_RATIO)
  csa_mask_ratio = calculate_indexer_mask_ratio(csa_k, seq_len // CSA_RATIO)
  csa_sparse_attn = (
      4 * batch_size * (seq_len * (seq_len / CSA_RATIO)) * config.num_query_heads * config.head_dim * csa_mask_ratio
  )
  csa_attn = prefix_attn + csa_sparse_attn + csa_indexer
  total_attn_flops = (num_sliding_layers * prefix_attn) + (num_hca_layers * hca_attn) + (num_csa_layers * csa_attn)
  total_weight_flops = (
      (num_layers * base_attention_weights) + (2 * num_layers * mhc_flops_per_layer)
      + (num_hca_layers * hca_pooler) + (num_csa_layers * csa_pooler) + total_ffn_flops_all_layers + embedding_flops
  )
  return total_attn_flops * 3 / 10**12, total_weight_flops * 3 / 10**12


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
      total_ffn_flops = calculate_routed_and_shared_ffn_tflops_per_device(config)
      is_ffn_flops_already_total = True
    elif config.decoder_block == DecoderBlockType.QWEN3_CUSTOM_MOE:
      in_dim = config.moe_expert_input_dim
      gate_flops = 2 * config.per_device_batch_size * config.max_target_length * in_dim * config.num_experts
      total_ffn_flops = (
          gate_flops
          + calculate_ffn_mamtul_tflops_per_device(config, config.moe_mlp_dim, in_dim=in_dim) * config.num_experts_per_tok
      )
    else:
      gate_flops = 2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.num_experts
      total_ffn_flops = (
          gate_flops + calculate_ffn_mamtul_tflops_per_device(config, config.moe_mlp_dim) * config.num_experts_per_tok
      )
  else:
    total_ffn_flops = calculate_ffn_mamtul_tflops_per_device(config, config.mlp_dim)
  total_ffn_flops_all_layers = (
      total_ffn_flops if is_ffn_flops_already_total else total_ffn_flops * config.num_decoder_layers
  )
  if config.attention_type == "mla":
    qkv_flops, causal_attention_flops, projection_flops = calculate_mla_tflops_per_device(config)
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
    attention_tflops, learnable_weight_tflops = calculate_gemma2_tflops_training_per_device(
        config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops
    )
  elif config.decoder_block == DecoderBlockType.GEMMA3:
    attention_tflops, learnable_weight_tflops = calculate_mixed_attention_model_tflops_training_per_device(
        config, total_ffn_flops_all_layers, qkv_flops, projection_flops, embedding_flops, attention_pattern_length=6
    )
  elif config.decoder_block == DecoderBlockType.GPT_OSS:
    attention_tflops, learnable_weight_tflops = calculate_mixed_attention_model_tflops_training_per_device(
        config, total_ffn_flops_all_layers, qkv_flops, projection_flops, embedding_flops, attention_pattern_length=2
    )
  elif config.decoder_block == DecoderBlockType.LLAMA4:
    attention_tflops = calculate_llama4_attention_tflops(config)
    learnable_weight_tflops = (
        (total_ffn_flops_all_layers + (qkv_flops + projection_flops) * config.num_decoder_layers + embedding_flops)
        * 3 / 10**12
    )
  elif config.decoder_block == DecoderBlockType.GEMMA4:
    attention_tflops, learnable_weight_tflops = calculate_gemma4_tflops_training_per_device(
        config, total_ffn_flops_all_layers, embedding_flops, attention_pattern_length=6
    )
  elif config.decoder_block == DecoderBlockType.GEMMA4_SMALL:
    attention_tflops, learnable_weight_tflops = calculate_gemma4_small_tflops_training_per_device(config, embedding_flops)
  elif config.decoder_block == DecoderBlockType.DEEPSEEK4:
    attention_tflops, learnable_weight_tflops = calculate_deepseek4_tflops_training_per_device(
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
    gdn_weight_flops_per_layer, gdn_attn_flops_per_layer = calculate_gated_delta_net_flops_per_device(config)
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
            calculate_ffn_mamtul_tflops_per_device(config, get_shared_expert_mlp_dim(config)) * config.shared_experts
        )
        routed_flops = calculate_ffn_mamtul_tflops_per_device(config, config.moe_mlp_dim) * config.num_experts_per_tok
        last_layer_ffn_flops = gate_flops + shared_flops + routed_flops
      else:
        routed_flops = calculate_ffn_mamtul_tflops_per_device(config, config.moe_mlp_dim) * config.num_experts_per_tok
        last_layer_ffn_flops = gate_flops + routed_flops
    else:
      last_layer_ffn_flops = calculate_ffn_mamtul_tflops_per_device(config, config.mlp_dim)
    mtp_weight_flops = (last_layer_ffn_flops + qkv_flops + projection_flops) * mtp_num_layers
    mtp_attn_flops = causal_attention_flops * mtp_num_layers
    learnable_weight_tflops += mtp_weight_flops * 3 / 10**12
    attention_tflops += mtp_attn_flops * 3 / 10**12
  if config.engram_layers:
    engram_learnable_tflops, engram_attention_tflops = calculate_engram_tflops(config)
    learnable_weight_tflops += engram_learnable_tflops
    attention_tflops += engram_attention_tflops
  if config.use_multimodal:
    mm_total_tflops, mm_learnable_weight_tflops, mm_attention_tflops = calculate_vision_encoder_tflops(config)
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


def calculate_prefill_tflops_per_device(num_model_parameters, prefill_length, config, log=True):
  learnable_weight_tflops = 2 * num_model_parameters * prefill_length / jax.device_count() / 1e12
  noncausal_attention_flops = (
      4 * config.num_query_heads * config.num_decoder_layers * config.head_dim * prefill_length**2
      / jax.device_count() / 1e12
  )
  causal_attention_tflops = noncausal_attention_flops / 2
  total_tflops = learnable_weight_tflops + causal_attention_tflops
  if log:
    print(
        "Per prefill step per device: \n",
        f"\tTotal TFLOPs: {total_tflops:.2f} \n",
        f"\t\tLearnable weight TFLOPs: {learnable_weight_tflops:.2f} ",
        f"({100 * learnable_weight_tflops/total_tflops:.2f})% of Total\n",
        f"\t\tCausal attention TFLOPs: {causal_attention_tflops:.2f} ",
        f"({100 * causal_attention_tflops/total_tflops:.2f})% of Total",
    )
  return total_tflops, learnable_weight_tflops, causal_attention_tflops


def get_nested_value(dictionary, nested_key, default=None):
  current_level = dictionary
  for key in nested_key:
    if not isinstance(current_level, dict) or key not in current_level:
      return default
    current_level = current_level[key]
  return current_level


def collect_intermediates_by_suffix(intermediate_outputs, *suffix_keys: str, ravel: bool = True) -> list:
  suffix = tuple(suffix_keys)
  n = len(suffix)
  values = []
  for path, val in jax.tree_util.tree_leaves_with_path(intermediate_outputs):
    path_keys = tuple(k.key for k in path if hasattr(k, "key"))
    if len(path_keys) >= n and path_keys[-n:] == suffix:
      values.append(jnp.ravel(val) if ravel else val)
  return values


def get_intermediate_value(model, nested_key, default=None, clear=False):
  intermediate_value = default
  match nested_key:
    case "out_projection_activations":
      layers = model.decoder.get_layers()
      last_layer = layers[-1]
      if hasattr(last_layer, "self_attention"):
        attn = last_layer.self_attention
        if hasattr(attn, nested_key):
          var = getattr(attn, nested_key)
          intermediate_value = var.get_value()[-1]
          if clear:
            delattr(attn, nested_key)
    case _:
      raise ValueError(f"Incorrect nested_key: {nested_key}")
  return intermediate_value


def update_state_param(state, target_path, value):
  """
  Updates a specific parameter in state.params at the given path.

  Args:
      state: The current TrainState.
      target_path: A tuple of keys matching the structure inside state.params.
      value: The value to apply.
  """

  def create_jax_path(target_path):
    path = []
    for k in target_path:
      path.append(jax.tree_util.DictKey(key=k))
    return tuple(path)

  def _apply_update(path, param):
    if path == updated_target_path:
      return param + value
    return param

  updated_target_path = create_jax_path(target_path)
  new_params = jax.tree_util.tree_map_with_path(_apply_update, state.params)
  return state.replace(params=new_params)


def init_decode_state(apply_fn, params) -> TrainState:
  return TrainState(step=0, apply_fn=apply_fn, params=params, tx=None, opt_state={})


def init_training_state(apply_fn, params, tx):
  return TrainState.create(apply_fn=apply_fn, params=params, tx=tx)


def init_initial_state(model, tx, config, is_training, key):
  input_shape = (config.micro_batch_size_to_train_on, config.max_target_length)
  image_shape = mm_processor.get_dummy_image_shape_for_init(
      config.model_name, batch_size=config.micro_batch_size_to_train_on
  )
  audio_shape = mm_processor.get_dummy_audio_shape_for_init(config)
  params_key, dropout_key, aqt_key = jax.random.split(key, 3)
  model_vars = model.init(
      {"params": params_key, "dropout": dropout_key, "aqt": aqt_key},
      np.ones(input_shape, dtype=jnp.int32),
      np.ones(input_shape, dtype=jnp.int32),
      encoder_images=np.ones(image_shape, dtype=jnp.int32) if config.use_multimodal else None,
      encoder_audios=np.ones(audio_shape, dtype=jnp.float32) if config.use_audio else None,
  )
  if is_training:
    return init_training_state(model.apply, model_vars, tx)
  return init_decode_state(model.apply, model_vars)


def print_shardings_params(params, params_sharding, mesh, logical_annotations=None):
  """Print state shardings comparing logical definition and physical result."""
  if not isinstance(params, nnx.State):
    if not hasattr(params, "params"):
      params = {"params": params}
    if not hasattr(params_sharding, "params"):
      params_sharding = {"params": params_sharding}
    if logical_annotations and not hasattr(logical_annotations, "params"):
      logical_annotations = {"params": logical_annotations}
  leaves_params, _ = jax.tree_util.tree_flatten_with_path(params)
  leaves_sharding, _ = jax.tree_util.tree_flatten_with_path(params_sharding)
  if logical_annotations is not None:
    leaves_logical, _ = jax.tree_util.tree_flatten_with_path(logical_annotations)
    for (path, leaf_val), (_, leaf_sharding), (_, leaf_logical_val) in zip(
        leaves_params, leaves_sharding, leaves_logical
    ):
      path_str = "/".join(str(part.key if hasattr(part, "key") else part.name) for part in path)
      shape = jax.typeof(leaf_val)
      pspec = sharding.remove_size_one_mesh_axis(leaf_sharding.spec, mesh)
      pspec_str = str(tuple(pspec))
      logical_str = str(leaf_logical_val)
      message = (
          f" {path_str}\n" f"    Shape:     {shape}\n" f"    Logical:   {logical_str}\n" f"    Physical:  {pspec_str}"
      )
      max_logging.info(message)
  else:
    for (path, leaf_val), (_, leaf_sharding) in zip(leaves_params, leaves_sharding):
      path_str = "/".join(str(part.key if hasattr(part, "key") else part.name) for part in path)
      shape = jax.typeof(leaf_val)
      pspec = sharding.remove_size_one_mesh_axis(leaf_sharding.spec, mesh)
      pspec_str = str(tuple(pspec))
      message = f" {path_str}\n" f"    Shape:     {shape}\n" f"    Physical:  {pspec_str}"
      max_logging.info(message)
  print(flush=True)


def create_device_mesh(config, devices=None):
  if devices is None:
    devices = jax.devices()

  if getattr(config, "elastic_enabled", False):
    devices = elastic_utils.live_devices(config)
    num_slices = len(elastic_utils.live_slice_indices(config))
  else:
    num_slices = config.num_slices

  if config.subslice_shape and config.enable_single_controller and num_slices == 1:
    max_logging.log(f"Trying to create a subslice with shape: {config.subslice_shape}")
    subslice_shape = tuple(int(x) for x in config.subslice_shape.split(","))
    device_coords = [device.coords for device in devices]
    device_coords_np = np.array(device_coords)
    min_coords = device_coords_np.min(axis=0)
    subslice_devices = []
    for device in devices:
      coords = device.coords
      if all(min_coords[i] <= coords[i] < min_coords[i] + subslice_shape[i] for i in range(len(subslice_shape))):
        subslice_devices.append(device)
    devices = subslice_devices

  num_devices = len(devices)
  num_slices = 1 if getattr(config, "inference_benchmark_test", False) else num_slices
  num_devices_per_slice = num_devices // num_slices
  ici_parallelism = config.ici_parallelism.copy()
  ici_parallelism = max_utils.fill_unspecified_mesh_axes(ici_parallelism, num_devices_per_slice, "ICI")
  allow_split_physical_axes = config.allow_split_physical_axes if config.allow_split_physical_axes else False

  if getattr(config, "use_lineage", False):
    dcn_parallelism = (
        max_utils.fill_unspecified_mesh_axes(config.dcn_parallelism.copy(), num_slices, "DCN") if num_slices > 1 else None
    )
    mesh = lineage_adapter.create_device_mesh(
        ici_parallelism, devices, dcn_parallelism=dcn_parallelism, allow_split_physical_axes=allow_split_physical_axes
    )
  elif num_slices > 1:
    dcn_parallelism = config.dcn_parallelism.copy()
    dcn_parallelism = max_utils.fill_unspecified_mesh_axes(dcn_parallelism, num_slices, "DCN")
    if max_utils.is_valid_custom_mesh(ici_parallelism, config.custom_mesh):
      mesh = max_utils.create_custom_device_mesh(ici_parallelism, dcn_parallelism, devices, config.custom_mesh)
    else:
      mesh = mesh_utils.create_hybrid_device_mesh(
          ici_parallelism, dcn_parallelism, devices, allow_split_physical_axes=allow_split_physical_axes
      )
  elif allow_split_physical_axes:
    if max_utils.is_valid_custom_mesh(ici_parallelism, config.custom_mesh):
      mesh = mesh_utils.create_device_mesh([16, 16], devices, contiguous_submeshes=False, allow_split_physical_axes=False)
      mesh = max_utils.reshape_mesh_to_rings(mesh, config.custom_mesh)
      mesh = np.reshape(mesh, ici_parallelism)
    else:
      mesh = mesh_utils.create_device_mesh(
          ici_parallelism, devices, contiguous_submeshes=False, allow_split_physical_axes=allow_split_physical_axes
      )
  else:
    mesh = mesh_utils.create_device_mesh(ici_parallelism, devices)
    if config.optimize_mesh_for_tpu_v6e:
      mesh = max_utils.optimize_mesh_for_tpu_v6e(mesh, devices)

  max_logging.log(f"Num_devices: {num_devices}, shape {mesh.shape}")
  return mesh


def get_mesh_from_config(config, devices=None):
  devices_array = create_device_mesh(config, devices)
  if config.shard_mode == ShardMode.EXPLICIT:
    axis_types = tuple([AxisType.Explicit] * len(config.mesh_axes))
  else:
    axis_types = tuple([AxisType.Auto] * len(config.mesh_axes))
  return Mesh(devices_array, config.mesh_axes, axis_types=axis_types)


def setup_training_state(
    data_iterator, config, mesh, checkpoint_manager, init_state_fn, snapshotter=None, step=-1, selected_source=None
):
  return setup_initial_state(
      data_iterator, config, mesh, checkpoint_manager, init_state_fn, True, snapshotter=snapshotter, step=step,
      selected_source=selected_source,
  )


def setup_initial_state(
    data_iterator, config, mesh, checkpoint_manager, init_state_fn, is_training=True, snapshotter=None, step=-1,
    selected_source=None,
):
  unboxed_abstract_state, state_mesh_annotations, state_mesh_shardings = get_abstract_state(
      config, mesh, init_state_fn, is_training
  )

  with axis_rules(config.logical_axis_rules):
    native = lineage_adapter.is_native(config)
    if native and selected_source is not None and selected_source.kind == "parameters":
      restored, raw_params = None, None
    else:
      restored, raw_params = checkpointing.load_state_if_possible(
          checkpoint_manager, data_iterator, "" if native else config.load_parameters_path, config.load_full_state_path,
          config.checkpoint_storage_concurrent_gb, unboxed_abstract_state, config.enable_single_replica_ckpt_restoring,
          config.dataset_type, step=step, use_ocdbt=config.checkpoint_storage_use_ocdbt,
          use_zarr3=config.checkpoint_storage_use_zarr3, enable_orbax_v1=config.enable_orbax_v1,
          checkpoint_conversion_fn=config.checkpoint_conversion_fn, source_checkpoint_layout=config.source_checkpoint_layout,
          expansion_factor_real_data=config.expansion_factor_real_data, maxtext_config=config, snapshotter=snapshotter,
          selected_source=selected_source,
      )
    if (native and restored is None and raw_params is None and config.load_parameters_path
        and (selected_source is None or selected_source.kind == "parameters")):
      raw_params = checkpointing.load_params_from_path(
          selected_source.path if selected_source is not None else config.load_parameters_path,
          lineage_adapter.maxtext_restore_target(config, mesh),
          config.checkpoint_storage_concurrent_gb, use_ocdbt=config.checkpoint_storage_use_ocdbt,
          use_zarr3=config.checkpoint_storage_use_zarr3,
          enable_single_replica_ckpt_restoring=bool(config.enable_single_replica_ckpt_restoring),
          colocated_python_checkpointing=bool(config.enable_single_controller and config.colocated_python_checkpointing),
      )
    was_restored = bool(restored is not None or raw_params is not None)
    init_state_partial = init_state_fn
    init_state_partial.__name__ = "initialize_state"
    state = jax.jit(lambda: nnx.state(init_state_partial()), in_shardings=None, out_shardings=state_mesh_shardings)()
    if raw_params and native:
      lineage_adapter.restore_params(state["model"], raw_params, config, mesh)
    elif raw_params:
      target_model = (
          state["model"] if (isinstance(state, (nnx.State, dict)) and "model" in state) else getattr(state, "model", state)
      )
      raw_model_params = (
          raw_params["model"] if (isinstance(raw_params, (nnx.State, dict)) and "model" in raw_params) else raw_params
      )
      if hasattr(raw_model_params, "to_pure_dict"):
        raw_model_params = raw_model_params.to_pure_dict()
      if isinstance(raw_model_params, dict) and "params" in raw_model_params:
        raw_model_params = raw_model_params["params"]
      target_pure = target_model.to_pure_dict() if hasattr(target_model, "to_pure_dict") else target_model

      def _reshard_aligned(target, raw):
        target_flat = traverse_util.flatten_dict(target)
        raw_flat = traverse_util.flatten_dict(raw)
        res_flat = {}
        for k, target_val in target_flat.items():
          if k in raw_flat and not isinstance(raw_flat[k], jax.ShapeDtypeStruct):
            raw_val = raw_flat[k]
            if hasattr(target_val, "sharding") and target_val.sharding is not None:
              res_flat[k] = jax.device_put(raw_val, target_val.sharding)
            else:
              res_flat[k] = raw_val
          else:
            res_flat[k] = target_val
        return traverse_util.unflatten_dict(res_flat)

      sharded_aligned = _reshard_aligned(target_pure, raw_model_params)
      nnx.update(target_model, sharded_aligned)

    if restored:
      is_emergency = isinstance(
          checkpoint_manager,
          (emergency_checkpoint_manager.CheckpointManager,
           emergency_replicator_checkpoint_manager.ReplicatorCheckpointManager),
      )
      overlay = restored if is_emergency else restored["items"]
      if not (config.enable_diloco and isinstance(overlay, diloco.DiLoCoTrainState)):
        overlay_pure_dict = overlay.to_pure_dict() if hasattr(overlay, "to_pure_dict") else overlay

        def _has_shape_dtype_struct(tree):
          return any(isinstance(x, jax.ShapeDtypeStruct) for x in jax.tree_util.tree_leaves(tree))

        def _merge_restored_overlay(ckpt_node, init_node):
          if _has_shape_dtype_struct(ckpt_node):
            if isinstance(ckpt_node, dict) and isinstance(init_node, dict):
              res = {}
              for k in init_node:
                if k in ckpt_node:
                  res[k] = _merge_restored_overlay(ckpt_node[k], init_node[k])
                else:
                  res[k] = init_node[k]
              return res
            else:
              return init_node
          return ckpt_node

        merged = _merge_restored_overlay(overlay_pure_dict, state.to_pure_dict())
        nnx.replace_by_pure_dict(state, merged)
      else:
        state = overlay

  if config.enable_diloco:
    state = diloco_utils.setup_diloco_initial_state(
        state=state, config=config, mesh=mesh, state_mesh_shardings=state_mesh_shardings, restored=restored
    )
  return state, state_mesh_annotations, state_mesh_shardings, data_iterator, was_restored


def setup_decode_state(config, mesh, checkpoint_manager, init_state_fn):
  """Set up a Linen decode state from a checkpoint or random initialization."""
  if not config.load_parameters_path:
    max_logging.log("No decode checkpoint specified - generating random weights.")
    abstract_state, state_mesh_annotations, state_mesh_shardings = get_abstract_state(config, mesh, init_state_fn, False)
    with axis_rules(config.logical_axis_rules):
      restored, raw_params = checkpointing.load_state_if_possible(
          checkpoint_manager, None, config.load_parameters_path, config.load_full_state_path,
          config.checkpoint_storage_concurrent_gb, abstract_state, config.enable_single_replica_ckpt_restoring,
          config.dataset_type, use_ocdbt=config.checkpoint_storage_use_ocdbt,
          use_zarr3=config.checkpoint_storage_use_zarr3, enable_orbax_v1=config.enable_orbax_v1,
          checkpoint_conversion_fn=config.checkpoint_conversion_fn, source_checkpoint_layout=config.source_checkpoint_layout,
          expansion_factor_real_data=config.expansion_factor_real_data, maxtext_config=config,
      )
      if restored:
        if isinstance(
            checkpoint_manager,
            (emergency_checkpoint_manager.CheckpointManager,
             emergency_replicator_checkpoint_manager.ReplicatorCheckpointManager),
        ):
          state = restored
        else:
          state = restored["items"]
      else:
        state = jax.jit(init_state_fn, in_shardings=None, out_shardings=state_mesh_shardings)()
        if config.weight_sparsity_n and config.weight_sparsity_m and raw_params:
          params = jax.tree_util.tree_map(
              lambda raw, initial: initial if isinstance(raw, jax.ShapeDtypeStruct) else raw, raw_params, state.params
          )
          state = state.replace(params=params)
        elif raw_params:
          state = state.replace(params=raw_params)
  else:
    max_logging.log(f"Loading decode params from {config.load_parameters_path}")
    abstract_state, state_mesh_annotations, _ = get_abstract_state(config, mesh, init_state_fn, False)
    with axis_rules(config.logical_axis_rules):
      params = checkpointing.load_params_from_path(
          config.load_parameters_path, abstract_state.params, config.checkpoint_storage_concurrent_gb,
          config.checkpoint_storage_use_ocdbt, config.checkpoint_storage_use_zarr3,
      )
    state = init_decode_state(None, params)

  state = max_utils.unbox_logicallypartioned(state)
  return state, state_mesh_annotations


def get_abstract_state(config, mesh, init_state_fn, is_training=True):
  if isinstance(init_state_fn, functools.partial) and init_state_fn.func is init_initial_state:
    with axis_rules(config.logical_axis_rules):
      abstract_state = jax.eval_shape(init_state_fn)
      logical_annotations = get_partition_spec(abstract_state)
      state_mesh_annotations = sharding.logical_to_mesh(logical_annotations, mesh)
      state_mesh_shardings = sharding.logical_to_mesh_sharding(logical_annotations, mesh)
      unboxed_abstract_state = max_utils.unbox_logicallypartioned(abstract_state)
    return unboxed_abstract_state, state_mesh_annotations, state_mesh_shardings
  return get_abstract_state_nnx(config, mesh, init_state_fn, is_training)


def get_abstract_state_nnx(config, mesh, nnx_init_trainstate_fn, is_training=True):
  assert nnx_init_trainstate_fn is not None, "get_abstract_state_nnx: init function must be given."
  with axis_rules(config.logical_axis_rules):
    abs_model = nnx.eval_shape(nnx_init_trainstate_fn)
    _, abs_var_state = nnx.split(abs_model)
    named_sharding_state = sharding.nnx_construct_named_sharding(abs_var_state, mesh)

    def _to_abstract_var(a_var, s_var):
      a_val = a_var.get_value()
      s_val = getattr(s_var, "sharding", None) if isinstance(s_var, nnx.Variable) else s_var
      if s_val is None and isinstance(s_var, nnx.Variable):
        s_val = s_var.get_value()

      def _extract_primary_sharding(s):
        if isinstance(s, (jax.sharding.Sharding, jax.sharding.PartitionSpec)):
          return s
        if hasattr(s, "qvalue"):
          return _extract_primary_sharding(s.qvalue)
        leaves = jax.tree.leaves(s)
        return leaves[0] if leaves else s

      def _make_abstract_leaf(leaf_a, leaf_s):
        leaf_s = _extract_primary_sharding(leaf_s)
        if hasattr(leaf_s, "spec") and len(leaf_s.spec) > len(leaf_a.shape):
          if "local_layers" in leaf_s.spec and len(leaf_s.spec) - 1 == len(leaf_a.shape):
            spec_tuple = tuple(axis for axis in leaf_s.spec if axis != "local_layers")
          else:
            spec_tuple = leaf_s.spec[: len(leaf_a.shape)]
          leaf_s = jax.sharding.NamedSharding(leaf_s.mesh, jax.sharding.PartitionSpec(*spec_tuple))
        return jax.ShapeDtypeStruct(leaf_a.shape, leaf_a.dtype, sharding=leaf_s)

      if type(a_val) in (jax.Array, jax.ShapeDtypeStruct) or (hasattr(a_val, "shape") and not hasattr(a_val, "qvalue")):
        new_val = _make_abstract_leaf(a_val, s_val)
      else:
        s_tree = (
            jax.tree.map(lambda _: s_val, a_val)
            if isinstance(s_val, (jax.sharding.Sharding, jax.sharding.PartitionSpec)) else s_val
        )
        new_val = jax.tree.map(
            _make_abstract_leaf, a_val, s_tree, is_leaf=lambda x: hasattr(x, "shape") and hasattr(x, "dtype")
        )
      return a_var.replace(value=new_val)

    abstract_state = jax.tree.map(
        _to_abstract_var, abs_var_state, named_sharding_state, is_leaf=lambda x: isinstance(x, nnx.Variable)
    )

  state_mesh_shardings = maxtext_utils_nnx.nnx_extract_named_sharding(abstract_state)
  if is_training and config.shard_optimizer_over_data:
    optimizer_sharding = jax.tree_util.tree_map_with_path(
        functools.partial(sharding.add_data_to_sharding, mesh), abstract_state.optimizer, state_mesh_shardings.optimizer
    )
    state_mesh_shardings.optimizer = optimizer_sharding
  if is_training and config.optimizer_memory_host_offload:
    optimizer_sharding = jax.tree_util.tree_map_with_path(
        maxtext_utils_nnx.move_memory_to_host, state_mesh_shardings.optimizer,
        is_leaf=lambda x: isinstance(x, NamedSharding),
    )
    state_mesh_shardings.optimizer = optimizer_sharding
  if is_training and config.parameter_memory_host_offload:
    assert config.param_scan_axis == 0, "You must set the scan axis 0 to enable parameter offloading."
    _, state_params, _ = nnx.split(state_mesh_shardings, nnx.Param, ...)
    state_params = jax.tree_util.tree_map_with_path(
        maxtext_utils_nnx.move_memory_to_host, state_params, is_leaf=lambda x: isinstance(x, NamedSharding)
    )
    nnx.update(state_mesh_shardings, state_params)
  abstract_sharded_state = maxtext_utils_nnx.set_named_sharding_nnx(abstract_state, state_mesh_shardings)
  state_mesh_annotations = maxtext_utils_nnx.get_partition_spec_nnx(state_mesh_shardings)
  return abstract_sharded_state, state_mesh_annotations, state_mesh_shardings


def _nnx_cache_partition_specs(abstract_model, config, mesh):
  _, cache_state, _ = nnx.split(abstract_model, nnx.Cache, ...)
  with axis_rules(config.logical_axis_rules):
    named_state = sharding.nnx_construct_named_sharding(cache_state, mesh)
  return jax.tree.map(lambda s: s.spec, named_state.to_pure_dict())


def get_prefill_kv_cache_annotations_nnx(abstract_model, config, mesh):
  return _nnx_cache_partition_specs(abstract_model, config, mesh)


def get_kv_cache_annotations_nnx(abstract_model, config, mesh):
  return _nnx_cache_partition_specs(abstract_model, config, mesh)


def save_quantized_checkpoint_if_configured(config, params):
  """Save quantized checkpoint if configured"""
  assert config.quantization, "quantization must be configured"
  if config.save_quantized_params_path:
    checkpointing.save_params_to_path(
        checkpoint_dir=config.save_quantized_params_path,
        params=params,
        use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3,
    )
  else:
    max_logging.log("Skipping saving quantized checkpoint as save_quantized_params_path is null.")


def add_config_to_summary_writer(config, summary_writer):
  if jax.process_index() == 0:
    if hasattr(config, "get_keys"):
      config_dict = config.get_keys()
    elif hasattr(config, "model_dump"):
      config_dict = config.model_dump()
    else:
      config_dict = dict(config)
    for key, value in config_dict.items():
      max_utils.add_text_to_summary_writer(key, str(value), summary_writer)


def maybe_dump_jaxpr(config, p_train_step, train_step_inputs):
  if not config.dump_jaxpr:
    return
  max_logging.log("Tracing train_step to jaxpr...")
  unwrapped_step = getattr(p_train_step, "__wrapped__", p_train_step)

  def to_abstract(x):
    if hasattr(x, "shape") and hasattr(x, "dtype"):
      return jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
    return x

  abstract_inputs = jax.tree.map(to_abstract, train_step_inputs)
  p_train_jaxpr = jax.make_jaxpr(unwrapped_step)(*abstract_inputs)
  local_filename = "train_step.jaxpr"
  local_path = os.path.join(config.dump_jaxpr_local_dir, local_filename)
  os.makedirs(config.dump_jaxpr_local_dir, exist_ok=True)
  with open(local_path, "w") as f:
    f.write(str(p_train_jaxpr))
  max_logging.log(f"Jaxpr dumped locally to {local_path}")
  if config.dump_jaxpr_gcs_dir:
    gcs_utils.upload_dump(
        config.dump_jaxpr_local_dir,
        config.dump_jaxpr_gcs_dir,
        module_name=local_filename,
        delete_local_after=config.dump_jaxpr_delete_local_after,
        all_host_upload=False,
    )


def prepare_kv_caches_for_scan(kv_caches, scan_length, block_len, stack=False):
  if kv_caches is None:
    return None
  if not isinstance(kv_caches, list):
    raise TypeError(f"kv_caches must be a list, got {type(kv_caches)}")
  grouped = [tuple(kv_caches[i * block_len : (i + 1) * block_len]) for i in range(scan_length)]
  if stack:
    return jax.tree_util.tree_map(lambda *args: jnp.stack(args, axis=0), *grouped)
  return grouped


def update_kv_caches_after_scan(kv_caches, returned_kv_cache, scan_length, block_len, stacked=False):
  if kv_caches is None or returned_kv_cache is None:
    return
  if not isinstance(kv_caches, list):
    raise TypeError(f"kv_caches must be a list, got {type(kv_caches)}")
  if stacked:
    unstacked = [jax.tree_util.tree_map(lambda x, i=i: x[i], returned_kv_cache) for i in range(scan_length)]
    for i in range(scan_length):
      start_idx = i * block_len
      for offset, updated_item in enumerate(unstacked[i]):
        kv_caches[start_idx + offset] = updated_item
  else:
    for i in range(scan_length):
      start_idx = i * block_len
      for offset, updated_item in enumerate(returned_kv_cache[i]):
        kv_caches[start_idx + offset] = updated_item
