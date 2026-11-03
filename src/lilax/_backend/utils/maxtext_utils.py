"""Native train-step signatures, batch shapes, and training-state utilities."""

import functools

from flax import nnx
from flax.core.spmd import logical_axis_rules as axis_rules
import jax
from jax.experimental.serialize_executable import deserialize_and_load
import jax.numpy as jnp
import optax

from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.common.common_types import ReorderStrategy
from lilax._backend.multimodal import processor as mm_processor
from lilax._backend.optimizers.schedules import create_learning_rate_schedule
from lilax._backend.utils import max_logging
from lilax._backend.distributed import sharding
from lilax._backend.layers.context_parallel import sequence_reorder

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
  reordered = sequence_reorder.reorder_causal_load_balanced(batch, cp_size, reorder_strategy, hardware)
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
