"""Convert a training checkpoint to a parameter-only decode checkpoint."""

import os.path

from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh

from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.common import state_initialization, train_state_nnx
from lilax._backend.common.common_types import DecoderBlockType
from lilax._backend.utils import gcs_utils, lora_utils, max_logging, max_utils
from lilax._backend.utils import device_mesh, maxtext_utils_nnx, model_creation_utils, train_utils
from lilax._backend.utils.training import state_setup

_SCAN_AXIS_NAMES = ("layers_per_stage", "layers", "circular_repeats")


def get_scan_axis(config, sharding_or_spec, x=None):
  spec = sharding_or_spec.spec if hasattr(sharding_or_spec, "spec") else sharding_or_spec
  for name in _SCAN_AXIS_NAMES:
    if name in spec:
      return spec.index(name)
  if x is not None and hasattr(x, "ndim") and x.ndim > config.param_scan_axis:
    return config.param_scan_axis
  return None


def _possibly_unroll_params(config, training_state, training_state_annotations, mesh):
  if config.scan_layers and config.force_unroll:
    _possibly_unroll_params_nnx(config, training_state, training_state_annotations, mesh)


def _possibly_unroll_params_nnx(config, state, state_mesh_shardings, mesh):
  param_state, other_state = nnx.split_state(state, nnx.Param, ...)
  param_paths = set(param_state.flat_state().paths)
  param_shardings = nnx.State.from_flat_path(
      [(path, value) for path, value in state_mesh_shardings.flat_state() if path in param_paths]
  )
  other_shardings = nnx.State.from_flat_path(
      [(path, value) for path, value in state_mesh_shardings.flat_state() if path not in param_paths]
  )
  decoder = param_state.model.decoder
  shardings = param_shardings.model.decoder

  def unroll(num_layers, name):
    layers = decoder.get(name)
    layer_shardings = shardings.get(name)
    if layers is None or layer_shardings is None:
      raise ValueError(f"Missing {name} in NNX state.model.decoder or state_mesh_shardings.")

    def drop_axis(sharding, x):
      spec = sharding.spec
      value = x[...] if isinstance(x, nnx.Variable) else x
      axis = get_scan_axis(config, sharding, value)
      return jax.sharding.PartitionSpec(*(spec[:axis] + spec[axis + 1 :])) if axis is not None else spec

    specs = jax.tree_util.tree_map(
        drop_axis, layer_shardings, layers,
        is_leaf=lambda x: isinstance(x, (jax.sharding.NamedSharding, nnx.Variable)),
    )
    new_sharding = jax.tree_util.tree_map(lambda spec: jax.sharding.NamedSharding(mesh, spec), specs)
    for i in range(num_layers):
      def slice_layer(input_layers):
        def slice_leaf(x, sharding):
          value = x[...] if isinstance(x, nnx.Variable) else x
          axis = get_scan_axis(config, sharding, value)
          if axis is None:
            return x
          value = jnp.take(value, i, axis=axis)
          return type(x)(value) if isinstance(x, nnx.Variable) else value
        return jax.tree_util.tree_map(
            slice_leaf, input_layers, layer_shardings,
            is_leaf=lambda x: isinstance(x, (jax.sharding.NamedSharding, nnx.Variable)),
        )
      decoder[f"{name}_{i}"] = jax.jit(slice_layer, out_shardings=new_sharding)(layers)
      shardings[f"{name}_{i}"] = new_sharding
    decoder.pop(name)
    shardings.pop(name)
    jax.tree_util.tree_map(lambda x: x.delete() if hasattr(x, "delete") else None, layers)

  if config.decoder_block == DecoderBlockType.DEEPSEEK:
    unroll(config.first_num_dense_layers, "dense_layers")
    unroll(config.num_decoder_layers - config.first_num_dense_layers, "moe_layers")
  else:
    unroll(config.num_decoder_layers, "layers")
  state.update(nnx.merge_state(param_state, other_state))
  state_mesh_shardings.update(nnx.merge_state(param_shardings, other_shardings))


def _read_train_checkpoint(config, checkpoint_manager, mesh):
  rngs = maxtext_utils_nnx.create_nnx_rngs(config, rng_key=random.PRNGKey(0))
  model = model_creation_utils.from_config(config, mesh=mesh, rngs=rngs)
  _, tx = train_utils.create_training_optimizer(config, model)
  create_model, _ = model_creation_utils.create_nnx_abstract_model(config, mesh)

  def init_state_fn():
    model = create_model()
    wrt = getattr(nnx, "LoRAParam", nnx.Param) if getattr(getattr(config, "lora", None), "enable_lora", False) else nnx.Param
    return train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, tx, wrt=wrt))

  state, _, shardings, _, _ = state_setup.setup_training_state(None, config, mesh, checkpoint_manager, init_state_fn)
  params, _ = nnx.split_state(state.model, nnx.Param, ...)
  num_params = max_utils.calculate_num_params_from_pytree(params)
  max_logging.log(f"In input checkpoint Number of model params={num_params/1e9:.3f} billion")
  return state, shardings


def _save_decode_checkpoint_nnx(config, state, checkpoint_manager):
  _, params, _ = nnx.split(state.model, nnx.Param, ...)
  model = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params.to_pure_dict())

  def wrap(node):
    return {key: wrap(value) for key, value in node.items()} if isinstance(node, dict) else {"value": node}

  if checkpoint_manager is not None:
    if checkpointing.save_checkpoint(checkpoint_manager, 0, wrap(model)):
      max_logging.log(f"saved an NNX decode checkpoint at {config.checkpoint_dir}")
    checkpointing.wait_until_finished(checkpoint_manager)


def _save_decode_checkpoint(config, state, checkpoint_manager):
  _save_decode_checkpoint_nnx(config, state, checkpoint_manager)


def _possibly_unroll_lora_params_nnx(config, state, annotations, mesh):
  if not config.scan_layers or not config.force_unroll:
    return
  decoder = state.params["decoder"]
  decoder_annotations = annotations.params["decoder"]

  def unroll(num_layers, name):
    layers = decoder.get(name)
    specs = decoder_annotations.get(name)
    if layers is None or specs is None:
      return

    def drop_axis(spec, x):
      axis = get_scan_axis(config, spec, x)
      return jax.sharding.PartitionSpec(*(spec[:axis] + spec[axis + 1 :])) if axis is not None else spec

    new_specs = jax.tree_util.tree_map(drop_axis, specs, layers, is_leaf=lambda x: isinstance(x, jax.sharding.PartitionSpec))
    sharding = jax.tree_util.tree_map(lambda spec: jax.sharding.NamedSharding(mesh, spec), new_specs)
    for i in range(num_layers):
      def slice_layer(input_layers):
        def slice_leaf(x, spec):
          axis = get_scan_axis(config, spec, x)
          return jnp.take(x, i, axis=axis) if axis is not None else x
        return jax.tree_util.tree_map(
            slice_leaf, input_layers, specs, is_leaf=lambda x: isinstance(x, jax.sharding.PartitionSpec)
        )
      decoder[f"{name}_{i}"] = jax.jit(slice_layer, out_shardings=sharding)(layers)
      decoder_annotations[f"{name}_{i}"] = new_specs
    del decoder[name]
    del decoder_annotations[name]
    jax.tree_util.tree_map(lambda x: x.delete() if hasattr(x, "delete") else None, layers)

  if config.decoder_block == DecoderBlockType.DEEPSEEK:
    unroll(config.first_num_dense_layers, "dense_layers")
    unroll(config.num_decoder_layers - config.first_num_dense_layers, "moe_layers")
  else:
    unroll(config.num_decoder_layers, "layers")


def _save_lora_decode_checkpoint_nnx(config, state, checkpoint_manager):
  decode_state = state_initialization.init_decode_state(None, jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), state.params))
  if checkpoint_manager is not None:
    if checkpointing.save_checkpoint(checkpoint_manager, 0, decode_state):
      max_logging.log(f"saved a LoRA decode checkpoint at {config.checkpoint_dir}")
    checkpointing.wait_until_finished(checkpoint_manager)


def _generate_lora_decode_checkpoints_nnx(config, mesh):
  rngs = maxtext_utils_nnx.create_nnx_rngs(config, rng_key=random.PRNGKey(0))
  model = model_creation_utils.from_config(config, mesh=mesh, rngs=rngs)
  _, tx = train_utils.create_training_optimizer(config, model)
  for lora_id in gcs_utils.gcs_list_directories(config.lora_input_adapters_path):
    directory = os.path.join(config.checkpoint_dir, "loras", lora_id, "")
    adapter_path = os.path.join(config.lora_input_adapters_path, lora_id, "")
    manager = checkpointing.create_orbax_checkpoint_manager(
        directory, config.enable_checkpointing, config.async_checkpointing, config.checkpoint_period
    )
    lora_config, state, annotations = lora_utils.setup_initial_lora_state(None, tx, config, mesh, manager, adapter_path)
    _possibly_unroll_lora_params_nnx(config, state, annotations, mesh)
    gcs_utils.write_dict_to_gcs_json(lora_config, os.path.join(directory, "adapter_config.json"))
    _save_lora_decode_checkpoint_nnx(config, state, manager)
    max_logging.log(f"Successfully saved LoRA checkpoint at: {os.path.join(directory, '0', 'items')}")


def _generate_lora_decode_checkpoints(config, mesh):
  _generate_lora_decode_checkpoints_nnx(config, mesh)


def generate_decode_checkpoint(config):
  mesh = Mesh(device_mesh.create_device_mesh(config), config.mesh_axes)
  assert config.checkpoint_dir, "checkpoint_dir not configured"
  path = epath.Path(config.checkpoint_dir)
  if path.exists() and jax.process_index() == 0:
    path.rmtree()
  directory = config.checkpoint_dir + "base/" if config.lora_input_adapters_path else config.checkpoint_dir
  manager = checkpointing.create_orbax_checkpoint_manager(
      directory, config.enable_checkpointing, config.async_checkpointing, config.checkpoint_period,
      use_ocdbt=config.checkpoint_storage_use_ocdbt, use_zarr3=config.checkpoint_storage_use_zarr3,
  )
  max_logging.log(f"Read training checkpoint from: {config.load_full_state_path}")
  state, annotations = _read_train_checkpoint(config, manager, mesh)
  assert state.optimizer.opt_state, "missing opt_state in training checkpoint"
  _possibly_unroll_params(config, state, annotations, mesh)
  max_logging.log(f"Save decode checkpoint at: {directory}")
  _save_decode_checkpoint(config, state, manager)
  max_logging.log(f"Successfully generated decode checkpoint at: {directory}0/items")
  if config.lora_input_adapters_path:
    _generate_lora_decode_checkpoints(config, mesh)
  return True
