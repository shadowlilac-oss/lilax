"""Common LoRA utilities for Linen, NNX, and Qwix adapters."""

from collections.abc import Mapping
from functools import lru_cache
import json
import os
import re
from typing import Optional

from flax import nnx
from flax.core.spmd import logical_axis_rules
from flax.training import train_state
import jax
import jax.numpy as jnp
from orbax import checkpoint as ocp
import qwix

from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.common import train_state_nnx
from lilax._backend.configs import pyconfig
from lilax._backend.utils import device_mesh, gcs_utils, max_logging, state_abstraction
from lilax._backend.utils.globals import MAXTEXT_CONFIGS_DIR
from lilax._backend.utils.model_creation import nnx_abstract_model
from lilax._backend.utils.tree_ops import partitioning


def apply_lora_on_base_params(base_params, lora_params, lora_scale_factor=1.0):
  """Apply Linen LoRA weights to base parameters in place."""

  def lora_update_or_base(base_weight, lora_a, lora_b):
    if lora_a is not None and lora_b is not None:
      delta = (jnp.einsum("br,rnd->bnd", lora_b, lora_a) * lora_scale_factor).astype(base_weight.dtype)
      return (base_weight + delta).astype(base_weight.dtype)
    return base_weight

  def apply_lora_recursively(base_params, lora_params, module_name):
    for name, param in lora_params.items():
      if isinstance(param, dict):
        apply_lora_recursively(base_params[name], param, f"{module_name}.{name}")
      elif param is not None:
        if name not in ["lora_a.kernel", "lora_b.kernel"]:
          raise ValueError(f"Unexpected non-lora specific weights ({module_name}.{name}) found in the lora_params")
        lora_b = lora_params["lora_a.kernel"]
        lora_a = lora_params["lora_b.kernel"]
        base_params["kernel"] = lora_update_or_base(base_params["kernel"], lora_a, lora_b)
        break

  apply_lora_recursively(base_params, lora_params, "")


def unapply_lora_from_base_params(base_params, lora_params, lora_scale_factor=1.0):
  """Remove Linen LoRA weights from base parameters in place."""

  def lora_update_or_base(base_weight, lora_a, lora_b):
    if lora_a is not None and lora_b is not None:
      delta = (jnp.einsum("br,rnd->bnd", lora_b, lora_a) * lora_scale_factor).astype(base_weight.dtype)
      return (base_weight - delta).astype(base_weight.dtype)
    return base_weight

  def unapply_lora_recursively(base_params, lora_params, module_name):
    for name, param in lora_params.items():
      if isinstance(param, dict):
        unapply_lora_recursively(base_params[name], param, f"{module_name}.{name}")
      elif param is not None:
        if name not in ["lora_a.kernel", "lora_b.kernel"]:
          raise ValueError(f"Unexpected non-lora specific weights ({module_name}.{name}) found in the lora_params")
        lora_b = lora_params["lora_a.kernel"]
        lora_a = lora_params["lora_b.kernel"]
        base_params["kernel"] = lora_update_or_base(base_params["kernel"], lora_a, lora_b)
        break

  unapply_lora_recursively(base_params, lora_params, "")


def load_adapter(config, base_abstract_state_params, adapter_config_path, adapter_weights_path):
  """Load an adapter configuration and its NNX-shaped parameters."""
  lora_params = None
  lora_config = None
  if adapter_config_path:
    if adapter_config_path.startswith("gs://"):
      lora_config = gcs_utils.read_json_from_gcs(adapter_config_path)
      commit_success = gcs_utils.gcs_path_exists(f"{adapter_weights_path}/commit_success.txt")
    else:
      with open(adapter_config_path, "rt", encoding="utf8") as config_file:
        lora_config = json.load(config_file)
      commit_success = os.path.exists(f"{adapter_weights_path}/commit_success.txt")
    if lora_config is None:
      raise FileNotFoundError(f"Failed to read lora_config from {adapter_config_path}.")
    if not commit_success:
      raise FileNotFoundError(f"Failed to read lora_weights from {adapter_weights_path}.")
    lora_state, _ = get_lora_abstract_state_nnx(base_abstract_state_params, lora_config)
    with logical_axis_rules(config.logical_axis_rules):
      lora_params = checkpointing.load_params_from_path(
          adapter_weights_path, lora_state.params, config.checkpoint_storage_concurrent_gb,
          config.checkpoint_storage_use_ocdbt, config.checkpoint_storage_use_zarr3,
      )
  return lora_params, lora_config


def setup_initial_lora_state(data_iterator, tx, config, mesh, checkpoint_manager, lora_adapter_path):
  """Initialize an NNX LoRA train state and optionally restore its parameters."""
  lora_state = None
  lora_state_annotations = None
  lora_config = None
  if lora_adapter_path:
    max_logging.log(f"Setting initial state of LoRA with lora_adapter_path = {lora_adapter_path}")
    create_model_partial, _ = nnx_abstract_model.create_nnx_abstract_model(config, mesh)

    def create_train_state_fn():
      nnx_model = create_model_partial()
      wrt = (
          getattr(nnx, "LoRAParam", nnx.Param)
          if getattr(getattr(config, "lora", None), "enable_lora", False)
          else nnx.Param
      )
      optimizer = nnx.Optimizer(nnx_model, tx, wrt=wrt)
      return train_state_nnx.TrainStateNNX(nnx_model, optimizer)

    unboxed_abstract_state, _, _ = state_abstraction.get_abstract_state(config, mesh, create_train_state_fn, True)
    lora_config_path = lora_adapter_path + "adapter_config.json"
    lora_config = gcs_utils.read_json_from_gcs(lora_config_path)
    base_abstract_params = _nnx_param_subtree(unboxed_abstract_state)
    lora_state, lora_state_annotations = get_lora_abstract_state_nnx(base_abstract_params, lora_config)
    lora_weights_path = f"{lora_adapter_path}/0/items"
    with logical_axis_rules(config.logical_axis_rules):
      restored_lora, raw_lora_params = checkpointing.load_state_if_possible(
          checkpoint_manager, data_iterator, lora_weights_path, config.load_full_state_path,
          config.checkpoint_storage_concurrent_gb, lora_state, config.enable_single_replica_ckpt_restoring,
          config.dataset_type, use_ocdbt=config.checkpoint_storage_use_ocdbt,
          use_zarr3=config.checkpoint_storage_use_zarr3,
      )
      if restored_lora:
        raise NotImplementedError("This codepath is not implemented for LoRA adapters yet.")
      lora_state = lora_state.replace(params=raw_lora_params)
      lora_state = partitioning.unbox_logicallypartioned(lora_state)
  return lora_config, lora_state, lora_state_annotations


def get_lora_abstract_state(base_abstract_params, lora_config):
  """Build the Linen abstract LoRA state and mesh annotations."""
  other_lora_format_to_jax_format = {
      "q_proj": "self_attention.query", "k_proj": "self_attention.key",
      "v_proj": "self_attention.value", "o_proj": "self_attention.out",
  }
  lora_target_modules = [other_lora_format_to_jax_format.get(s, s) for s in lora_config["target_modules"]]
  lora_rank = int(lora_config["r"])
  lora_abstract_params = {}

  def get_lora_param_shape(base_array_shape, lora_rank, lora_module):
    base_array_dimensions = len(base_array_shape)
    if base_array_dimensions > 4:
      raise ValueError(
          f"Encountered unexpected shape={base_array_shape} of array in base params. Array dimensions > 4 not supported."
      )
    if lora_module in ["self_attention.query", "self_attention.key", "self_attention.value"]:
      lora_a_shape = base_array_shape[:-2] + (lora_rank,)
      lora_b_shape = (lora_rank,) + base_array_shape[1:]
    elif lora_module in ["self_attention.out"]:
      lora_a_shape = base_array_shape[:-1] + (lora_rank,)
      if base_array_dimensions == 4:
        lora_b_shape = (lora_rank, base_array_shape[1], base_array_shape[-1])
      else:
        lora_b_shape = (lora_rank, base_array_shape[-1])
    else:
      raise ValueError(f"Unsupported lora_module={lora_module}")
    return lora_a_shape, lora_b_shape

  def get_lora_param_sharding(base_param_sharding, lora_module):
    if base_param_sharding is None:
      return None, None
    base_sharding_pspec_size = len(base_param_sharding.spec)
    if base_sharding_pspec_size > 4:
      raise ValueError("Encountered unexpected size of PartitionSpec in sharding. Size > 4 is not supported")
    base_mesh = base_param_sharding.mesh
    base_memory_kind = base_param_sharding.memory_kind
    base_pspec = base_param_sharding.spec
    if lora_module in ["self_attention.query", "self_attention.key", "self_attention.value"]:
      lora_a_pspec = jax.sharding.PartitionSpec(*(base_pspec[:-2] + ((),)))
      lora_b_pspec = jax.sharding.PartitionSpec(*(((),) + base_pspec[1:]))
    elif lora_module in ["self_attention.out"]:
      lora_a_pspec = jax.sharding.PartitionSpec(*(base_pspec[:-1] + ((),)))
      if base_sharding_pspec_size == 4:
        lora_b_pspec = jax.sharding.PartitionSpec((), base_pspec[1], base_pspec[-1])
      else:
        lora_b_pspec = jax.sharding.PartitionSpec((), base_pspec[-1])
    else:
      raise ValueError(f"Unsupported lora_module={lora_module}")
    lora_a_sharding = jax.sharding.NamedSharding(mesh=base_mesh, spec=lora_a_pspec, memory_kind=base_memory_kind)
    lora_b_sharding = jax.sharding.NamedSharding(mesh=base_mesh, spec=lora_b_pspec, memory_kind=base_memory_kind)
    return lora_a_sharding, lora_b_sharding

  def module_is_target_module(module, target_modules):
    for target_module in target_modules:
      if target_module in module:
        return target_module
    return None

  def add_lora_params(lora_params, module_name, base_params, lora_rank, lora_target_modules):
    for name, param in base_params.items():
      if isinstance(param, dict):
        lora_params[name] = {}
        add_lora_params(lora_params[name], f"{module_name}.{name}", param, lora_rank, lora_target_modules)
      else:
        if name not in ["kernel", "scale", "embedding"]:
          raise ValueError(f"Unexpected key={name} exists in the abstract params of base model.")
        if not isinstance(param, jax.ShapeDtypeStruct):
          raise ValueError("Unexpected type found in the abstract params of the base model.")
        target_module = module_is_target_module(module_name, lora_target_modules)
        if target_module is not None:
          lora_a_shape, lora_b_shape = get_lora_param_shape(param.shape, lora_rank, target_module)
          lora_a_sharding, lora_b_sharding = get_lora_param_sharding(param.sharding, target_module)
          lora_params["lora_a.kernel"] = jax.ShapeDtypeStruct(
              shape=lora_a_shape, dtype=param.dtype, sharding=lora_a_sharding
          )
          lora_params["lora_b.kernel"] = jax.ShapeDtypeStruct(
              shape=lora_b_shape, dtype=param.dtype, sharding=lora_b_sharding
          )
        else:
          lora_params[name] = None

  add_lora_params(lora_abstract_params, "", base_abstract_params, lora_rank, lora_target_modules)
  unboxed_abstract_lora_state = train_state.TrainState(
      step=0, apply_fn=None, params=lora_abstract_params, tx=None, opt_state={}
  )
  lora_state_mesh_annotations = train_state.TrainState(
      step=0, apply_fn=None, params=jax.tree_util.tree_map(lambda x: x.sharding.spec, lora_abstract_params),
      tx=None, opt_state={},
  )
  return unboxed_abstract_lora_state, lora_state_mesh_annotations


@lru_cache(maxsize=1)
def _load_lora_module_configs() -> dict:
  """Load the default LoRA module path mapping."""
  config_path = os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "lora_module_path.yml")
  return pyconfig._load_config(config_path)


def _get_lora_module_path(mt_config: pyconfig.HyperParameters) -> str:
  """Get the configured or architecture-specific LoRA module regex."""
  if mt_config.lora.lora_module_path:
    return mt_config.lora.lora_module_path
  lora_configs = _load_lora_module_configs()
  model_name = mt_config.model_name.lower()
  matched_key = next((key for key in lora_configs if key != "default" and model_name.startswith(key)), "default")
  if matched_key == "default":
    max_logging.log(f"Warning: Model '{model_name}' is unverified; falling back to default LoRA path.")
  else:
    max_logging.log(f"Auto-detected lora_module_path for model '{model_name}' (matched: '{matched_key}')")
  raw_path = lora_configs.get(matched_key, "decoder/layers/.*(self_attention/(query|key|value|out)|mlp/(wi_0|wi_1|wo))")
  layer_pattern = r"layers(?:_[0-9]+|/[0-9]+)?/"
  final_path = str(raw_path).replace("layers/", layer_pattern)
  max_logging.log(f"Using lora_module_path: {final_path}")
  return final_path


def _build_lora_provider(mt_config: pyconfig.HyperParameters) -> qwix.LoraProvider:
  """Build a Qwix LoRA provider from the native configuration."""
  lora_module_path = _get_lora_module_path(mt_config)
  lora_kwargs = {
      "module_path": lora_module_path,
      "rank": mt_config.lora.lora_rank,
      "alpha": mt_config.lora.lora_alpha,
      "dropout": 0.0,
      "weight_qtype": mt_config.lora.lora_weight_qtype,
      "tile_size": mt_config.lora.lora_tile_size,
  }
  lora_type = "QLoRA" if mt_config.lora.lora_weight_qtype else "LoRA"
  max_logging.log(
      f"{lora_type} configured: rank={mt_config.lora.lora_rank} alpha={mt_config.lora.lora_alpha} "
      f"qtype={mt_config.lora.lora_weight_qtype} tile_size={mt_config.lora.lora_tile_size}"
  )
  max_logging.log(f"Using lora_module_path: {lora_module_path}")
  return qwix.LoraProvider(**lora_kwargs)


def _prepare_dummy_inputs(
    mesh: Optional[jax.sharding.Mesh] = None, dummy_bs: int = 1, seq_len: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Build dummy decoder inputs sized for the mesh."""
  if mesh is not None:
    for axis in ("data", "fsdp", "fsdp_transpose", "expert"):
      dummy_bs *= mesh.shape.get(axis, 1)
    for axis in ("tensor_sequence", "context"):
      seq_len *= mesh.shape.get(axis, 1)
  decoder_input_tokens = jnp.zeros((dummy_bs, seq_len), dtype=jnp.int32)
  decoder_positions = jnp.zeros((dummy_bs, seq_len), dtype=jnp.int32)
  return decoder_input_tokens, decoder_positions


def is_lora_enabled(model: nnx.Module) -> bool:
  """Check whether the model contains LoRA parameters."""
  for _, value in nnx.iter_graph(model):
    if isinstance(value, nnx.LoRAParam):
      return True
  return False


def _verify_lora_parameters(
    lora_model: nnx.Module, mt_config: pyconfig.HyperParameters, matched_modules: Optional[set[str]] = None,
) -> None:
  """Validate that LoRA parameters or target modules were matched."""
  if matched_modules is None:
    matched_modules = {
        "/".join(str(part) for part in path[:-1])
        for path, value in nnx.iter_graph(lora_model)
        if isinstance(value, nnx.LoRAParam) and len(path) > 1
    }
  if matched_modules or is_lora_enabled(lora_model):
    wrapped_modules = sorted(list(matched_modules))
    if wrapped_modules:
      max_logging.log(
          f"LoRA configured: module_path='{_get_lora_module_path(mt_config)}' successfully matched "
          f"{len(wrapped_modules)} target submodules."
      )
      preview_limit = 20
      preview_modules = wrapped_modules[:preview_limit]
      max_logging.log(f"Sample matched submodules ({len(preview_modules)} of {len(wrapped_modules)}): {preview_modules}")
    else:
      max_logging.log("LoRA is enabled. (Detailed submodules match report skipped due to mock model or empty state)")
    return
  lora_module_path = _get_lora_module_path(mt_config)
  compiled_module_path = re.compile(lora_module_path)
  matched_module_paths = [
      "/".join(str(part) for part in path)
      for path, _ in nnx.iter_modules(lora_model)
      if path and compiled_module_path.search("/".join(str(part) for part in path))
  ]
  if not matched_module_paths:
    max_logging.log(f"Error: LoRA module_path='{lora_module_path}' did not match any weights.")
    raise ValueError("LoRA enabled but no LoRA parameters found in decoder/model state.")
  simplified_matches = sorted(
      {"/".join("*" if part.isdigit() else part for part in path.split("/")) for path in matched_module_paths}
  )
  max_logging.log(f"LoRA target verification: successfully matched {len(matched_module_paths)} modules.")
  max_logging.log(f"Matched submodule patterns: {simplified_matches}")
  raise ValueError(
      "LoRA module path matched target modules, but nnx.LoRAParam is still "
      "missing. For Tunix PeftTrainer, LoRA params must be materialized before "
      "trainer initialization, otherwise it falls back to full-model training. "
      f"Sample matches: {matched_module_paths[:10]}"
  )


def sync_lora_metadata(config: pyconfig.HyperParameters) -> None:
  """Synchronize adapter rank and alpha from checkpoint metadata."""
  lora_restore_path = config.lora.lora_restore_path
  if not lora_restore_path:
    return
  custom_metadata = checkpointing.load_checkpoint_metadata(lora_restore_path)
  lora_meta = custom_metadata.get("lora")
  if lora_meta:
    meta_rank = lora_meta.get("lora_rank", config.lora.lora_rank)
    meta_alpha = lora_meta.get("lora_alpha", config.lora.lora_alpha)
    if config.lora.lora_rank not in (0, meta_rank):
      raise ValueError(
          f"Configured lora_rank ({config.lora.lora_rank}) does not match "
          f"checkpoint metadata lora_rank ({meta_rank}) at {lora_restore_path}."
      )
    if config.lora.lora_alpha not in (0.0, meta_alpha):
      raise ValueError(
          f"Configured lora_alpha ({config.lora.lora_alpha}) does not match "
          f"checkpoint metadata lora_alpha ({meta_alpha}) at {lora_restore_path}."
      )
    config.lora.lora_rank = meta_rank
    config.lora.lora_alpha = meta_alpha
    max_logging.log(
        f"Synced LoRA parameters from Orbax metadata at {lora_restore_path}: "
        f"rank={config.lora.lora_rank}, alpha={config.lora.lora_alpha}"
    )


def apply_lora_to_model(
    model: nnx.Module, mesh: Optional[jax.sharding.Mesh], mt_config: pyconfig.HyperParameters,
) -> nnx.Module:
  """Apply Qwix LoRA or QLoRA when enabled in the configuration."""
  if mt_config.lora_input_adapters_path:
    max_logging.log("MaxText LoRA adapters loaded, skipping Qwix LoRA application")
    return model
  if not mt_config.lora.enable_lora:
    return model
  lora_provider = _build_lora_provider(mt_config)
  decoder = getattr(model, "decoder", model)
  model_rngs = getattr(decoder, "rngs", None)
  decoder_input_tokens, decoder_positions = _prepare_dummy_inputs(mesh)
  lora_model = qwix.apply_lora_to_model(
      model, lora_provider, decoder_input_tokens=decoder_input_tokens,
      decoder_positions=decoder_positions, rngs=model_rngs,
  )
  replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()) if mesh is not None else None
  matched_modules = set()
  for path, val in nnx.iter_graph(lora_model):
    if hasattr(val, "__dict__") and "qwix_rngs" in val.__dict__:
      del val.qwix_rngs
    if isinstance(val, nnx.LoRAParam):
      if len(path) > 1:
        matched_modules.add("/".join(str(part) for part in path[:-1]))
      val.set_metadata(sharding=jax.sharding.PartitionSpec(), out_sharding=None, sharding_names=None)
      if replicated_sharding is not None:
        arr = val.get_value()
        if isinstance(arr, jax.Array) and arr.sharding != replicated_sharding:
          arr_jnp = jnp.asarray(arr)
          resharded_arr = jax.make_array_from_callback(arr.shape, replicated_sharding, lambda idx, v=arr_jnp: v[idx])
          val.set_value(resharded_arr)
  _verify_lora_parameters(lora_model, mt_config, matched_modules)
  return lora_model


def apply_converted_lora_adapter(model: nnx.Module, adapter: Mapping) -> nnx.Module:
  """Apply the direct tree returned by convert_lora_to_maxtext_adapter to an NNX model."""
  lora_state = nnx.state(model, nnx.LoRAParam)
  variables = {tuple(str(part) for part in path): variable for path, variable in nnx.to_flat_state(lora_state)}
  pending = []

  def collect(tree, path):
    if isinstance(tree, Mapping) and set(tree) == {"value"}:
      variable = variables.get(path)
      if variable is None:
        raise ValueError(f"Converted LoRA parameter has no model match: {'/'.join(path)}")
      value = tree["value"]
      target = variable.get_value()
      if value.shape != target.shape:
        raise ValueError(f"Converted LoRA parameter {'/'.join(path)} has shape {value.shape}, expected {target.shape}")
      sharding = getattr(target, "sharding", None)
      pending.append((variable, jax.device_put(jnp.asarray(value, dtype=target.dtype), sharding)))
      return
    if not isinstance(tree, Mapping) or not tree:
      raise ValueError(f"Invalid converted LoRA tree at {'/'.join(path)}")
    for key, subtree in tree.items():
      collect(subtree, path + (str(key),))

  collect(adapter, ())
  for variable, value in pending:
    variable.set_value(value)
  nnx.update(model, lora_state)
  return model


def restore_lora_from_path(model: nnx.Module, mt_config: pyconfig.HyperParameters) -> nnx.Module:
  """Restore LoRA weights from an Orbax checkpoint into the model."""
  lora_restore_path = mt_config.lora.lora_restore_path
  if not lora_restore_path:
    return model
  if not is_lora_enabled(model):
    lora_module_path = _get_lora_module_path(mt_config)
    if not mt_config.lora.enable_lora:
      raise ValueError(
          "lora_restore_path is set but LoRA is not enabled on the model. "
          f"Set lora.enable_lora=True and verify lora_module_path ('{lora_module_path}') matches model modules."
      )
  sync_lora_metadata(mt_config)
  mesh = getattr(model, "mesh", None)
  if mesh is None:
    try:
      mesh = device_mesh.get_mesh_from_config(mt_config)
    except Exception:
      mesh = None
  abstract_lora_params = nnx.state(model, nnx.LoRAParam)

  def _build_target_leaf(value):
    val = value.get_value() if hasattr(value, "get_value") else getattr(value, "value", value)
    pspec = getattr(value, "sharding", None)
    if not isinstance(pspec, jax.sharding.PartitionSpec):
      pspec = jax.sharding.PartitionSpec()
    if mesh is not None:
      sharding = jax.sharding.NamedSharding(mesh, pspec)
    else:
      sharding = getattr(val, "sharding", None)
    if hasattr(val, "shape") and hasattr(val, "dtype"):
      return {"value": jax.ShapeDtypeStruct(shape=val.shape, dtype=val.dtype, sharding=sharding)}
    return {"value": val}

  target_for_restore = jax.tree.map(
      _build_target_leaf, abstract_lora_params, is_leaf=lambda node: isinstance(node, nnx.Variable),
  )
  sharding_tree = jax.tree.map(lambda value: getattr(value, "sharding", None), target_for_restore)
  restore_args_tree = ocp.checkpoint_utils.construct_restore_args(target_for_restore, sharding_tree)
  try:
    restore_args = ocp.args.PyTreeRestore(item=target_for_restore, restore_args=restore_args_tree, partial_restore=True)
    restored_lora_params = ocp.PyTreeCheckpointer().restore(lora_restore_path, args=restore_args)
  except Exception as error:
    max_logging.log(f"Guided restore failed: {error}. Falling back to basic restore.")
    restored_lora_params = ocp.PyTreeCheckpointer().restore(lora_restore_path)

  def _map_to_state(path, variable):
    if not isinstance(variable, nnx.Variable):
      return
    str_path = [str(key.key if hasattr(key, "key") else (key.name if hasattr(key, "name") else key)) for key in path]
    curr = restored_lora_params
    for part in str_path:
      if isinstance(curr, dict) and part in curr:
        curr = curr[part]
      elif hasattr(curr, part):
        curr = getattr(curr, part)
      else:
        return
    if isinstance(curr, dict) and "value" in curr:
      matched_val = curr["value"]
    elif hasattr(curr, "get_value"):
      matched_val = curr.get_value()
    elif hasattr(curr, "value"):
      matched_val = getattr(curr, "value")
    else:
      matched_val = curr
    if isinstance(matched_val, jax.ShapeDtypeStruct):
      raise ValueError(f"Parameter at path {'/'.join(str_path)} was not restored from checkpoint.")
    target_sharding = getattr(variable, "sharding", None)
    if not isinstance(target_sharding, jax.sharding.Sharding):
      try:
        mesh = getattr(model, "mesh", None) or device_mesh.get_mesh_from_config(mt_config)
        if mesh:
          if isinstance(target_sharding, jax.sharding.PartitionSpec):
            pspec = target_sharding
          else:
            pspec = jax.sharding.PartitionSpec()
          target_sharding = jax.sharding.NamedSharding(mesh, pspec)
        else:
          target_sharding = None
      except Exception:
        target_sharding = None
    if target_sharding is not None:
      try:
        matched_val = jax.device_put(matched_val, target_sharding)
      except Exception:
        pass
    if hasattr(variable, "set_value"):
      variable.set_value(matched_val)
    else:
      variable.value = matched_val

  jax.tree_util.tree_map_with_path(
      _map_to_state, abstract_lora_params, is_leaf=lambda node: isinstance(node, nnx.Variable),
  )
  nnx.pop(model, nnx.Intermediate)
  max_logging.log(f"LoRA restore complete from '{lora_restore_path}'.")
  return model


def _is_nnx_branch(value):
  """Check whether an NNX tree node is a branch."""
  return isinstance(value, Mapping)


def _nnx_param_subtree(unboxed_abstract_state):
  """Return the model substate of an abstract NNX train state."""
  return unboxed_abstract_state["model"] if "model" in unboxed_abstract_state else unboxed_abstract_state


def apply_lora_on_base_params_nnx(base_params, lora_params, lora_scale_factor=1.0):
  """Apply LoRA deltas to an NNX-shaped parameter tree in place."""

  def recurse(base_node, lora_node, path):
    if "lora_a.kernel" in lora_node:
      lora_a = lora_node["lora_a.kernel"]
      lora_b = lora_node["lora_b.kernel"]
      if lora_a is not None and lora_b is not None:
        delta = (jnp.einsum("er,rnd->end", lora_a, lora_b) * lora_scale_factor).astype(base_node["kernel"].dtype)
        base_node["kernel"] = (base_node["kernel"] + delta).astype(base_node["kernel"].dtype)
      return
    for name, lora_child in lora_node.items():
      if _is_nnx_branch(lora_child):
        recurse(base_node[name], lora_child, f"{path}.{name}")
      elif lora_child is not None:
        raise ValueError(f"Unexpected non-lora key ({path}.{name}) in lora_params")

  recurse(base_params, lora_params, "")


def unapply_lora_from_base_params_nnx(base_params, lora_params, lora_scale_factor=1.0):
  """Remove LoRA deltas from an NNX-shaped parameter tree in place."""

  def recurse(base_node, lora_node, path):
    if "lora_a.kernel" in lora_node:
      lora_a = lora_node["lora_a.kernel"]
      lora_b = lora_node["lora_b.kernel"]
      if lora_a is not None and lora_b is not None:
        delta = (jnp.einsum("er,rnd->end", lora_a, lora_b) * lora_scale_factor).astype(base_node["kernel"].dtype)
        base_node["kernel"] = (base_node["kernel"] - delta).astype(base_node["kernel"].dtype)
      return
    for name, lora_child in lora_node.items():
      if _is_nnx_branch(lora_child):
        recurse(base_node[name], lora_child, f"{path}.{name}")
      elif lora_child is not None:
        raise ValueError(f"Unexpected non-lora key ({path}.{name}) in lora_params")

  recurse(base_params, lora_params, "")


def get_lora_abstract_state_nnx(base_abstract_params, lora_config):
  """Build abstract LoRA parameters and mesh annotations for an NNX tree."""
  other_lora_format_to_jax_format = {
      "q_proj": "self_attention.query", "k_proj": "self_attention.key",
      "v_proj": "self_attention.value", "o_proj": "self_attention.out",
  }
  lora_target_modules = [other_lora_format_to_jax_format.get(s, s) for s in lora_config["target_modules"]]
  lora_rank = int(lora_config["r"])

  def get_lora_param_shape(base_array_shape, lora_module):
    if len(base_array_shape) > 4:
      raise ValueError(f"Unsupported base array shape {base_array_shape} (>4D)")
    if lora_module in ("self_attention.query", "self_attention.key", "self_attention.value"):
      lora_a_shape = base_array_shape[:-2] + (lora_rank,)
      lora_b_shape = (lora_rank,) + base_array_shape[1:]
    elif lora_module == "self_attention.out":
      lora_a_shape = base_array_shape[:-1] + (lora_rank,)
      if len(base_array_shape) == 4:
        lora_b_shape = (lora_rank, base_array_shape[1], base_array_shape[-1])
      else:
        lora_b_shape = (lora_rank, base_array_shape[-1])
    else:
      raise ValueError(f"Unsupported lora_module={lora_module}")
    return lora_a_shape, lora_b_shape

  def get_lora_param_sharding(base_param_sharding, lora_module):
    if base_param_sharding is None:
      return None, None
    base_pspec = base_param_sharding.spec
    if len(base_pspec) > 4:
      raise ValueError("PartitionSpec size > 4 not supported")
    if lora_module in ("self_attention.query", "self_attention.key", "self_attention.value"):
      lora_a_pspec = jax.sharding.PartitionSpec(*(base_pspec[:-2] + ((),)))
      lora_b_pspec = jax.sharding.PartitionSpec(*(((),) + base_pspec[1:]))
    elif lora_module == "self_attention.out":
      lora_a_pspec = jax.sharding.PartitionSpec(*(base_pspec[:-1] + ((),)))
      if len(base_pspec) == 4:
        lora_b_pspec = jax.sharding.PartitionSpec((), base_pspec[1], base_pspec[-1])
      else:
        lora_b_pspec = jax.sharding.PartitionSpec((), base_pspec[-1])
    else:
      raise ValueError(f"Unsupported lora_module={lora_module}")
    mesh = base_param_sharding.mesh
    mem_kind = base_param_sharding.memory_kind
    return (
        jax.sharding.NamedSharding(mesh=mesh, spec=lora_a_pspec, memory_kind=mem_kind),
        jax.sharding.NamedSharding(mesh=mesh, spec=lora_b_pspec, memory_kind=mem_kind),
    )

  def module_is_target(module_path):
    for target in lora_target_modules:
      if target in module_path:
        return target
    return None

  def add_lora(out_node, base_node, path):
    for name, child in base_node.items():
      if _is_nnx_branch(child):
        out_node[name] = {}
        add_lora(out_node[name], child, f"{path}.{name}")
      else:
        if name not in ("kernel", "scale", "embedding"):
          raise ValueError(f"Unexpected key={name} in base abstract params at {path}")
        if not isinstance(child, jax.ShapeDtypeStruct):
          raise ValueError(f"Unexpected leaf type {type(child).__name__} at {path}.{name}")
        target_module = module_is_target(path)
        if target_module is not None:
          a_shape, b_shape = get_lora_param_shape(child.shape, target_module)
          a_sharding, b_sharding = get_lora_param_sharding(child.sharding, target_module)
          out_node["lora_a.kernel"] = jax.ShapeDtypeStruct(shape=a_shape, dtype=child.dtype, sharding=a_sharding)
          out_node["lora_b.kernel"] = jax.ShapeDtypeStruct(shape=b_shape, dtype=child.dtype, sharding=b_sharding)
        else:
          out_node[name] = None

  lora_abstract_params = {}
  add_lora(lora_abstract_params, base_abstract_params, "")
  unboxed_abstract_lora_state = train_state.TrainState(
      step=0, apply_fn=None, params=lora_abstract_params, tx=None, opt_state={},
  )
  lora_state_mesh_annotations = train_state.TrainState(
      step=0, apply_fn=None,
      params=jax.tree_util.tree_map(lambda value: value.sharding.spec if value.sharding is not None else None,
                                    lora_abstract_params),
      tx=None, opt_state={},
  )
  return unboxed_abstract_lora_state, lora_state_mesh_annotations
