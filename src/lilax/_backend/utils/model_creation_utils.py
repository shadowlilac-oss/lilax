"""Utils that are only interesting for creating a model in MaxText."""

import collections
from collections.abc import Sequence
import contextlib
import dataclasses
from functools import partial
import math
from typing import Any, Callable, overload

from flax import nnx
from flax.core.meta import Partitioned
from flax.core.spmd import logical_axis_rules
from etils import epath
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
from orbax import checkpoint as ocp
from orbax.checkpoint.metadata import ArrayMetadata

from lilax._backend.checkpoint_conversion.cpu_bootstrap import convert_on_cpu
from lilax._backend.common.checkpoints import checkpoint_context, checkpointing
from lilax._backend.common import state_initialization
from lilax._backend.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_TRAIN
from lilax._backend.configs import pyconfig
from lilax._backend.experimental.lineage import lineage_adapter
from lilax._backend.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from lilax._backend.integration.vllm.convert_utils import _partition_size
from lilax._backend.layers import nnx_wrappers, quantizations
from lilax._backend.m3.core import checkpoint as m3_checkpoint
from lilax._backend.models import models
from lilax._backend.utils import device_mesh, max_logging, maxtext_utils_nnx, sharding, state_abstraction


def _is_orbax_array_metadata(node):
  return isinstance(node, ArrayMetadata)


def _is_array_meta_or_qtensor(node: Any) -> bool:
  """Returns whether a node is array metadata or a QTensor metadata tree."""
  if _is_orbax_array_metadata(node):
    return True
  if not isinstance(node, dict) or "qvalue" not in node:
    return False
  return all(_is_orbax_array_metadata(item) for item in jax.tree.leaves(node, is_leaf=_is_orbax_array_metadata))


def _checkpoint_uses_value_wrapper(meta_tree: Any, param_state: Any) -> bool:
  """Returns whether checkpoint metadata wraps NNX variable leaves under 'value'."""
  if not isinstance(meta_tree, dict) or not meta_tree:
    return False

  if param_state is not None and hasattr(param_state, "flat_state"):
    for path, var in param_state.flat_state():
      if not isinstance(var, nnx.Param):
        continue
      node = meta_tree
      resolved = True
      for key in path:
        if not isinstance(node, dict):
          resolved = False
          break
        if key in node:
          node = node[key]
        elif str(key) in node:
          node = node[str(key)]
        else:
          resolved = False
          break
      if not resolved:
        continue
      if _is_orbax_array_metadata(node):
        return False
      if isinstance(node, dict) and "value" in node:
        return True

  def _has_wrapped_value(node: Any) -> bool:
    if not isinstance(node, dict):
      return False
    if "value" in node and _is_array_meta_or_qtensor(node["value"]):
      return True
    return any(_has_wrapped_value(value) for value in node.values())

  return _has_wrapped_value(meta_tree)


_VLLM_REPEAT_AXES = frozenset({"kv_heads", ("expert", "model")})
_VLLM_ZERO_PAD_AXES = frozenset({"mlp_moe", "activation_mlp", ("attn_dp", "model")})


def _normalize_logical_axes(axes):
  """Convert logical partition axes to a tuple when available."""
  if axes is None:
    return None
  if isinstance(axes, jax.sharding.PartitionSpec):
    return tuple(axes)
  if isinstance(axes, (tuple, list)):
    return tuple(axes)
  return None


def _zero_pad_axis(arr, axis, extra):
  """Append zeros to an axis, padding local shards when possible."""
  if extra == 0:
    return arr

  arr_sharding = getattr(arr, "sharding", None)
  pad_width = [(0, 0)] * arr.ndim
  if isinstance(arr_sharding, jax.sharding.NamedSharding):
    spec = arr_sharding.spec
    partition = spec[axis] if axis < len(spec) else None
    shards_along_axis = _partition_size(partition, arr_sharding.mesh)
    if shards_along_axis > 1:
      if extra % shards_along_axis != 0:
        raise ValueError(
            f"Cannot per-shard zero-pad axis {axis}: extra={extra} not divisible by "
            f"shards_along_axis={shards_along_axis} (sharding spec entry {partition!r})."
        )
      pad_width[axis] = (0, extra // shards_along_axis)

      def _pad_local(value):
        return jnp.pad(value, pad_width)

      return jax.shard_map(
          _pad_local, mesh=arr_sharding.mesh, in_specs=spec, out_specs=spec, check_vma=False
      )(arr)

  pad_width[axis] = (0, extra)
  return jnp.pad(arr, pad_width)


def _align_checkpoint_to_model_shapes(ckpt_arr, model_arr, logical_axes=None):
  """Align a checkpoint array to the model shape and sharding."""
  ckpt_shape = ckpt_arr.shape
  model_shape = model_arr.shape
  if ckpt_shape == model_shape:
    return jax.device_put(ckpt_arr, model_arr.sharding)
  if len(ckpt_shape) != len(model_shape):
    raise ValueError(
        f"Checkpoint and model arrays have different ranks: {ckpt_shape} vs {model_shape}. "
        "If the checkpoint was saved with scan_layers=True (stacked layers), convert it to "
        "unscanned format before loading with vLLM (vllm.yml sets scan_layers=False)."
    )
  if len(ckpt_shape) == 2 and ckpt_shape == model_shape[::-1]:
    return jax.device_put(jnp.transpose(ckpt_arr), model_arr.sharding)

  axes = _normalize_logical_axes(logical_axes)
  if axes is None or len(axes) != len(model_shape):
    axes = (None,) * len(model_shape)

  result = ckpt_arr
  for axis, (ckpt_dim, model_dim, axis_name) in enumerate(zip(ckpt_shape, model_shape, axes)):
    if model_dim == ckpt_dim:
      continue
    if axis_name in _VLLM_ZERO_PAD_AXES:
      if model_dim < ckpt_dim:
        raise ValueError(
            f"axis {axis} (logical={axis_name!r}): model_dim={model_dim} smaller than "
            f"ckpt_dim={ckpt_dim}; shapes ckpt={ckpt_shape} model={model_shape}"
        )
      result = _zero_pad_axis(result, axis, model_dim - ckpt_dim)
    elif axis_name in _VLLM_REPEAT_AXES:
      if model_dim % ckpt_dim != 0:
        raise ValueError(
            f"axis {axis} (logical={axis_name!r}): model_dim={model_dim} not divisible by "
            f"ckpt_dim={ckpt_dim}; shapes ckpt={ckpt_shape} model={model_shape}"
        )
      result = jnp.repeat(result, model_dim // ckpt_dim, axis=axis)
    else:
      if model_dim % ckpt_dim != 0:
        raise ValueError(
            f"Cannot align axis {axis} (logical={axis_name!r}): model_dim={model_dim} not "
            f"divisible by ckpt_dim={ckpt_dim}, and axis is not registered in _VLLM_REPEAT_AXES "
            f"({sorted(str(x) for x in _VLLM_REPEAT_AXES)}) or "
            f"_VLLM_ZERO_PAD_AXES ({sorted(str(x) for x in _VLLM_ZERO_PAD_AXES)}). "
            f"Full shapes ckpt={ckpt_shape} model={model_shape}"
        )
      result = jnp.repeat(result, model_dim // ckpt_dim, axis=axis)
  return jax.device_put(result, model_arr.sharding)


def _intersect_with_metadata(target, meta_tree):
  """Returns the sub-tree of target whose paths exist in checkpoint metadata."""
  if not hasattr(target, "items"):
    return target
  out = {}
  for key, value in target.items():
    if hasattr(meta_tree, "items") and key in meta_tree:
      sub_tree = _intersect_with_metadata(value, meta_tree[key])
      if not hasattr(sub_tree, "items") or sub_tree:
        out[key] = sub_tree
  return out


def _fuse_moe_weights(ckpt_tree, model_arrays_tree):
  """Fuse and interleave separate MoE input projections for the target sharding."""
  def _is_fusion_site(node):
    return (
        isinstance(node, dict)
        and "wi_0" in node
        and "wi_1" in node
        and not isinstance(node["wi_0"], dict)
        and not isinstance(node["wi_1"], dict)
    )

  def _key_str(key):
    if hasattr(key, "key"):
      return key.key
    if hasattr(key, "attr"):
      return key.attr
    if hasattr(key, "name"):
      return key.name
    return key

  def _lookup_model(path):
    node = model_arrays_tree
    for key in path:
      name = _key_str(key)
      if isinstance(node, dict) and name in node:
        node = node[name]
      else:
        return None
    return node

  def _maybe_fuse(path, ckpt_node):
    if not _is_fusion_site(ckpt_node):
      return ckpt_node
    model_node = _lookup_model(path)
    if not isinstance(model_node, dict) or "wi" not in model_node:
      return ckpt_node

    wi_model = model_node["wi"]
    axis = wi_model.ndim - 1
    n_shards = 1
    wi_sharding = getattr(wi_model, "sharding", None)
    if isinstance(wi_sharding, jax.sharding.NamedSharding):
      spec = wi_sharding.spec
      partition = spec[axis] if axis < len(spec) else None
      n_shards = _partition_size(partition, wi_sharding.mesh)
    target_half_dim = wi_model.shape[-1] // 2

    def _pad_and_chunk(arr, target_total_size):
      shape = arr.shape
      chunk_size = shape[-1] // n_shards
      target_chunk_size = target_total_size // n_shards
      pad_amount = target_chunk_size - chunk_size
      arr_reshaped = arr.reshape(*shape[:-1], n_shards, chunk_size)
      if pad_amount > 0:
        pad_widths = [(0, 0)] * arr_reshaped.ndim
        pad_widths[-1] = (0, pad_amount)
        arr_reshaped = jnp.pad(arr_reshaped, pad_widths)
      return arr_reshaped

    padded_chunked_wi_0 = _pad_and_chunk(ckpt_node["wi_0"], target_half_dim)
    padded_chunked_wi_1 = _pad_and_chunk(ckpt_node["wi_1"], target_half_dim)
    wi_interleaved = jnp.concatenate([padded_chunked_wi_0, padded_chunked_wi_1], axis=-1)
    new_node = {key: value for key, value in ckpt_node.items() if key not in ("wi_0", "wi_1")}
    new_node["wi"] = wi_interleaved.reshape(*wi_model.shape)
    return new_node

  return jax.tree_util.tree_map_with_path(_maybe_fuse, ckpt_tree, is_leaf=_is_fusion_site)


def _stored_shape_evenly_shardable(restore_arg, stored_shape):
  """Return whether the stored shape is evenly partitioned by the restore sharding."""
  restore_sharding = restore_arg.sharding
  if not isinstance(restore_sharding, jax.sharding.NamedSharding):
    return False
  spec = restore_sharding.spec
  for axis_idx, dim in enumerate(stored_shape):
    partition = spec[axis_idx] if axis_idx < len(spec) else None
    if dim % _partition_size(partition, restore_sharding.mesh) != 0:
      return False
  return True


def _fix_restore_args_for_shape_mismatch(restore_args, stored_metadata_tree, mesh):
  """Adjust restore arguments for checkpoint arrays with different stored shapes."""
  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  def _key_str(key):
    if hasattr(key, "key"):
      return str(key.key)
    if hasattr(key, "attr"):
      return str(key.attr)
    if hasattr(key, "name"):
      return str(key.name)
    return str(key)

  def _lookup_stored_meta(path):
    node = stored_metadata_tree
    for key in path:
      if isinstance(key, jax.tree_util.SequenceKey):
        if isinstance(node, (list, tuple)) and 0 <= key.idx < len(node):
          node = node[key.idx]
          continue
        return None
      if not isinstance(node, dict):
        return None
      name = _key_str(key)
      if name in node:
        node = node[name]
        continue
      raw = str(key)
      if raw in node:
        node = node[raw]
        continue
      return None
    return node

  mismatched_paths_sharded = []
  mismatched_paths_replicated = []
  rank_mismatched_paths = []
  missing_paths = []
  found_array_count = [0]

  def _fix_one(path, restore_arg):
    if not isinstance(restore_arg, ocp.ArrayRestoreArgs):
      return restore_arg
    stored_meta = _lookup_stored_meta(path)
    if stored_meta is None:
      missing_paths.append(f"  {'.'.join(_key_str(key) for key in path)}")
      return restore_arg
    if _is_orbax_array_metadata(stored_meta):
      stored_shape = tuple(stored_meta.shape)
      if restore_arg.global_shape is not None and restore_arg.global_shape != stored_shape:
        if len(stored_shape) != len(restore_arg.global_shape):
          rank_mismatched_paths.append(
              f"  {'.'.join(_key_str(key) for key in path)}: "
              f"checkpoint shape {stored_shape} (rank {len(stored_shape)}) "
              f"vs model shape {restore_arg.global_shape} (rank {len(restore_arg.global_shape)})"
          )
        else:
          found_array_count[0] += 1
          path_str = f"  {'.'.join(_key_str(key) for key in path)}: stored={stored_shape} -> model={restore_arg.global_shape}"
          if _stored_shape_evenly_shardable(restore_arg, stored_shape):
            mismatched_paths_sharded.append(path_str)
            return dataclasses.replace(restore_arg, global_shape=stored_shape, shape=stored_shape)
          mismatched_paths_replicated.append(path_str)
          return dataclasses.replace(
              restore_arg, global_shape=None, shape=None, sharding=replicated, mesh=None, mesh_axes=None
          )
      else:
        found_array_count[0] += 1
    return restore_arg

  fixed = jax.tree_util.tree_map_with_path(_fix_one, restore_args, is_leaf=lambda x: isinstance(x, ocp.ArrayRestoreArgs))
  if rank_mismatched_paths:
    sample = "\n".join(rank_mismatched_paths[:5])
    more = f"\n  ... and {len(rank_mismatched_paths) - 5} more" if len(rank_mismatched_paths) > 5 else ""
    raise ValueError(f"Checkpoint rank mismatches detected ({len(rank_mismatched_paths)} arrays):\n{sample}{more}")

  total_arrays = found_array_count[0] + len(rank_mismatched_paths) + len(missing_paths)
  if total_arrays > 0 and len(missing_paths) / total_arrays > 0.8:
    sample = "\n".join(missing_paths[:5])
    more = f"\n  ... and {len(missing_paths) - 5} more" if len(missing_paths) > 5 else ""
    raise ValueError(
        f"Checkpoint structure mismatch: {len(missing_paths)} of {total_arrays}"
        " model parameter paths were not found in the checkpoint.\nExample"
        f" missing paths:\n{sample}{more}"
    )

  if mismatched_paths_sharded:
    max_logging.log(
        f"Checkpoint shape mismatches ({len(mismatched_paths_sharded)} arrays): loading sharded at "
        "stored shape and padding each local shard after restore.\n" + "\n".join(mismatched_paths_sharded)
    )
  if mismatched_paths_replicated:
    max_logging.log(
        f"Checkpoint shape mismatches ({len(mismatched_paths_replicated)} arrays): loading with replicated "
        "sharding (stored shape not evenly partitionable across mesh) and expanding after restore.\n"
        + "\n".join(mismatched_paths_replicated)
    )
  return fixed


@overload
def from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
    mesh: Mesh | None = None,
    *,
    model_mode: str = MODEL_MODE_TRAIN,
    rngs: None = None,
    quant_mode_str: str = "train",
) -> nnx_wrappers.ToLinen: ...


@overload
def from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
    mesh: Mesh | None = None,
    *,
    model_mode: str = MODEL_MODE_TRAIN,
    rngs: nnx.Rngs,
    quant_mode_str: str = "train",
) -> models.Transformer: ...


def from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
    mesh: Mesh | None = None,
    *,
    model_mode: str = MODEL_MODE_TRAIN,
    rngs: nnx.Rngs | None = None,
    quant_mode_str: str = "train",
) -> nnx_wrappers.ToLinen | models.Transformer:
  """Construct a model using the supplied configuration, mesh, and RNGs."""
  if mesh is None:
    mesh = device_mesh.get_mesh_from_config(config, devices)
  return create_model(config, mesh, model_mode=model_mode, rngs=rngs, quant_mode_str=quant_mode_str)


def get_transformer_model(config, mesh, quant, model_mode: str = MODEL_MODE_TRAIN, rngs: nnx.Rngs | None = None):
  """Returns the transformer model based on the configuration."""
  if rngs is not None and lineage_adapter.is_native(config):
    return lineage_adapter.LineageTransformer(config, mesh, quant=quant, rngs=rngs, model_mode=model_mode)
  if rngs is not None:
    return models.Transformer(config, mesh, quant=quant, rngs=rngs, model_mode=model_mode)
  return models.transformer_as_linen(config, mesh, quant=quant, model_mode=model_mode)


def create_model(
    config, mesh, model_mode: str = MODEL_MODE_TRAIN, rngs: nnx.Rngs | None = None, *, quant_mode_str: str = "train"
):
  """Instantiates and returns the model object, sharded across the mesh."""
  if config.use_m3_model:
    from lilax._backend.m3 import models as m3_models
    from lilax._backend.m3.infra import scan as m3_scan

    model = m3_models.create_model(config, mesh, rngs=rngs, quant_mode_str=quant_mode_str)
    return m3_scan.maybe_apply_scan(model, config)

  quant = quantizations.configure_quantization(config, quant_mode_str=quant_mode_str)
  model = get_transformer_model(config, mesh, quant, model_mode=model_mode, rngs=rngs)
  return quantizations.maybe_quantize_model(model, config)


def get_nnx_create_model_fn(
    config, mesh=None, devices=None, model_mode=MODEL_MODE_TRAIN, rng_key=None, *, quant_mode_str: str = "train"
) -> Callable:
  def _create_model():
    rngs = maxtext_utils_nnx.create_nnx_rngs(config, model_mode=model_mode, rng_key=rng_key)
    return from_config(config, devices, mesh, rngs=rngs, model_mode=model_mode, quant_mode_str=quant_mode_str)

  return _create_model


def _apply_m3_default_named_sharding(abs_var_state: Any, named_sharding_state: Any, mesh: Any) -> Any:
  """Assign 1D named sharding to unpartitioned M3 variables."""
  shard_axes = tuple(axis for axis in ("fsdp", "fsdp_transpose") if axis in mesh.axis_names and int(mesh.shape[axis]) > 1)
  if not shard_axes:
    fallback_axes = [
        (axis, int(mesh.shape[axis]))
        for axis in mesh.axis_names if axis not in ("data", "stage") and int(mesh.shape[axis]) > 1
    ]
    if not fallback_axes:
      max_logging.log(
          "WARNING: M3 default sharding found no fsdp/fsdp_transpose or"
          " non-data axis with size > 1; keeping weights replicated."
      )
      return named_sharding_state
    shard_axes = (max(fallback_axes, key=lambda item: item[1])[0],)
  axis_size = math.prod(int(mesh.shape[axis]) for axis in shard_axes)
  spec_axis = shard_axes[0] if len(shard_axes) == 1 else shard_axes

  def _shard_leaf(path, abstract_var, sharding_var):
    current_sharding = sharding_var.get_value() if hasattr(sharding_var, "get_value") else sharding_var
    is_unsharded = (
        not hasattr(current_sharding, "spec")
        or current_sharding.spec == jax.sharding.PartitionSpec()
        or all(axis is None for axis in current_sharding.spec)
    )
    if not is_unsharded:
      return sharding_var
    inner_abstract = abstract_var.get_value() if hasattr(abstract_var, "get_value") else abstract_var
    if not hasattr(inner_abstract, "shape"):
      return sharding_var
    shape = inner_abstract.shape
    path_strs = tuple(str(getattr(key, "key", getattr(key, "attr", key))) for key in path)
    start_dim = 1 if "cycles" in path_strs else 0
    for dim in range(start_dim, len(shape)):
      if shape[dim] % axis_size == 0:
        new_spec = jax.sharding.PartitionSpec(*(spec_axis if index == dim else None for index in range(len(shape))))
        new_sharding = jax.sharding.NamedSharding(mesh, new_spec)
        return sharding_var.replace(new_sharding) if hasattr(sharding_var, "replace") else new_sharding
    return sharding_var

  return jax.tree_util.tree_map_with_path(
      _shard_leaf, abs_var_state, named_sharding_state, is_leaf=lambda node: isinstance(node, nnx.Variable)
  )


def create_nnx_abstract_model(
    config, mesh=None, devices=None, model_mode=MODEL_MODE_TRAIN, rng_key=None, *, quant_mode_str: str = "train"
) -> tuple[Callable, nnx.Module]:
  """Create a model factory and an abstract, sharded NNX model."""
  with logical_axis_rules(config.logical_axis_rules):
    create_model_fn = get_nnx_create_model_fn(config, mesh, devices, model_mode, rng_key, quant_mode_str=quant_mode_str)
    abs_model = nnx.eval_shape(create_model_fn)
    if mesh is None:
      mesh = abs_model.mesh
    graphdef, abs_var_state = nnx.split(abs_model)
    named_sharding_state = sharding.nnx_construct_named_sharding(abs_var_state, mesh)
    if getattr(config, "use_m3_model", False) and mesh is not None and getattr(mesh, "size", 1) > 1:
      named_sharding_state = _apply_m3_default_named_sharding(abs_var_state, named_sharding_state, mesh)
    abstract_state = jax.tree.map(
        lambda abstract, named: jax.ShapeDtypeStruct(abstract.shape, abstract.dtype, sharding=named),
        abs_var_state,
        named_sharding_state,
    )
    return create_model_fn, nnx.merge(graphdef, abstract_state)


def setup_configs_and_devices(config, *, resolve_config, devices=None):
  """Set up device allocation and configs for training and inference."""
  return setup_configs_and_devices_from_config(config, resolve_config=resolve_config, devices=devices)


def setup_configs_and_devices_from_config(config, *, resolve_config, devices=None):
  """Set up device allocation and configs using independently resolved overrides."""
  if devices is None:
    devices = jax.devices()
  if config.num_trainer_slices == -1 and config.num_samplers_slices == -1:
    max_logging.log("Running on a single slice")
    num_vms = len(devices) // config.chips_per_vm
    trainer_devices = devices
    sampler_devices = devices
    if num_vms >= 2 and config.use_pathways:
      max_logging.log(f"{num_vms} VMs detected, allocating trainer and sampler devices, and using Pathways.")
      num_devices = len(devices)
      num_trainer_devices = int(num_devices * config.trainer_devices_fraction)
      num_sampler_devices = int(num_devices * config.sampler_devices_fraction)
      trainer_devices = devices[:num_trainer_devices]
      sampler_devices = devices[num_devices - num_sampler_devices :]
      if config.trainer_devices_fraction != 1.0:
        max_logging.log(f"Using first {len(trainer_devices)} devices as Trainer devices")
      if config.sampler_devices_fraction != 1.0:
        max_logging.log(f"Using last {len(sampler_devices)} devices as Sampler devices")
    trainer_config = config
    sampler_config = config
  elif config.num_trainer_slices > 0 and config.num_samplers_slices > 0:
    max_logging.log("Running with Multislice")
    devices_by_slice = collections.defaultdict(list)
    for device in devices:
      devices_by_slice[device.slice_index].append(device)
    slice_indices = sorted(devices_by_slice.keys())
    if len(slice_indices) < config.num_trainer_slices + config.num_samplers_slices:
      raise ValueError("Not enough slices for trainer and samplers")

    trainer_devices = []
    for index in range(config.num_trainer_slices):
      trainer_devices.extend(devices_by_slice[slice_indices[index]])
    sampler_devices = []
    for index in range(config.num_trainer_slices, config.num_trainer_slices + config.num_samplers_slices):
      sampler_devices.extend(devices_by_slice[slice_indices[index]])

    trainer_devices_per_slice = len(trainer_devices) // config.num_trainer_slices
    trainer_fsdp = trainer_devices_per_slice
    tp = config.ici_tensor_parallelism
    if tp > 1:
      if trainer_devices_per_slice % tp != 0:
        raise ValueError(
            f"trainer_devices_per_slice ({trainer_devices_per_slice}) must be divisible by tensor parallelism ({tp})"
        )
      if config.ici_fsdp_parallelism != -1 and config.ici_fsdp_parallelism * tp != trainer_devices_per_slice:
        raise ValueError(
            f"ici_fsdp_parallelism ({config.ici_fsdp_parallelism}) * ici_tensor_parallelism ({tp}) must equal "
            f"devices_per_slice ({trainer_devices_per_slice})"
        )
      trainer_fsdp = trainer_devices_per_slice // tp

    trainer_kwargs = {
        "num_slices": config.num_trainer_slices,
        "ici_fsdp_parallelism": trainer_fsdp,
        "ici_tensor_parallelism": tp,
        "dcn_data_parallelism": config.num_trainer_slices,
    }
    sampler_kwargs = {
        "num_slices": config.num_samplers_slices,
        "ici_fsdp_parallelism": len(sampler_devices) // config.num_samplers_slices,
        "ici_tensor_parallelism": -1,
        "dcn_data_parallelism": config.num_samplers_slices,
    }
    trainer_config = resolve_config(trainer_kwargs)
    sampler_config = resolve_config(sampler_kwargs)
  else:
    raise ValueError("num_trainer_slices and num_samplers_slices should be both -1 or positive")

  return trainer_config, sampler_config, trainer_devices, sampler_devices


def get_rollout_kwargs_for_parallelism(sampler_config, num_sampler_devices):
  """Get rollout kwargs for vLLM rollout when using data parallelism."""
  dp = sampler_config.rollout_data_parallelism
  tp = sampler_config.rollout_tensor_parallelism
  ep = sampler_config.rollout_expert_parallelism
  num_auto = sum(1 for value in [tp, dp, ep] if value == -1)
  if num_auto > 1:
    raise ValueError(
        "At most one of rollout_tensor_parallelism, rollout_data_parallelism, "
        "rollout_expert_parallelism can be -1 (auto-derived).\n"
        f"Currently resolved values:\n"
        f"  - rollout_tensor_parallelism = {tp}\n"
        f"  - rollout_data_parallelism = {dp}\n"
        f"  - rollout_expert_parallelism = {ep}\n\n"
        "To fix this, you must explicitly define at least two of these parameters in your command line arguments.\n"
        "For example, try adding 'rollout_tensor_parallelism=4' to your command."
    )
  if dp == -1:
    if num_sampler_devices % (tp * ep) != 0:
      raise ValueError(
          f"num_sampler_devices({num_sampler_devices}) must be divisible by "
          f"rollout_tensor_parallelism({tp}) * rollout_expert_parallelism({ep}) "
          "when rollout_data_parallelism is -1."
      )
    dp = num_sampler_devices // tp // ep
  elif tp == -1:
    if num_sampler_devices % (dp * ep) != 0:
      raise ValueError(
          f"num_sampler_devices({num_sampler_devices}) must be divisible by "
          f"rollout_data_parallelism({dp}) * rollout_expert_parallelism({ep}) "
          "when rollout_tensor_parallelism is -1."
      )
    tp = num_sampler_devices // dp // ep
  elif ep == -1:
    if num_sampler_devices % (tp * dp) != 0:
      raise ValueError(
          f"num_sampler_devices({num_sampler_devices}) must be divisible by "
          f"rollout_tensor_parallelism({tp}) * rollout_data_parallelism({dp}) "
          "when rollout_expert_parallelism is -1."
      )
    ep = num_sampler_devices // tp // dp
  elif tp * dp * ep != num_sampler_devices:
    raise ValueError(
        f"rollout_tensor_parallelism({tp}) * "
        f"rollout_data_parallelism({dp}) * "
        f"rollout_expert_parallelism({ep}) "
        f"!= len(sampler_devices)({num_sampler_devices})"
    )
  rollout_kwargs = {}
  rollout_kwargs["tensor_parallel_size"] = tp
  rollout_kwargs["data_parallel_size"] = dp
  rollout_kwargs["expert_parallel_size"] = ep
  return rollout_kwargs


def create_models_and_meshes(trainer_config, sampler_config, trainer_devices, sampler_devices, tokenizer_pad_id=None):
  """Create reference and actor models and their respective meshes."""
  max_logging.log("Creating reference model and also meshes for reference and rollout")
  reference_model, reference_mesh = from_pretrained(
      trainer_config, devices=trainer_devices, wrap_with_tunix_adapter=True, tokenizer_pad_id=tokenizer_pad_id
  )
  devices_array = device_mesh.create_device_mesh(sampler_config, sampler_devices)
  rollout_mesh = Mesh(devices_array, sampler_config.mesh_axes)

  if trainer_config.load_checkpoint_only_once:
    max_logging.log("Creating policy model by copying reference model instead of restoring from checkpoint again.")
    with reference_mesh:
      actor_base_model = nnx.clone(reference_model.base)
      use_no_op_mappings = "maxtext_config" in trainer_config.vllm_additional_config
      actor_model = TunixMaxTextAdapter(
          base_model=actor_base_model, use_no_op_mappings=use_no_op_mappings, pad_id=tokenizer_pad_id
      )
      actor_model.config = None
    actor_mesh = reference_mesh
  else:
    max_logging.log("Creating policy model with same config as reference model on trainer mesh")
    actor_model, actor_mesh = from_pretrained(
        trainer_config, devices=trainer_devices, wrap_with_tunix_adapter=True, tokenizer_pad_id=tokenizer_pad_id
    )

  return reference_model, reference_mesh, actor_model, actor_mesh, rollout_mesh


def verify_and_sync_scan_layers(config):
  """Verify and sync scan_layers based on checkpoint metadata."""
  if not config.load_parameters_path:
    return config

  custom_metadata = checkpointing.load_checkpoint_metadata(config.load_parameters_path)
  saved_scan_layers = custom_metadata.get("scan_layers")
  if not isinstance(saved_scan_layers, bool):
    return config

  if saved_scan_layers == config.scan_layers:
    return config

  pydantic_config = getattr(config, "_pydantic_config", config)
  model_fields_set = getattr(pydantic_config, "model_fields_set", None)
  is_explicit = "scan_layers" in model_fields_set if model_fields_set is not None else True

  if is_explicit:
    if saved_scan_layers != config.scan_layers:
      raise ValueError(
          f"Configuration mismatch: Your run specifies scan_layers={config.scan_layers}, "
          f"but the checkpoint was saved with scan_layers={saved_scan_layers}."
      )
  else:
    max_logging.log(f"Setting scan_layers={saved_scan_layers} loaded from checkpoint metadata.")
    new_pydantic_config = pydantic_config.model_copy(update={"scan_layers": saved_scan_layers})
    if getattr(config, "_pydantic_config", None) is not None:
      config = pyconfig.HyperParameters(new_pydantic_config)
    else:
      config = new_pydantic_config

  return config


def from_pretrained(
    config,
    mesh=None,
    devices=None,
    model_mode=MODEL_MODE_TRAIN,
    rng_key=None,
    wrap_with_tunix_adapter=False,
    tokenizer_pad_id=None,
    *,
    quant_mode_str: str = "train",
    return_effective_config=False,
):
  """Create an NNX model with sharded parameters, optionally restoring a checkpoint."""
  original_mesh = mesh
  if config.convert_checkpoint_if_possible and not config.load_parameters_path:
    load_parameters_path = epath.Path(config.base_output_directory) / "0" / "items"
    if not load_parameters_path.exists():
      load_parameters_path = convert_on_cpu(config, None, config.base_output_directory)
    pydantic_config = getattr(config, "_pydantic_config", config)
    config = pyconfig.HyperParameters(pydantic_config.model_copy(update={"load_parameters_path": load_parameters_path}))

  config = verify_and_sync_scan_layers(config)
  create_model_fn, abstract_model = create_nnx_abstract_model(
      config, mesh, devices, model_mode, rng_key, quant_mode_str=quant_mode_str
  )
  _, abstract_state = nnx.split(abstract_model)
  specs = nnx.get_partition_spec(abstract_state)
  model = maxtext_utils_nnx.create_nnx_sharded_model(abstract_model, create_model_fn, mesh=mesh)
  sharded_state = nnx.state(model)
  if mesh is None:
    mesh = model.mesh

  with mesh:
    maxtext_lineage_checkpoint = False
    if config.load_parameters_path and lineage_adapter.is_native(config):
      path = epath.Path(checkpointing._normalize_checkpoint_root(config.load_parameters_path))
      checkpointable_name = "items" if (path / "items").exists() else None
      metadata_context = (
          checkpoint_context.build_context(colocated_python_checkpointing=True)
          if config.enable_single_controller and config.colocated_python_checkpointing else contextlib.nullcontext()
      )
      with metadata_context:
        metadata_tree = ocp.metadata(path, checkpointable_name=checkpointable_name).metadata
      pending = [metadata_tree]
      formats = set()
      while pending:
        tree = pending.pop()
        if not isinstance(tree, dict):
          continue
        if "token_embedder" in tree and "decoder" in tree:
          formats.add("maxtext")
        if "embed" in tree and "decoder" in tree and "head" in tree:
          formats.add("lineage")
        pending.extend(tree[key] for key in ("base", "model", "model_params", "params") if key in tree)
      if len(formats) != 1:
        raise ValueError(f"Cannot determine Lineage checkpoint layout from metadata at '{config.load_parameters_path}'.")
      maxtext_lineage_checkpoint = "maxtext" in formats

    if maxtext_lineage_checkpoint:
      raw_params = checkpointing.load_params_from_path(
          config.load_parameters_path, lineage_adapter.maxtext_restore_target(config, mesh),
          config.checkpoint_storage_concurrent_gb, use_ocdbt=config.checkpoint_storage_use_ocdbt,
          use_zarr3=config.checkpoint_storage_use_zarr3,
          enable_single_replica_ckpt_restoring=bool(config.enable_single_replica_ckpt_restoring),
          colocated_python_checkpointing=bool(config.enable_single_controller and config.colocated_python_checkpointing),
      )
      lineage_adapter.restore_params(sharded_state, raw_params, config, mesh)
      nnx.update(model, sharded_state)
    elif config.load_parameters_path:
      ckptr = ocp.Checkpointer(
          ocp.PyTreeCheckpointHandler(
              restore_concurrent_gb=config.checkpoint_storage_concurrent_gb,
              save_concurrent_gb=config.checkpoint_storage_concurrent_gb,
              use_ocdbt=config.checkpoint_storage_use_ocdbt,
              use_zarr3=config.checkpoint_storage_use_zarr3,
          )
      )
      metadata = ckptr.metadata(config.load_parameters_path)
      if metadata is None or metadata.item_metadata is None:
        raise ValueError(f"No valid Orbax checkpoint found at '{config.load_parameters_path}'. "
                         "Please check your load_parameters_path.")

      def adjust_target(target, meta_tree, is_nnx, use_value_wrapper=True):
        if not hasattr(target, "items") or not hasattr(meta_tree, "items"):
          return target
        new_target = {}
        for key, value in target.items():
          if key == "wi" and "wi" not in meta_tree and "wi_0" in meta_tree and "wi_1" in meta_tree:
            array = value["value"] if is_nnx and use_value_wrapper else value
            half_dim = array.shape[-1] // 2
            split = jax.ShapeDtypeStruct(shape=array.shape[:-1] + (half_dim,), dtype=array.dtype, sharding=array.sharding)
            new_target["wi_0"] = {"value": split} if is_nnx and use_value_wrapper else split
            new_target["wi_1"] = {"value": split} if is_nnx and use_value_wrapper else split
          else:
            new_target[key] = adjust_target(value, meta_tree.get(key, {}), is_nnx, use_value_wrapper)
        return new_target

      param_state = sharded_state.filter(
          lambda path, var: not isinstance(var, (nnx.RngState, nnx.Cache, nnx.Intermediate, nnx.BatchStat))
      )
      metadata_tree = metadata.item_metadata.tree
      is_nnx_checkpoint = not ("params" in metadata_tree and "params" in metadata_tree.get("params", {}))
      has_base_key = False
      use_value_wrapper = False
      if not is_nnx_checkpoint:
        target = jax.tree.map(lambda value: value[...], param_state, is_leaf=lambda node: isinstance(node, nnx.Variable))
        if config.use_m3_model:
          target = m3_checkpoint.match_checkpoint_layer_names(target, metadata_tree["params"]["params"])
        target = adjust_target(target, metadata_tree["params"]["params"], False)
        collections = metadata_tree["params"]
        item = {"params": {"params": target}}
        for collection, collection_metadata in collections.items():
          if collection != "params":
            collection_target = _intersect_with_metadata(target, collection_metadata)
            if collection_target:
              item["params"][collection] = collection_target
        restore_args = {"params": {
            collection: _fix_restore_args_for_shape_mismatch(
                ocp.checkpoint_utils.construct_restore_args(collection_target), collections[collection], mesh
            ) for collection, collection_target in item["params"].items()
        }}
      else:
        has_base_key = "base" in metadata_tree
        param_metadata = metadata_tree.get("base", metadata_tree)
        use_value_wrapper = _checkpoint_uses_value_wrapper(param_metadata, param_state)

        def build_target(value):
          inner = value.get_value() if hasattr(value, "get_value") else value[...]
          if hasattr(inner, "shape"):
            return {"value": value[...]} if use_value_wrapper else value[...]
          inner = jax.tree.map(
              lambda leaf: leaf.value if isinstance(leaf, Partitioned) else leaf,
              inner, is_leaf=lambda leaf: isinstance(leaf, Partitioned),
          )
          return {"value": inner} if use_value_wrapper else inner

        target = jax.tree.map(build_target, param_state, is_leaf=lambda node: isinstance(node, nnx.Variable))
        if config.use_m3_model:
          target = m3_checkpoint.match_checkpoint_layer_names(target, param_metadata)
        target = adjust_target(target, param_metadata, True, use_value_wrapper)
        item = {"base": target} if has_base_key else target
        restore_args = _fix_restore_args_for_shape_mismatch(
            ocp.checkpoint_utils.construct_restore_args(target), param_metadata, mesh
        )
        if has_base_key:
          restore_args = {"base": restore_args}

      def free_device_memory(path, node):
        is_custom = any("custom_linear" in str(getattr(part, "key", part)) for part in path)
        if (isinstance(node, nnx.Variable)
            and not isinstance(node, (nnx.RngState, nnx.Cache, nnx.Intermediate, nnx.BatchStat))
            and not is_custom):
          inner = node.get_value() if hasattr(node, "get_value") else node[...]
          for leaf in jax.tree_util.tree_leaves(inner):
            if isinstance(leaf, jax.Array) and not leaf.is_deleted():
              leaf.delete()
        elif isinstance(node, jax.Array) and not node.is_deleted() and not is_custom:
          node.delete()
        return node

      jax.tree_util.tree_map_with_path(free_device_memory, sharded_state, is_leaf=lambda node: isinstance(node, nnx.Variable))
      restored = ckptr.restore(epath.Path(config.load_parameters_path), item=item, transforms={}, restore_args=restore_args)
      if is_nnx_checkpoint:
        restored_root = restored["base"] if has_base_key else restored
        checkpoint = jax.tree.map(
            lambda value: value["value"], restored_root,
            is_leaf=lambda value: isinstance(value, dict) and "value" in value and not isinstance(value.get("value"), dict),
        ) if use_value_wrapper else restored_root
      else:
        checkpoint = restored["params"]["params"]
        for collection, collection_tree in restored["params"].items():
          if collection != "params":
            checkpoint = checkpointing._deep_merge_dicts(checkpoint, collection_tree)

      if not checkpoint:
        raise ValueError(
            f"Checkpoint restore from '{config.load_parameters_path}' yielded no parameters. "
            "This usually means the checkpoint format is incompatible with the model configuration "
            "(e.g. a scanned checkpoint loaded with scan_layers=False, or vice versa). "
            "Please ensure the checkpoint format matches the scan_layers setting."
        )
      if config.use_m3_model:
        checkpoint = m3_checkpoint.from_legacy_layer_names(checkpoint, sharded_state)
      model_arrays = jax.tree.map(
          lambda value: value.get_value() if hasattr(value, "get_value") else value[...],
          sharded_state, is_leaf=lambda node: isinstance(node, nnx.Variable),
      )
      logical_axes = jax.tree.map(
          lambda value: value.get_value(), specs, is_leaf=lambda node: isinstance(node, nnx.Variable)
      )

      def to_dict(tree):
        return {key: to_dict(value) for key, value in tree.items()} if hasattr(tree, "items") else tree

      model_arrays, checkpoint, logical_axes = to_dict(model_arrays), to_dict(checkpoint), to_dict(logical_axes)
      checkpoint = _fuse_moe_weights(checkpoint, model_arrays)
      del restored

      def filter_to_model_keys(ckpt, model_tree):
        if not hasattr(ckpt, "items") or not hasattr(model_tree, "items"):
          return ckpt
        return {key: filter_to_model_keys(ckpt[key], model_tree[key]) for key in model_tree if key in ckpt}

      def walk_align(ckpt, model_array, axes):
        if isinstance(ckpt, dict):
          return {key: walk_align(value, model_array[key], axes.get(key) if isinstance(axes, dict) else None)
                  for key, value in ckpt.items()}
        if not isinstance(ckpt, (jax.Array, jax.ShapeDtypeStruct, np.ndarray)):
          return ckpt
        return _align_checkpoint_to_model_shapes(ckpt, model_array, axes)

      checkpoint = walk_align(filter_to_model_keys(checkpoint, model_arrays), model_arrays, logical_axes)
      nnx.update(model, checkpoint)
      unrestored = [
          jax.tree_util.keystr(path) for path, leaf in jax.tree_util.tree_flatten_with_path(nnx.state(model))[0]
          if isinstance(leaf, jax.Array) and leaf.is_deleted()
      ]
      if unrestored:
        raise ValueError(
            f"Checkpoint '{config.load_parameters_path}' has no value for {len(unrestored)} model "
            f"variables: {unrestored[:5]}"
        )

    if wrap_with_tunix_adapter:
      with mesh:
        use_no_op_mappings = "maxtext_config" in config.vllm_additional_config
        model = TunixMaxTextAdapter(base_model=model, use_no_op_mappings=use_no_op_mappings, pad_id=tokenizer_pad_id)
        model.config = None
    result = model if original_mesh else (model, mesh)
    return (result, config) if return_effective_config else result


def setup_decode_state_from_nnx(model, config, rng, mesh):
  """Load an NNX checkpoint into a linen decode TrainState."""
  init_state_fn = partial(state_initialization.init_initial_state, model, None, config, False, rng)
  _, state_mesh_annotations, _ = state_abstraction.get_abstract_state(config, mesh, init_state_fn, False)
  nnx_model = from_pretrained(config, mesh=mesh, model_mode=MODEL_MODE_AUTOREGRESSIVE)

  def _state_to_dict(tree):
    if isinstance(tree, nnx.Variable):
      return tree.get_value()
    if hasattr(tree, "items") and not isinstance(tree, jax.Array):
      return {key: _state_to_dict(value) for key, value in tree.items()}
    return tree

  nnx_param_state = nnx.state(nnx_model, nnx.Param)
  raw_params = _state_to_dict(nnx_param_state)
  if config.use_m3_model:
    raw_params = m3_checkpoint.to_legacy_layer_names(raw_params)
  del nnx_model, nnx_param_state

  params = {"params": raw_params}
  state = state_initialization.init_decode_state(model.apply, params)
  return state, state_mesh_annotations
