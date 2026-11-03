"""Restore full training states from Linen and external checkpoint layouts."""

import importlib
from etils import epath
from flax import nnx
import jax
from lilax._backend.common import train_state_nnx
from lilax._backend.common.checkpoints import checkpoint_context
from lilax._backend.common.checkpoints.linen_compatibility import (
    _deep_merge_dicts, _expected_and_restored_params, _raise_on_weight_mismatch,
)
from lilax._backend.common.checkpoints.lora_filter import _filter_lora_trainable_state
from lilax._backend.common.checkpoints.restore_source_types import _normalize_checkpoint_root
from lilax._backend.trainers.diloco import checkpointing as diloco_checkpoint_utils
from lilax._backend.utils import max_logging
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint._src.arrays import sharding as sharding_utils


def _linen_items_to_nnx(restored_linen, abstract_nnx_state):
  linen_state, aux_state, ephemeral = train_state_nnx.split_for_checkpoint(abstract_nnx_state)
  weights = train_state_nnx.from_linen_checkpoint_dict(restored_linen)
  if "model" in weights:
    nnx.replace_by_pure_dict(linen_state, {"model": weights["model"]})
  if "optimizer" in weights:
    nnx.replace_by_pure_dict(linen_state, {"optimizer": weights["optimizer"]})
  nnx_aux = restored_linen.get("nnx_aux")
  if nnx_aux:
    train_state_nnx.apply_checkpoint_aux(aux_state, nnx_aux)
  elif isinstance(restored_linen.get("params"), dict):
    custom_collections = {key: value for key, value in restored_linen["params"].items() if key != "params" and isinstance(value, dict)}
    if custom_collections:
      merged_custom = {}
      for col_data in custom_collections.values():
        merged_custom = _deep_merge_dicts(merged_custom, col_data)
      train_state_nnx.apply_checkpoint_aux(aux_state, {"model": merged_custom})
  return nnx.merge_state(linen_state, aux_state, ephemeral)


def _load_linen_checkpoint_into_nnx(
    path, abstract_nnx_state, checkpoint_storage_concurrent_gb, use_ocdbt, use_zarr3,
    enable_single_replica_ckpt_restoring: bool = False, config=None,
):
  max_logging.log(f"Restoring Linen-layout checkpoint into NNX state at {path}")
  if config and getattr(config, "enable_diloco", False):
    return diloco_checkpoint_utils.restore_diloco_checkpoint(
        path, abstract_nnx_state, checkpoint_storage_concurrent_gb, use_ocdbt=use_ocdbt, use_zarr3=use_zarr3, config=config
    )
  linen_abstract = train_state_nnx.to_checkpoint_dict(abstract_nnx_state)
  if config and getattr(getattr(config, "lora", None), "enable_lora", False):
    linen_abstract = _filter_lora_trainable_state(linen_abstract, preserve_optimizer=True)
  context = checkpoint_context.build_context(
      use_ocdbt=use_ocdbt, use_zarr3=use_zarr3, checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
      partial_load=True, enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
      colocated_python_checkpointing=bool(getattr(config, "colocated_python_checkpointing", False)),
  )
  root = epath.Path(_normalize_checkpoint_root(path))
  with context:
    checkpointable_name = "items" if (root / "items").exists() else None
    restored = ocp.load(root, linen_abstract, checkpointable_name=checkpointable_name)
  return restored_linen_to_nnx(restored, abstract_nnx_state, config=config)


def restored_linen_to_nnx(restored_linen, abstract_nnx_state, config=None):
  _raise_on_weight_mismatch(*_expected_and_restored_params(abstract_nnx_state, restored_linen), config=config)
  return _linen_items_to_nnx(restored_linen, abstract_nnx_state)


def _resolve_conversion_fn(checkpoint_conversion_fn):
  if checkpoint_conversion_fn is None:
    raise ValueError(
        "source_checkpoint_layout='safetensors' needs `checkpoint_conversion_fn` to map the "
        "checkpoint's weights onto the model's, e.g. checkpoint_conversion_fn=my_pkg.my_module.my_fn."
    )
  if callable(checkpoint_conversion_fn):
    return checkpoint_conversion_fn
  module_name, _, fn_name = str(checkpoint_conversion_fn).rpartition(".")
  if not module_name:
    raise ValueError(f"`checkpoint_conversion_fn` must be a dotted path to a function, got {checkpoint_conversion_fn!r}.")
  try:
    fn = getattr(importlib.import_module(module_name), fn_name, None)
  except ImportError as error:
    raise ValueError(f"Could not import `checkpoint_conversion_fn` {checkpoint_conversion_fn!r}: {error}") from error
  if not callable(fn):
    raise ValueError(f"`checkpoint_conversion_fn` {checkpoint_conversion_fn!r} is not a function.")
  return fn


def _load_full_state_from_path(
    path, abstract_unboxed_pre_state, checkpoint_conversion_fn, source_checkpoint_layout,
    checkpoint_storage_concurrent_gb, use_ocdbt, use_zarr3, enable_single_replica_ckpt_restoring: bool = False,
    maxtext_config=None,
):
  if source_checkpoint_layout == "orbax":
    if isinstance(abstract_unboxed_pre_state, nnx.State):
      return _load_linen_checkpoint_into_nnx(
          path, abstract_unboxed_pre_state, checkpoint_storage_concurrent_gb, use_ocdbt, use_zarr3,
          enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring, config=maxtext_config,
      )
    context = checkpoint_context.build_context(
        use_ocdbt=use_ocdbt, use_zarr3=use_zarr3, checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
        checkpoint_layout=ocp.options.CheckpointLayout.ORBAX,
        enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
        colocated_python_checkpointing=bool(getattr(maxtext_config, "colocated_python_checkpointing", False)),
    )
    with context:
      return ocp.load(path, abstract_unboxed_pre_state)
  if source_checkpoint_layout == "safetensors":
    if enable_single_replica_ckpt_restoring:
      max_logging.warning("enable_single_replica_ckpt_restoring is not supported for safetensors layout.")
    conversion_fn = _resolve_conversion_fn(checkpoint_conversion_fn)
    context = checkpoint_context.build_context(
        checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
        checkpoint_layout=ocp.options.CheckpointLayout.SAFETENSORS,
    )
    with context:
      metadata = ocp.metadata(path)
      simple_abstract_state = metadata.metadata
      shardings = sharding_utils.construct_maximal_shardings(simple_abstract_state)

      def combine_sharding(sds, shardings):
        return jax.ShapeDtypeStruct(shape=sds.shape, dtype=sds.dtype, sharding=shardings)

      sharded_abstract_state = jax.tree.map(combine_sharding, simple_abstract_state, shardings)
      pre_transformed_state = ocp.load(path, sharded_abstract_state)
    state = conversion_fn(pre_transformed_state)
    if isinstance(abstract_unboxed_pre_state, nnx.State) and not isinstance(state, nnx.State):
      state = restored_linen_to_nnx(state, abstract_unboxed_pre_state, config=maxtext_config)
    return state
  raise ocp.errors.InvalidLayoutError(f"Unknown checkpoint layout: {source_checkpoint_layout}")
