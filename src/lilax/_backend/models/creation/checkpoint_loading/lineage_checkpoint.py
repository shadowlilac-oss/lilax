"""Detect and restore MaxText-layout checkpoints for native Lineage models."""

import contextlib

from etils import epath
from flax import nnx
from orbax import checkpoint as ocp

from lilax._backend.common.checkpoints import checkpoint_context, checkpointing, restore_source_types
from lilax._backend.experimental.lineage import lineage_adapter


def is_maxtext_layout(config):
  """Return whether the checkpoint at load_parameters_path stores MaxText rather than Lineage parameters."""
  path = epath.Path(restore_source_types._normalize_checkpoint_root(config.load_parameters_path))
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
  return "maxtext" in formats


def restore_maxtext_layout(model, sharded_state, config, mesh):
  """Restore MaxText-layout checkpoint parameters into a native Lineage model."""
  raw_params = checkpointing.load_params_from_path(
      config.load_parameters_path, lineage_adapter.maxtext_restore_target(config, mesh),
      config.checkpoint_storage_concurrent_gb, use_ocdbt=config.checkpoint_storage_use_ocdbt,
      use_zarr3=config.checkpoint_storage_use_zarr3,
      enable_single_replica_ckpt_restoring=bool(config.enable_single_replica_ckpt_restoring),
      colocated_python_checkpointing=bool(config.enable_single_controller and config.colocated_python_checkpointing),
  )
  lineage_adapter.restore_params(sharded_state, raw_params, config, mesh)
  nnx.update(model, sharded_state)
