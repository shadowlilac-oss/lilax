"""Stitch vision and LLM checkpoints into a unified model checkpoint."""

from typing import Any, Dict

from etils import epath
from flax import nnx
import jax
from orbax import checkpoint as ocp

from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.utils import max_logging, max_utils, maxtext_utils, maxtext_utils_nnx, model_creation_utils


def _unwrap_var(v):
  """Unwrap flax nnx variables to raw JAX arrays."""
  return v.get_value() if hasattr(v, "get_value") else v


def _restore_subtrees_from_path(
    ckpt_path: str, subtrees_abstract: Dict[str, Any], ckptr: ocp.Checkpointer
) -> Dict[str, Any]:
  """Restore subtrees using the source checkpoint's parameter layout."""
  metadata = ckptr.metadata(epath.Path(ckpt_path))
  tree = metadata.item_metadata.tree
  has_params_params = "params" in tree and isinstance(tree.get("params"), dict) and "params" in tree["params"]
  has_params = "params" in tree and not has_params_params
  has_base = "base" in tree

  subtrees_restore_args = ocp.checkpoint_utils.construct_restore_args(subtrees_abstract)
  if has_params_params:
    item = {"params": {"params": subtrees_abstract}}
    restore_args = {"params": {"params": subtrees_restore_args}}
  elif has_params:
    item = {"params": subtrees_abstract}
    restore_args = {"params": subtrees_restore_args}
  elif has_base:
    item = {"base": subtrees_abstract}
    restore_args = {"base": subtrees_restore_args}
  else:
    item = subtrees_abstract
    restore_args = subtrees_restore_args

  restored = ckptr.restore(epath.Path(ckpt_path), item=item, transforms={}, restore_args=restore_args)
  if has_params_params:
    return restored["params"]["params"]
  elif has_params:
    return restored["params"]
  elif has_base:
    return restored["base"]
  else:
    return restored


def _assemble(k: str, v: Any, stitched_subtrees: Dict[str, Any]) -> Any:
  """Merge restored subtrees with freshly initialized target parameters."""
  if k in stitched_subtrees:
    restored_val = stitched_subtrees[k]
    if isinstance(restored_val, dict) and isinstance(v, dict):
      return {
          sub_module_name: restored_val.get(sub_module_name, fresh_weights)
          for sub_module_name, fresh_weights in v.items()
      }
    return restored_val

  max_logging.log(f"Keeping fresh random normal initialization for new layer: '{k}'")
  return v


def stitch_and_save_checkpoints(
    config: Any,
    vision_checkpoint_path: str,
    llm_checkpoint_path: str,
    output_checkpoint_path: str,
):
  """Stitch vision and LLM weights into a MaxText checkpoint using the target config."""
  max_logging.log("=" * 60)
  max_logging.log("Starting Omni Multi-Directory Checkpoint Stitching...")

  vision_model_name = getattr(config, "vision_encoder_block", None)
  llm_model_name = getattr(config, "decoder_block", None)
  vision_projector_type = getattr(config, "vision_projector_type", None)
  assert vision_model_name, "vision_encoder_block must be configured for vision component."
  assert llm_model_name, "decoder_block must be configured for LLM component."
  assert vision_projector_type, "vision_projector_type must be configured for vision component."

  max_logging.log(f"  Vision (Model {vision_model_name}) Path: {vision_checkpoint_path}")
  max_logging.log(f"  LLM (Model {llm_model_name}) Path:    {llm_checkpoint_path}")
  max_logging.log(f"  Projector Type:        {vision_projector_type}")
  max_logging.log(f"  Output Stitched Path:  {output_checkpoint_path}")
  max_logging.log("=" * 60)

  mesh = maxtext_utils.get_mesh_from_config(config)
  init_rng = jax.random.PRNGKey(config.init_weights_seed)
  max_logging.log("Generating target omni model from config with initial random weights...")
  with jax.set_mesh(mesh):
    rngs = maxtext_utils_nnx.create_nnx_rngs(config, rng_key=init_rng)
    model = model_creation_utils.from_config(config, mesh=mesh, rngs=rngs)
    init_params = nnx.state(model, nnx.Param)

  is_nnx = isinstance(init_params, nnx.State)
  params_dict = init_params.to_pure_dict() if is_nnx else init_params
  inner_params = params_dict.get("params", params_dict)
  inner_params = jax.tree.map(_unwrap_var, inner_params, is_leaf=lambda n: isinstance(n, nnx.Variable))

  ckptr = ocp.Checkpointer(
      ocp.PyTreeCheckpointHandler(
          restore_concurrent_gb=config.checkpoint_storage_concurrent_gb,
          save_concurrent_gb=config.checkpoint_storage_concurrent_gb,
          use_ocdbt=config.checkpoint_storage_use_ocdbt,
          use_zarr3=config.checkpoint_storage_use_zarr3,
      )
  )
  stitched_subtrees = {}

  if "vision_encoder" in inner_params and vision_checkpoint_path:
    max_logging.log(f"Restoring 'vision_encoder' from {vision_checkpoint_path}...")
    vision_encoder_abstract = {
        k: v
        for k, v in inner_params["vision_encoder"].items()
        if not ("projector" in k.lower() or "embedder" in k.lower())
    }
    vision_abstract = {"vision_encoder": vision_encoder_abstract}
    vision_restored = _restore_subtrees_from_path(vision_checkpoint_path, vision_abstract, ckptr)
    stitched_subtrees["vision_encoder"] = vision_restored["vision_encoder"]

  llm_keys = [k for k in ["decoder", "token_embedder"] if k in inner_params]
  if llm_keys and llm_checkpoint_path:
    max_logging.log(f"Restoring LLM subtrees ({llm_keys}) from {llm_checkpoint_path}...")
    llm_abstract = {k: inner_params[k] for k in llm_keys}
    llm_restored = _restore_subtrees_from_path(llm_checkpoint_path, llm_abstract, ckptr)
    for k in llm_keys:
      stitched_subtrees[k] = llm_restored[k]

  stitched_inner = {k: _assemble(k, v, stitched_subtrees) for k, v in inner_params.items()}
  final_params = {"params": stitched_inner}

  output_checkpoint_dir = epath.Path(output_checkpoint_path)
  if output_checkpoint_dir.name == "items":
    output_checkpoint_dir = output_checkpoint_dir.parent
  max_logging.log(f"Saving stitched checkpoint to: {output_checkpoint_dir}")
  checkpointing.save_params_to_path(
      str(output_checkpoint_dir),
      final_params,
      use_ocdbt=config.checkpoint_storage_use_ocdbt,
      use_zarr3=config.checkpoint_storage_use_zarr3,
  )
  total_params = max_utils.calculate_num_params_from_pytree(final_params)
  max_logging.log(f"Total Stitched Model Parameters: {total_params:,} (~{total_params/1e9:.3f}B)")
  max_logging.log("Checkpoint stitching complete!")
