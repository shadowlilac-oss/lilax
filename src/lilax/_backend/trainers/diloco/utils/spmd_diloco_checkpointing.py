"""Checkpointing and restoration for SPMD DiLoCo."""

from typing import Any

from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
import optax
from orbax.checkpoint import v1 as ocp

from lilax._backend.common.checkpoints import checkpoint_context
from lilax._backend.common import train_state_nnx
from lilax._backend.trainers.diloco import diloco
from lilax._backend.trainers.diloco.utils.nnx_state_utils import replace_nnx_model_params


def restore_diloco_checkpoint(
    path: str | epath.Path, abstract_nnx_state: Any, checkpoint_storage_concurrent_gb: int,
    use_ocdbt: bool = True, use_zarr3: bool = True, config: Any = None,
) -> Any:
  """Restore a checkpoint into a DiLoCo train state."""
  diloco_abstract = to_diloco_checkpoint_dict(abstract_nnx_state, config=config)
  root = epath.Path(str(path).rstrip("/"))
  if root.name == "items":
    root = root.parent
  context = checkpoint_context.build_context(
      use_ocdbt=use_ocdbt, use_zarr3=use_zarr3,
      checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb, partial_load=True,
  )
  with context:
    checkpointable_name = "items" if (root / "items").exists() else None
    restored = ocp.load(root, diloco_abstract, checkpointable_name=checkpointable_name)
  return from_diloco_checkpoint_dict(restored, abstract_nnx_state, config=config)


def is_diloco_checkpoint(restored_dict: Any) -> bool:
  """Check whether the checkpoint contains the complete multi-replica state."""
  if not isinstance(restored_dict, dict):
    return False
  return "inner_state" in restored_dict and "outer_opt_state" in restored_dict


def to_diloco_checkpoint_dict(state: Any, config: Any = None) -> dict[str, Any]:
  """Package a DiLoCo or abstract single-replica state for Orbax."""
  num_replicas = getattr(config, "dcn_diloco_parallelism", getattr(config, "num_diloco_replicas", 2))
  if isinstance(state, diloco.DiLoCoTrainState):
    inner_state = state.inner_state
    params = state.params
    outer_opt_state = state.outer_opt_state
    step = state.step
  elif hasattr(state, "inner_state"):
    inner_state = state.inner_state
    params = getattr(state, "params", None)
    outer_opt_state = getattr(state, "outer_opt_state", None)
    step = getattr(state, "step", jnp.array(0, dtype=jnp.int32))
  else:
    def add_diloco_dim(leaf):
      if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        new_shape = (num_replicas, *leaf.shape)
        sharding = getattr(leaf, "sharding", None)
        if isinstance(sharding, jax.sharding.NamedSharding) and "diloco" in sharding.mesh.axis_names:
          new_spec = jax.sharding.PartitionSpec("diloco", *sharding.spec)
          sharding = jax.sharding.NamedSharding(mesh=sharding.mesh, spec=new_spec)
        if isinstance(leaf, jax.ShapeDtypeStruct):
          return jax.ShapeDtypeStruct(new_shape, leaf.dtype, sharding=sharding)
        return jnp.broadcast_to(leaf, new_shape)
      return leaf

    inner_state = jax.tree_util.tree_map(add_diloco_dim, state)
    if hasattr(state, "model"):
      _, params, _ = nnx.split(state.model, nnx.Param, ...)
      params = params.to_pure_dict() if hasattr(params, "to_pure_dict") else params
    elif hasattr(state, "params"):
      params = state.params
    else:
      params = state
    outer_optimizer = optax.sgd(
        getattr(config, "diloco_outer_lr", 0.1), momentum=getattr(config, "diloco_outer_momentum", 0.9), nesterov=True,
    )
    outer_opt_state = outer_optimizer.init(params)
    step = getattr(getattr(state, "optimizer", None), "step", jnp.array(0, dtype=jnp.int32))
  if isinstance(inner_state, (nnx.State, train_state_nnx.TrainStateNNX)):
    inner_state_dict = train_state_nnx.to_checkpoint_dict(inner_state)
  elif hasattr(inner_state, "to_pure_dict"):
    inner_state_dict = inner_state.to_pure_dict()
  elif isinstance(inner_state, dict):
    inner_state_dict = inner_state
  else:
    inner_state_dict = inner_state
  if hasattr(params, "to_pure_dict"):
    params_dict = params.to_pure_dict()
  elif isinstance(params, (nnx.State, nnx.Module)):
    params_dict = nnx.state(params).to_pure_dict()
  elif isinstance(params, dict):
    params_dict = params
  else:
    params_dict = params
  step_val = step.get_value() if hasattr(step, "get_value") else step
  if isinstance(params_dict, dict) and "params" not in params_dict:
    outer_params_dict = {"params": params_dict}
  else:
    outer_params_dict = params_dict
  return {"inner_state": inner_state_dict, "params": outer_params_dict, "outer_opt_state": outer_opt_state, "step": step_val}


def from_diloco_checkpoint_dict(restored_dict: dict[str, Any], abstract_diloco_state: Any, config: Any = None) -> Any:
  """Restore full DiLoCo checkpoints or model-only checkpoints."""
  num_replicas = getattr(config, "dcn_diloco_parallelism", getattr(config, "num_diloco_replicas", 2))

  def add_diloco_dim(leaf):
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
      new_shape = (num_replicas, *leaf.shape)
      sharding = getattr(leaf, "sharding", None)
      if isinstance(sharding, jax.sharding.NamedSharding) and "diloco" in sharding.mesh.axis_names:
        new_spec = jax.sharding.PartitionSpec("diloco", *sharding.spec)
        sharding = jax.sharding.NamedSharding(mesh=sharding.mesh, spec=new_spec)
      if isinstance(leaf, jax.ShapeDtypeStruct):
        return jax.ShapeDtypeStruct(new_shape, leaf.dtype, sharding=sharding)
      return jnp.broadcast_to(leaf, new_shape)
    return leaf

  if is_diloco_checkpoint(restored_dict):
    inner_dict = restored_dict["inner_state"]
    if isinstance(abstract_diloco_state, diloco.DiLoCoTrainState):
      abstract_inner = abstract_diloco_state.inner_state
    else:
      abstract_inner = jax.tree_util.tree_map(add_diloco_dim, abstract_diloco_state)
    if isinstance(abstract_inner, nnx.Module):
      abstract_inner = nnx.state(abstract_inner)
    if isinstance(abstract_inner, (nnx.State, train_state_nnx.TrainStateNNX)):
      linen_state, aux_state, ephemeral = train_state_nnx.split_for_checkpoint(abstract_inner)
      weights = train_state_nnx.from_linen_checkpoint_dict(inner_dict)
      if "model" in weights:
        nnx.replace_by_pure_dict(linen_state, {"model": weights["model"]})
      if "optimizer" in weights:
        nnx.replace_by_pure_dict(linen_state, {"optimizer": weights["optimizer"]})
      nnx_aux = inner_dict.get("nnx_aux")
      if nnx_aux:
        train_state_nnx.apply_checkpoint_aux(aux_state, nnx_aux)
      inner_state = nnx.merge_state(linen_state, aux_state, ephemeral)
    else:
      inner_state = inner_dict
    params = restored_dict["params"]
    if isinstance(params, dict) and "params" in params:
      params = params["params"]
    return diloco.DiLoCoTrainState(
        inner_state=inner_state, params=params, outer_opt_state=restored_dict["outer_opt_state"], step=restored_dict["step"],
    )
  raw_params = None
  if "params" in restored_dict:
    raw_params = restored_dict["params"]
    if isinstance(raw_params, dict) and "params" in raw_params:
      raw_params = raw_params["params"]
  elif "model" in restored_dict:
    raw_params = restored_dict["model"]
  else:
    raw_params = restored_dict
  step = restored_dict.get("step", jnp.array(0, dtype=jnp.int32))
  broadcasted_model_params = jax.tree_util.tree_map(add_diloco_dim, raw_params)
  if isinstance(abstract_diloco_state, diloco.DiLoCoTrainState):
    abstract_inner = abstract_diloco_state.inner_state
  else:
    abstract_inner = jax.tree_util.tree_map(add_diloco_dim, abstract_diloco_state)
  inner_state = replace_nnx_model_params(abstract_inner, broadcasted_model_params)
  outer_optimizer = optax.sgd(
      getattr(config, "diloco_outer_lr", 0.1), momentum=getattr(config, "diloco_outer_momentum", 0.9), nesterov=True,
  )
  outer_opt_state = outer_optimizer.init(raw_params)
  return diloco.DiLoCoTrainState(inner_state=inner_state, params=raw_params, outer_opt_state=outer_opt_state, step=step)
