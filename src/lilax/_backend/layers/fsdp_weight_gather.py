"""FSDP weight gathering and its rematerialization policy."""

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from lilax._backend.common.common_types import ShardMode
from lilax._backend.utils import sharding

_FSDP_MESH_AXES = ("fsdp", "fsdp_transpose")
_GATHERED_FSDP_WEIGHT = "gathered_fsdp_weight"


def _drop_fsdp_mesh_axes(entry):
  """Return one PartitionSpec entry with the FSDP mesh axes removed."""
  if entry is None:
    return None
  if isinstance(entry, str):
    return None if entry in _FSDP_MESH_AXES else entry
  rest = tuple(a for a in entry if a not in _FSDP_MESH_AXES)
  if not rest:
    return None
  return rest[0] if len(rest) == 1 else rest


def save_gathered_fsdp_weights_policy(policy):
  """Extend a remat policy so the hoisted FSDP weight gathers survive as residuals."""
  keep = jax.checkpoint_policies.save_only_these_names(_GATHERED_FSDP_WEIGHT)
  return keep if policy is None else jax.checkpoint_policies.save_from_both_policies(policy, keep)


def hoist_fsdp_weight_gather(params, mesh, config):
  """Gather each FSDP-sharded layer kernel once, in compute dtype, at the top of a scan body.

  Left alone, the FSDP all-gather of a weight is emitted wherever the weight is first
  read, which under remat means once in the forward pass and again in the recomputed
  backward pass. Doing it explicitly here gives the gather a name, so the remat policy
  can keep the gathered value as a residual and the backward pass reads it instead of
  re-gathering. See `save_fsdp_gathered_weights` in base.yml.

  Numerics are unchanged: the only arithmetic difference is that the cast to
  `config.dtype` happens before the gather rather than inside the consuming dot, and
  only kernels (rank >= 2 float arrays) are touched, never the rank-1 norm scales.
  """
  if not config.save_fsdp_gathered_weights:
    return params

  def gathered(value, spec):
    """(pspec, cast value) for a kernel worth gathering here, or None to leave it be."""
    new_spec = tuple(_drop_fsdp_mesh_axes(e) for e in spec)
    if tuple(spec) == new_spec:
      return None
    if getattr(value, "ndim", 0) < 2 or not jnp.issubdtype(value.dtype, jnp.floating):
      return None
    # Every consumer casts its kernel to config.dtype before the dot, so doing it here
    # instead is arithmetically identical -- but it halves both the bytes on the wire
    # and the bytes the residual costs in HBM. That halving is what makes the residual
    # pay for itself: kept at f32 it is a wash (see the guide, section 4.11).
    return jax.sharding.PartitionSpec(*new_spec), jnp.asarray(value, config.dtype)

  if config.shard_mode == ShardMode.EXPLICIT:

    def hoist(x):
      if not hasattr(x, "shape") or getattr(x, "ndim", 0) < 2:
        return x
      spec = getattr(jax.typeof(x).sharding, "spec", None)
      if spec is None:
        return x
      out = gathered(x, tuple(spec))
      if out is None:
        return x
      pspec, value = out
      return checkpoint_name(jax.sharding.reshard(value, pspec), _GATHERED_FSDP_WEIGHT)

    return jax.tree.map(hoist, params)

  # Under shard_mode: auto jax.typeof(x).sharding.spec is all-None inside the traced
  # body, so the logical axis names have to be read off the nnx.Variable metadata and
  # lowered through the logical axis rules by hand.
  rules = sharding.get_logical_axis_rules() or config.logical_axis_rules

  def hoist_auto(x):
    if not isinstance(x, nnx.Variable) or x.type.__name__ == "_overwrite_with_gradient":
      return x
    metadata = x.get_metadata()
    # Same precedence `variable_to_logically_partitioned` uses, so both read the same names.
    names = next((metadata[k] for k in ("out_sharding", "sharding_names", "sharding") if metadata.get(k)), None)
    value = x.get_value()
    if not names or not hasattr(value, "shape") or len(names) != getattr(value, "ndim", 0):
      return x
    out = gathered(value, tuple(sharding.logical_to_mesh_axes(tuple(names), mesh, rules=rules)))
    if out is None:
      return x
    pspec, value = out
    pinned = jax.lax.with_sharding_constraint(value, jax.sharding.NamedSharding(mesh, pspec))
    return x.replace(checkpoint_name(pinned, _GATHERED_FSDP_WEIGHT))

  return jax.tree.map(hoist_auto, params, is_leaf=lambda x: isinstance(x, nnx.Variable))
