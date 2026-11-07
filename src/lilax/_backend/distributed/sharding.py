"""Sharding utilities for native MaxText consumers."""

import collections
from collections.abc import Iterable
import inspect
from pathlib import Path

from flax import nnx
from flax.core.spmd import get_logical_axis_rules as flax_get_logical_axis_rules
import jax
from jax.core import Tracer
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, reshard
from lilax._backend.common.common_types import ShardMode
from lilax._backend.observability import max_logging
from lilax._backend.observability.diagnostics import pytree
from orbax.checkpoint._src.arrays.sharding import construct_maximal_shardings

_LOGGED_ACTIVATION_SHARDINGS = set()
_ACTIVATION_SHARDINGS_DUMP = []


def clear_input_shardings_dump():
  """Clear the input shardings dump."""
  _LOGGED_ACTIVATION_SHARDINGS.clear()
  _ACTIVATION_SHARDINGS_DUMP.clear()


def get_input_data_sharding(config, mesh, rules=None):
  """Get the input data sharding for the model."""
  if rules is None:
    rules = config.logical_axis_rules
  if config.enable_diloco:
    return create_sharding(mesh, ["diloco"] + config.input_data_sharding_logical_axes, rules=rules)
  return create_sharding(mesh, config.input_data_sharding_logical_axes, rules=rules)


def get_logical_axis_rules():
  """Get the logical axis rules for the flax model"""
  return flax_get_logical_axis_rules()


class _UnassignedAxis:
  """Marks a logical dimension that no rule has mapped to a mesh axis yet."""

  def __repr__(self):
    return "UnassignedAxis"

  def __bool__(self):
    return False


_UNASSIGNED_AXIS = _UnassignedAxis()


def _mesh_axes_free(new_assignment, existing_assignments):
  """True if none of the mesh axes in `new_assignment` is used in `existing_assignments`."""
  new = set(jax.tree_util.tree_leaves(new_assignment))
  existing = set(jax.tree_util.tree_leaves(existing_assignments))
  new.discard(P.UNCONSTRAINED)
  new.discard(None)
  return not existing.intersection(new)


def logical_to_spec(logical_names, rules=None):
  """Maps logical dimension names to a PartitionSpec using the logical axis rules.

  Rules are (logical name, mesh axes) pairs in priority order. A dimension takes
  the first matching rule whose mesh axes no other dimension uses yet;
  dimensions no rule assigns are left unsharded. Without `rules`, uses the ones
  set by `logical_axis_rules`. Same behavior as Flax's
  `flax.linen.logical_to_mesh_axes`.
  """
  if logical_names is None:
    return None
  if rules is None:
    rules = get_logical_axis_rules()
  counts = collections.Counter(logical_names)
  dups = tuple(k for k, v in counts.items() if v > 1 and isinstance(k, str))
  if dups:
    raise ValueError(f"Unsupported: Dimensions {dups} occur more than once in array names.")
  if not isinstance(rules, (tuple, list)):
    raise ValueError("Unknown axis rule specification type.")
  result = [(_UNASSIGNED_AXIS if isinstance(name, str) else name) for name in logical_names]
  for rule_logical_name, rule_mesh_axes in rules:
    if rule_logical_name in logical_names:
      pos = logical_names.index(rule_logical_name)
      if _mesh_axes_free(rule_mesh_axes, result) and result[pos] is _UNASSIGNED_AXIS:
        result[pos] = rule_mesh_axes
  return P(*(None if x is _UNASSIGNED_AXIS else x for x in result))


def with_logical_constraint(x, logical_axes):
  """Apply an ambient logical sharding constraint when one is available."""
  rules = get_logical_axis_rules()
  if not rules or logical_axes is None or jax.sharding.get_abstract_mesh().empty:
    return x
  return jax.lax.with_sharding_constraint(x, logical_to_spec(logical_axes, rules))


def _get_sharding_desc(inputs, extra_stack_level):
  """Get the inputs sharding description using inspect."""
  frame = inspect.currentframe()
  for _ in range(1 + extra_stack_level):
    if frame is not None:
      frame = frame.f_back
  if frame is not None:
    callers_local_vars = frame.f_locals.items()
    names = [var_name for var_name, var_val in callers_local_vars if var_val is inputs]
    if names:
      caller_filename = Path(inspect.stack()[1 + extra_stack_level].filename).name
      return f"{caller_filename[:-3]}/{names[0]}"
  return "Unknown"


def maybe_shard_with_name(
    inputs, named_sharding, shard_mode, debug_sharding=False, extra_stack_level=0, sharding_desc="", logical_axes=None
):
  """Hint or enforce a named sharding according to shard mode."""
  if inputs is None:
    return None
  if hasattr(inputs, "ndim"):
    named_sharding = truncate_out_sharding(named_sharding, inputs.ndim)
  if (
      isinstance(named_sharding, NamedSharding)
      and hasattr(inputs, "shape")
      and getattr(named_sharding, "mesh", None) is not None
      and (isinstance(inputs, nnx.Variable) or hasattr(inputs, "value"))
  ):
    adj_spec = adjust_pspec_for_indivisible_shapes(named_sharding.spec, inputs.shape, named_sharding.mesh)
    if adj_spec != named_sharding.spec:
      named_sharding = NamedSharding(named_sharding.mesh, adj_spec)
  if debug_sharding and isinstance(inputs, Tracer) and isinstance(named_sharding, NamedSharding):
    if not sharding_desc:
      sharding_desc = _get_sharding_desc(inputs, extra_stack_level + 1)
    if not logical_axes:
      logical_axes = "Unknown"
    elif isinstance(logical_axes, list):
      logical_axes = tuple(logical_axes)
    pspec = remove_size_one_mesh_axis(getattr(named_sharding, "spec"), getattr(named_sharding, "mesh"))
    log_key = (sharding_desc, str(jax.typeof(inputs)), tuple(pspec), extra_stack_level)
    if log_key not in _LOGGED_ACTIVATION_SHARDINGS:
      max_logging.info(f"{sharding_desc} Logical: {log_key[1]:.<60} {logical_axes}.", stacklevel=3 + extra_stack_level)
      max_logging.info(f"{sharding_desc} Physical: {log_key[1]:.<60} {log_key[2]}.", stacklevel=3 + extra_stack_level)
      _LOGGED_ACTIVATION_SHARDINGS.add(log_key)
      _ACTIVATION_SHARDINGS_DUMP.append(
          {f"{sharding_desc}: {log_key[1]}": {"logic_axes": f"{logical_axes}", "PartitionSpec": f"P{log_key[2]}"}}
      )
  if shard_mode == ShardMode.EXPLICIT:
    return reshard(inputs, named_sharding)
  return jax.lax.with_sharding_constraint(inputs, named_sharding)


def maybe_shard_with_pspec(
    inputs, pspec: P | None, mesh, shard_mode, debug_sharding=False, extra_stack_level=0, logical_axes=None
):
  """Apply a physical sharding constraint while preserving logical-axis debug metadata."""
  if pspec is None:
    return None
  return maybe_shard_with_name(
      inputs, NamedSharding(mesh, pspec), shard_mode=shard_mode, debug_sharding=debug_sharding,
      extra_stack_level=extra_stack_level + 1, logical_axes=logical_axes,
  )


def maybe_shard_with_logical(
    inputs, logical_axes, mesh, shard_mode, rules=None, debug_sharding=False, extra_stack_level=0, sharding_desc=""
):
  """Apply a sharding constraint described by logical axes."""
  if inputs is None:
    return None
  if debug_sharding and not sharding_desc:
    sharding_desc = _get_sharding_desc(inputs, extra_stack_level + 1)
  return maybe_shard_with_name(
      inputs, create_sharding(mesh, logical_axes, rules=rules), shard_mode, debug_sharding=debug_sharding,
      extra_stack_level=extra_stack_level + 1, sharding_desc=sharding_desc, logical_axes=logical_axes,
  )


def remove_size_one_mesh_axis(spec, mesh):
  """Removes mesh axes from a PartitionSpec (P) where the axis size is 1."""
  if spec is None:
    return None
  new_spec = []
  for s in spec:
    if s is None or s == P.UNCONSTRAINED:
      new_spec.append(s)
    elif isinstance(s, tuple):
      new_spec.append(tuple(i for i in s if mesh.shape.get(i, 1) != 1))
    else:
      new_spec.append(None if mesh.shape.get(s, 1) == 1 else s)
  return P(*new_spec, unreduced=spec.unreduced, reduced=spec.reduced)


def mesh_axes_for_dim(axis_names):
  """Return the mesh axes attached to one tensor dimension."""
  if axis_names is None:
    return ()
  if isinstance(axis_names, str):
    return (axis_names,)
  return tuple(axis for axis in axis_names if axis is not None)


def batch_mesh_axes(mesh, rules=None):
  """Return nontrivial mesh axes sharding the activation batch dimension."""
  spec = logical_to_mesh_axes(("activation_batch",), mesh, rules=rules)
  if spec is None or not spec.partitions:
    return frozenset()
  return frozenset(axis for axis in mesh_axes_for_dim(spec.partitions[0]) if mesh.shape.get(axis, 1) > 1)


def mesh_axes_size(mesh, axes, *, label):
  """Return the product of mesh sizes for a set of axes."""
  size = 1
  for axis in axes:
    if axis not in mesh.shape:
      raise ValueError(f"{label} requires mesh axis {axis!r} to exist.")
    size *= mesh.shape[axis]
  return size


def with_axis_on_dim(axis_names, axis, dim):
  """Return sharding axis names with one dimension replaced."""
  if axis_names is None:
    return None
  if dim >= len(axis_names):
    raise ValueError(f"Dimension {dim} is out of range for sharding axis names {axis_names}.")
  axes = list(axis_names)
  axes[dim] = axis
  if isinstance(axis_names, P):
    return P(*axes, unreduced=axis_names.unreduced, reduced=axis_names.reduced)
  if isinstance(axis_names, tuple):
    return tuple(axes)
  return axes


def adjust_pspec_for_indivisible_shapes(spec: P, shape: tuple[int, ...], mesh) -> P:
  """Removes physical mesh axes from spec where array dimension is not divisible by the mesh axis size."""
  if spec is None or mesh is None or not shape:
    return spec
  new_spec = []
  for i, s in enumerate(spec):
    if i >= len(shape) or s is None or s == P.UNCONSTRAINED:
      new_spec.append(s)
    else:
      dim_len = shape[i]
      if isinstance(s, tuple):
        valid_axes = []
        cum_product = 1
        for axis_name in s:
          axis_size = mesh.shape.get(axis_name, 1) if hasattr(mesh, "shape") else 1
          if dim_len % (cum_product * axis_size) == 0:
            valid_axes.append(axis_name)
            cum_product *= axis_size
        new_spec.append(tuple(valid_axes) if valid_axes else None)
      else:
        axis_size = mesh.shape.get(s, 1) if hasattr(mesh, "shape") else 1
        if dim_len % axis_size == 0:
          new_spec.append(s)
        else:
          new_spec.append(None)
  return P(*new_spec, unreduced=spec.unreduced, reduced=spec.reduced)


def get_nnx_var_named_sharding_with_scan_axis(v: nnx.Variable, mesh) -> nnx.Variable:
  """Compute NamedSharding for an NNX variable, correctly handling the scan axis."""
  val = v.get_value()
  if not hasattr(val, "shape"):
    if jax.tree_util.tree_leaves(val):
      first_leaf = jax.tree_util.tree_leaves(val)[0]
      if hasattr(first_leaf, "shape"):
        leaf_var = get_nnx_var_named_sharding_with_scan_axis(v.replace(value=first_leaf), mesh)
        leaf_sharding = leaf_var.get_value()
        if not isinstance(leaf_sharding, NamedSharding):
          leaf_sharding = NamedSharding(mesh, P())
        return v.replace(jax.tree.map(lambda _: leaf_sharding, val))
      replicated = NamedSharding(mesh, P())
      return v.replace(jax.tree.map(lambda _: replicated, val))
    return v
  metadata = v.get_metadata()
  out_sharding = metadata.get("out_sharding") or metadata.get("sharding_names") or metadata.get("sharding")
  if not out_sharding:
    pspec = P()
  elif isinstance(out_sharding, jax.sharding.NamedSharding):
    return v.replace(out_sharding)
  elif isinstance(out_sharding, jax.sharding.PartitionSpec):
    pspec = out_sharding
  else:
    out_sharding = [out_sharding] if isinstance(out_sharding, str) else list(out_sharding)
    if "param_scan_axis" in metadata and nnx.PARTITION_NAME in metadata:
      partition_name = metadata[nnx.PARTITION_NAME]
      scan_axis = metadata.get("param_scan_axis", 0)
      if partition_name not in out_sharding:
        out_sharding.insert(scan_axis, partition_name)
    elif len(val.shape) > len(out_sharding):
      diff = len(val.shape) - len(out_sharding)
      out_sharding = list(out_sharding) + [None] * diff
    out_sharding = tuple(out_sharding)
    context_rules = get_logical_axis_rules()
    local_rules = metadata.get("sharding_rules", ())
    if context_rules or local_rules:
      local_rules_list = list(local_rules) if local_rules is not None else []
      context_rules_list = list(context_rules) if context_rules is not None else []
      rules = local_rules_list + context_rules_list
      pspec = logical_to_mesh_axes(out_sharding, mesh, rules=rules)
    else:
      pspec = P(*out_sharding)
      if mesh is not None:
        pspec = remove_size_one_mesh_axis(pspec, mesh)
    if pspec is not None and nnx.PARTITION_NAME not in metadata:
      orig_sharding = metadata.get("out_sharding") or metadata.get("sharding_names") or metadata.get("sharding")
      if isinstance(orig_sharding, str):
        orig_len = 1
      elif isinstance(orig_sharding, (list, tuple)):
        orig_len = len(orig_sharding)
      else:
        orig_len = 0
      if 0 < orig_len < len(pspec):
        pspec = P(*pspec[:orig_len])
  return v.replace(NamedSharding(mesh, pspec))


def nnx_construct_named_sharding(abs_var_state: nnx.State, mesh) -> nnx.State:
  """Compute NamedSharding for each NNX variable, correctly handling the scan (stacked layers) axis."""
  return jax.tree.map(
      lambda x: get_nnx_var_named_sharding_with_scan_axis(x, mesh),
      abs_var_state,
      is_leaf=lambda x: isinstance(x, nnx.Variable),
  )


class CompoundLogicalAxis(tuple):
  """Logical name of one dimension sharded over several logical axes."""


jax.tree_util.register_pytree_node(
    CompoundLogicalAxis, lambda x: (tuple(x), None), lambda _, children: CompoundLogicalAxis(children)
)


def logical_to_mesh_axes(logical_names, mesh, rules=None):
  """Remove size one mesh axes given logical names."""
  if logical_names is None or not any(isinstance(name, CompoundLogicalAxis) for name in logical_names):
    return remove_size_one_mesh_axis(logical_to_spec(logical_names, rules=rules), mesh)
  groups = [tuple(name) if isinstance(name, CompoundLogicalAxis) else (name,) for name in logical_names]
  flat_spec = list(logical_to_spec(tuple(n for group in groups for n in group), rules=rules))
  grouped = []
  for group in groups:
    resolved, flat_spec = flat_spec[: len(group)], flat_spec[len(group) :]
    axes = [ax for item in resolved if item is not None for ax in ((item,) if isinstance(item, str) else item)]
    grouped.append(None if not axes else axes[0] if len(axes) == 1 else tuple(axes))
  return remove_size_one_mesh_axis(P(*grouped), mesh)


def logical_to_mesh(tree, mesh, rules=None):
  """Remove size one mesh axes given a logical spec pytree."""
  if tree is None:
    return None
  return jax.tree.map(lambda x: logical_to_mesh_axes(x, mesh, rules=rules), tree, is_leaf=lambda x: isinstance(x, P))


def logical_to_mesh_sharding(tree, mesh, rules=None):
  """Return a sharding pytree given a logical spec pytree."""
  if tree is None:
    return None
  return jax.tree.map(
      lambda x: NamedSharding(mesh, x), logical_to_mesh(tree, mesh, rules=rules), is_leaf=lambda x: isinstance(x, P)
  )


def create_sharding(mesh, logical_names, rules=None):
  """Create NamedSharding with given logical names."""
  spec = logical_to_mesh_axes(logical_names, mesh, rules=rules)
  if spec is None:
    spec = P()
  return NamedSharding(mesh, spec)


def vocab_parallel_one_hot(targets, num_classes, mesh, logical_axes, shard_mode, *, dtype=jnp.float32):
  """One-hot targets directly in the logits' layout."""
  out_sharding = create_sharding(mesh, logical_axes)
  targets = maybe_shard_with_pspec(targets, P(*out_sharding.spec[: targets.ndim]), mesh, shard_mode)
  shape = (*targets.shape, num_classes)
  if shard_mode == ShardMode.EXPLICIT:
    vocab_ids = jax.lax.broadcasted_iota(targets.dtype, shape, targets.ndim, out_sharding=out_sharding)
  else:
    vocab_ids = maybe_shard_with_pspec(
        jax.lax.broadcasted_iota(targets.dtype, shape, targets.ndim), out_sharding.spec, mesh, shard_mode
    )
  return (vocab_ids == targets[..., None]).astype(dtype)


def lm_head_logical_axes(vocab_parallel: bool):
  """Return logical axes for the selected untied LM head orientation."""
  if vocab_parallel:
    return (
        ("activation_embed_and_logits_batch_no_fsdp", "activation_length", "activation_embed"),
        ("embed_vocab_replicated", "vocab_fsdp"),
        ("activation_embed_and_logits_batch_no_fsdp", "activation_length", "activation_vocab_fsdp"),
    )
  return (
      None,
      ("embed_vocab", "vocab"),
      ("activation_embed_and_logits_batch", "activation_length", "activation_vocab"),
  )


def _truncate_pspec(pspec: P, out_ndim: int) -> P:
  """Drop trailing entries while preserving reduced and unreduced annotations."""
  return P(*pspec.partitions[:out_ndim], unreduced=pspec.unreduced, reduced=pspec.reduced)


def truncate_out_sharding(out_sharding, out_ndim: int):
  """Truncate output sharding when its spec exceeds the tensor rank."""
  if out_sharding is None:
    return None
  if isinstance(out_sharding, NamedSharding):
    if len(out_sharding.spec) > out_ndim:
      return NamedSharding(out_sharding.mesh, _truncate_pspec(out_sharding.spec, out_ndim))
  elif isinstance(out_sharding, P):
    if len(out_sharding) > out_ndim:
      return _truncate_pspec(out_sharding, out_ndim)
  elif isinstance(out_sharding, (tuple, list)):
    if len(out_sharding) > out_ndim:
      return tuple(out_sharding[:out_ndim])
  return out_sharding


def get_mesh_axes_used_by_tensor_spec(tensor_sharding_spec):
  """Extract mesh axes used by a tensor's PartitionSpec."""
  return sum(
      [[axis] if isinstance(axis, str) else list(axis) if isinstance(axis, Iterable) else []
       for axis in tensor_sharding_spec],
      [],
  )


def _get_nontrival_mesh_axes(mesh):
  """Return configured target mesh axes with more than one shard."""
  target_sharding_axes_config = [
      "fsdp", "fsdp_transpose", "sequence", "context", "context_usp_ulysses", "context_autoregressive",
      "tensor", "tensor_sequence", "stage", "expert",
  ]
  return {axis for axis in target_sharding_axes_config if axis in mesh.axis_names and mesh.shape[axis] > 1}


def _analyze_sharding(params, mesh, valid_target_mesh_axes):
  """Find parameters not sharded on all target mesh axes."""
  unsharded_params_total_size = 0
  problematic_tensors_details = []
  for path, p_leaf in jax.tree_util.tree_leaves_with_path(params):
    param_name_str = jax.tree_util.keystr(path)
    is_lora_param = isinstance(p_leaf, getattr(nnx, "LoRAParam", ()))
    is_lora = is_lora_param or "lora" in param_name_str.lower()
    if isinstance(p_leaf, nnx.Variable):
      p_leaf = p_leaf.value
    spec = getattr(getattr(p_leaf, "sharding", None), "spec", None)
    if spec is None and is_lora:
      spec = P()
    assert isinstance(spec, P), f"Expected '.sharding.spec' for parameter '{param_name_str}' to be a PartitionSpec."
    mesh_axes_used = get_mesh_axes_used_by_tensor_spec(spec)
    is_sharded_on_all_target_axis = all(axis in mesh_axes_used for axis in valid_target_mesh_axes)
    if not is_sharded_on_all_target_axis:
      unsharded_params_total_size += p_leaf.size
      unsharded_axes = set(valid_target_mesh_axes) - set(mesh_axes_used)
      problematic_tensors_details.append(
          {
              "name": param_name_str,
              "size": p_leaf.size,
              "shape": p_leaf.shape,
              "spec": str(spec),
              "available_axes": sorted(list(valid_target_mesh_axes)),
              "unsharded_axes": sorted(list(unsharded_axes)),
          }
      )
  return unsharded_params_total_size, problematic_tensors_details


def _raise_if_unsharded_exceeds_tolerance(unsharded_size, total_size, tolerance, problematic_tensors_details):
  """Raise when the proportion of unsharded parameters exceeds tolerance."""
  if total_size <= 0:
    raise ValueError("Total size must be greater than zero.")
  unsharded_param_perc = unsharded_size / total_size
  if unsharded_param_perc > tolerance:
    problematic_tensors_details.sort(key=lambda x: x["size"], reverse=True)
    error_msg_lines = [f"Unsharded parameter percentage ({unsharded_param_perc:.2%})" f"exceeds tolerance ({tolerance:.2%})."]
    error_msg_lines.append(
        "The following large tensors are replicated (unsharded) but could be sharded on at "
        "least one of the available axes:"
    )
    for detail in problematic_tensors_details[:5]:
      error_msg_lines.append(
          f" - Name: {detail['name']}(Size: {detail['size']}, Shape: {detail['spec']}, Spec: {detail['spec']}) "
          f" is unsharded on axis: {detail['unsharded_axes']}"
          f" could be sharded on: {detail['available_axes']}"
      )
    raise AssertionError("\n".join(error_msg_lines))


def assert_params_sufficiently_sharded(params, mesh, tolerance):
  """Assert replicated parameters remain within the configured tolerance."""
  total_num_params = pytree.calculate_bytes_from_pytree(params)
  valid_target_mesh_axes = _get_nontrival_mesh_axes(mesh)
  if not valid_target_mesh_axes:
    return
  unsharded_params_total_size, problematic_tensors_details = _analyze_sharding(params, mesh, valid_target_mesh_axes)
  _raise_if_unsharded_exceeds_tolerance(
      unsharded_params_total_size, total_num_params, tolerance, problematic_tensors_details
  )


def add_data_to_sharding(mesh, path, aval, sharding):
  """Adds 'data' dimension to sharding spec if compatible and not already present."""
  if not isinstance(sharding, jax.sharding.NamedSharding):
    raise AssertionError(f"Expected NamedSharding, found {sharding} of {type(sharding)=} at {jax.tree_util.keystr(path)}")

  pspec = sharding.spec
  if len(pspec) != len(aval.shape):
    if len(pspec) > len(aval.shape):
      if "local_layers" in pspec and len(pspec) - 1 == len(aval.shape):
        pspec = tuple(axis for axis in pspec if axis != "local_layers")
      else:
        pspec = pspec[: len(aval.shape)]
    else:
      pspec = tuple(pspec) + (None,) * (len(aval.shape) - len(pspec))
    sharding = jax.sharding.NamedSharding(sharding.mesh, jax.sharding.PartitionSpec(*pspec))

  try:
    sharded_shape = sharding.shard_shape(aval.shape)
  except Exception as e:
    raise AssertionError(f"Could not shard {jax.tree_util.keystr(path)} of shape={aval.shape} with {sharding=}") from e
  pspec = sharding.spec

  if "data" in jax.tree.leaves(tuple(pspec)):
    return sharding

  for idx, (size, partition) in enumerate(zip(sharded_shape, pspec)):
    if partition is None:
      partition = ()
    if isinstance(partition, str):
      partition = (partition,)
    if size % mesh.shape["data"] == 0 and (partition is None or "tensor" not in partition):
      added_component = ("data",) + partition
      new_pspec = jax.sharding.PartitionSpec(*(pspec[:idx] + (added_component,) + pspec[idx + 1 :]))
      new_sharding = jax.sharding.NamedSharding(sharding.mesh, new_pspec)
      return new_sharding
  return sharding


def maybe_update_params_sharding_with_opt(config, state_mesh_shardings):
  """Update parameter shardings to match optimizer state when enabled."""
  return maybe_update_params_sharding_with_opt_nnx(config, state_mesh_shardings)


def maybe_update_params_sharding_with_opt_nnx(config, state_mesh_shardings) -> tuple[nnx.State, nnx.State]:
  """Update NNX parameter shardings from optimizer first-moment shardings."""
  model_shardings = state_mesh_shardings.model

  def _extract_param_only(state):
    result = {}
    for key, value in state.items():
      if isinstance(value, nnx.Param):
        result[key] = value
      elif isinstance(value, nnx.Variable):
        pass
      elif hasattr(value, "items"):
        sub = _extract_param_only(value)
        if sub:
          result[key] = sub
      else:
        result[key] = value
    return result

  prev_params_shardings = nnx.State(_extract_param_only(model_shardings))
  if not config.shard_optimizer_over_data:
    return prev_params_shardings, state_mesh_shardings

  sharded_fp32_params = None
  if "opt_state" in state_mesh_shardings.optimizer:
    opt_state = state_mesh_shardings.optimizer.opt_state

    def collect_mu_trees(obj, out):
      if isinstance(obj, nnx.Variable):
        return
      if hasattr(obj, "get") and "mu" in obj:
        out.append(obj["mu"])
      elif hasattr(obj, "values"):
        for value in obj.values():
          collect_mu_trees(value, out)
      elif isinstance(obj, (list, tuple)):
        for value in obj:
          collect_mu_trees(value, out)

    mu_trees = []
    collect_mu_trees(opt_state, mu_trees)
    sharded_fp32_params = mu_trees or None
  if sharded_fp32_params is None:
    actual_type = type(state_mesh_shardings.optimizer.get("opt_state", "None"))
    raise NotImplementedError(f"Could not find Adam/Muon optimizer state in: {actual_type}")

  mu_lookup = {}
  for mu_tree in mu_trees:
    for path, mu_var in jax.tree_util.tree_leaves_with_path(mu_tree, is_leaf=lambda x: isinstance(x, nnx.Variable)):
      if not isinstance(mu_var, nnx.Variable):
        continue
      mu_value = mu_var.get_value()
      if not isinstance(mu_value, NamedSharding):
        continue
      mu_lookup.setdefault(path, mu_value)

  def _update_model_var(path, var):
    if path in mu_lookup:
      return var.replace(mu_lookup[path])
    return var

  new_model_shardings = jax.tree_util.tree_map_with_path(
      _update_model_var, model_shardings, is_leaf=lambda x: isinstance(x, nnx.Variable)
  )
  updated_state = jax.tree_util.tree_map(lambda x: x, state_mesh_shardings, is_leaf=lambda x: isinstance(x, nnx.Variable))
  updated_state.model = new_model_shardings
  return prev_params_shardings, updated_state


def build_zero1_input_state_mesh_shardings(config, state_mesh_shardings, params_shardings):
  """Build train-step input shardings under Zero-1 optimizer sharding."""
  if not config.shard_optimizer_over_data:
    return state_mesh_shardings
  input_state = jax.tree_util.tree_map(
      lambda x: x, state_mesh_shardings, is_leaf=lambda x: isinstance(x, nnx.Variable)
  )

  def _overlay(model_node, params_node):
    for key, value in params_node.items():
      if isinstance(value, nnx.Variable):
        model_node[key] = value
      elif hasattr(value, "items"):
        _overlay(model_node[key], value)

  _overlay(input_state.model, params_shardings)
  return input_state


def reshard_train_state(state, state_mesh_shardings, enable_single_controller: bool):
  """Reshard train state, avoiding host staging in single-controller mode."""
  if enable_single_controller:
    return jax.jit(lambda s: s, out_shardings=state_mesh_shardings)(state)
  return jax.device_put(state, state_mesh_shardings)


def logical_axis_rules_pp_act_as_dp(logical_rules):
  """Add stage before data in rules where pipeline stages act as data parallelism."""
  new_rules = []
  for key, physical_axes in logical_rules:
    if isinstance(physical_axes, str):
      physical_axes = (physical_axes,)
    else:
      physical_axes = tuple(physical_axes)
    new_physical_axes = tuple(axis for axis in physical_axes if axis != "stage")
    if "data" in new_physical_axes:
      data_idx = new_physical_axes.index("data")
      new_physical_axes = new_physical_axes[0:data_idx] + ("stage",) + new_physical_axes[data_idx:]
    new_rules.append((key, new_physical_axes))
  return tuple(new_rules)


def get_formatted_sharding_annotations(params, mesh=None):
  """Generate a readable sharding report for each parameter."""
  annotation_lines = ["Comprehensice Weight Sharding Annotations:"]
  if mesh:
    annotation_lines.append(f"Mesh axes: {mesh.axis_names}, Mesh shape: {mesh.shape}")
    annotation_lines.append("-" * 30)
  for path, p_leaf in jax.tree_util.tree_leaves_with_path(params):
    param_name_str = jax.tree_util.keystr(path)
    shape_str = str(p_leaf.shape)
    sharding_desc = "N/A"
    if hasattr(p_leaf, "sharding"):
      if hasattr(p_leaf.sharding, "spec") and p_leaf.sharding.spec is not None:
        spec_parts = []
        for item in p_leaf.sharding.spec:
          spec_parts.append(str(item) if item is not None else "Replicated")
        sharding_desc = f"PartitionSpec({', '.join(spec_parts)})"
      elif hasattr(p_leaf.sharding, "spec") and p_leaf.sharding.spec is None:
        sharding_desc = "Fully Replicated (spec is None)"
      else:
        sharding_desc = str(p_leaf.sharding)
    else:
      sharding_desc = "No .sharding attribute found"
    annotation_lines.append(f" - Param: {param_name_str}\n" f"   Shape: {shape_str}\n" f"   Sharding: {sharding_desc}")
  return "\n".join(annotation_lines)


FSDP_MESH_AXES = ("fsdp", "fsdp_transpose")
FSDP_CO_AXES = FSDP_MESH_AXES + ("context", "context_usp_ulysses", "expert")


def fsdp_axes_for_dim(entry):
  """Return FSDP and following co-sharded axes from a spec entry."""
  axes = () if entry is None else (tuple(entry) if isinstance(entry, (tuple, list)) else (entry,))
  first_fsdp = next((i for i, axis in enumerate(axes) if axis in FSDP_MESH_AXES), None)
  if first_fsdp is None:
    return ()
  return tuple(axis for axis in axes[first_fsdp:] if axis in FSDP_CO_AXES)


def remove_mesh_axes_from_partition_spec(pspec, axes_to_remove, dims=None):
  """Return a spec with the requested mesh axes removed from selected dimensions."""
  axes_to_remove = set(axes_to_remove)
  new_spec = []
  for index, axis in enumerate(pspec):
    if axis is None or (dims is not None and index not in dims):
      new_spec.append(axis)
    elif isinstance(axis, str):
      new_spec.append(None if axis in axes_to_remove else axis)
    elif isinstance(axis, (list, tuple)):
      new_spec.append(tuple(item for item in axis if item not in axes_to_remove) or None)
    else:
      raise ValueError(f"Unsupported axis type: {type(axis)}")
  return P(*new_spec)


def remove_incompatible_mesh_axes_from_partition_spec(pspec, shape, mesh, dims=None, *, allow_remove_axes=False):
  """Replicate selected dimensions that cannot be evenly sharded when permitted."""
  if len(pspec) > len(shape):
    raise ValueError(f"PartitionSpec rank {len(pspec)} exceeds tensor rank {len(shape)}")
  if dims is None:
    dims_to_check = None
  else:
    rank = len(shape)
    dims_to_check = set()
    for dim in dims:
      normalized_dim = dim + rank if dim < 0 else dim
      if normalized_dim < 0 or normalized_dim >= rank:
        raise ValueError(f"Dimension index {dim} is out of bounds for tensor rank {rank}")
      dims_to_check.add(normalized_dim)
  compatible_pspec = pspec
  for dim, (dim_size, partition) in enumerate(zip(shape, pspec)):
    if (dims_to_check is not None and dim not in dims_to_check) or partition is None or partition == P.UNCONSTRAINED:
      continue
    if isinstance(partition, str):
      mesh_axes = (partition,)
    elif isinstance(partition, (list, tuple)):
      mesh_axes = tuple(partition)
    else:
      raise ValueError(f"Unsupported axis type: {type(partition)}")
    shard_count = 1
    for mesh_axis in mesh_axes:
      shard_count *= mesh.shape[mesh_axis]
    if dim_size % shard_count:
      if not allow_remove_axes:
        raise ValueError(
            f"Tensor dimension {dim} with size {dim_size} is not evenly divisible by "
            f"{shard_count} shards from mesh axes {mesh_axes}. Pass "
            "allow_remove_axes=True only when replicating this dimension is safe."
        )
      compatible_pspec = remove_mesh_axes_from_partition_spec(compatible_pspec, mesh_axes, dims=(dim,))
  return compatible_pspec


def remove_mesh_axes_from_sharding(sharding_tree, axes_to_remove):
  """Remove the requested axes from each named sharding in a tree."""
  def _peel(named_sharding):
    if isinstance(named_sharding, NamedSharding):
      return NamedSharding(named_sharding.mesh, remove_mesh_axes_from_partition_spec(named_sharding.spec, axes_to_remove))
    return named_sharding

  return jax.tree.map(_peel, sharding_tree)


def remove_fsdp_from_partition_spec(pspec):
  """Remove FSDP and subsequent co-sharded axes from each dimension."""
  for index, entry in enumerate(pspec):
    to_remove = fsdp_axes_for_dim(entry)
    if to_remove:
      pspec = remove_mesh_axes_from_partition_spec(pspec, to_remove, dims=(index,))
  return pspec


def remove_fsdp_from_sharding(sharding_tree):
  """Remove FSDP and co-sharded axes from a sharding tree."""
  def _peel(named_sharding):
    if isinstance(named_sharding, NamedSharding):
      return NamedSharding(named_sharding.mesh, remove_fsdp_from_partition_spec(named_sharding.spec))
    return named_sharding

  return jax.tree.map(_peel, sharding_tree)


def remove_expert_from_partition_spec(pspec, dims_to_peel):
  """Remove the expert mesh axis from selected dimensions."""
  return remove_mesh_axes_from_partition_spec(pspec, ("expert",), dims=dims_to_peel)


def get_physical_spec_without_axes(full_logical, mesh, axes_to_remove, logical_axis_rules=None):
  """Resolve logical shardings and remove the requested physical axes."""
  physical = logical_to_mesh_sharding(full_logical, mesh=mesh, rules=logical_axis_rules)
  return remove_mesh_axes_from_sharding(physical, axes_to_remove)


def get_physical_spec_no_fsdp(full_logical, mesh, logical_axis_rules):
  """Resolve logical shardings without FSDP mesh axes."""
  return get_physical_spec_without_axes(full_logical, mesh, FSDP_MESH_AXES, logical_axis_rules)


def all_gather_over_fsdp(variables, sharding_info, mesh, logical_axis_rules, shard_mode):
  """Gather FSDP-sharded variables using a sharding constraint."""
  physical_constraint_no_fsdp = get_physical_spec_no_fsdp(sharding_info, mesh, logical_axis_rules)
  return maybe_shard_with_name(variables, physical_constraint_no_fsdp, shard_mode=shard_mode)
