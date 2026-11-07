# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pipeline layer wrapping a decoder layer(s). Supports circular pipelining"""

import functools

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from flax import nnx


def get_mesh_axis_dim_indices(physical_partition_spec, axis_name="fsdp"):
  def find_axis_index(pspec):
    if pspec is None:
      return -1
    for i, axis in enumerate(pspec):
      if isinstance(axis, (list, tuple)):
        if axis_name in axis:
          return i
      elif axis == axis_name:
        return i
    return -1

  return jax.tree.map(find_axis_index, physical_partition_spec)


def derive_stage_weight_partition_specs(physical_partition_spec, axes_to_remove):
  def _process_pps(path, pps):
    path_keys = [getattr(p, "key", str(p)) for p in path]
    is_moe_block_0 = "MoeBlock_0" in path_keys
    processed_pps = remove_gathered_mesh_axes(pps, is_moe_block_0, axes_to_remove=axes_to_remove)
    return P(*processed_pps[1:])

  return jax.tree_util.tree_map_with_path(_process_pps, physical_partition_spec)


def remove_gathered_mesh_axes(pps, is_moe_block_0, axes_to_remove):
  axes_to_remove = list(axes_to_remove)
  if not is_moe_block_0:
    axes_to_remove.append("expert")

  if isinstance(pps, P):
    new_spec = []
    for axis in pps:
      if axis is None:
        new_spec.append(None)
      elif isinstance(axis, str):
        if axis not in axes_to_remove:
          new_spec.append(axis)
        else:
          new_spec.append(None)
      elif isinstance(axis, (list, tuple)):
        keep_context = is_moe_block_0 and "expert" in axis
        new_axis = [a for a in axis if a not in axes_to_remove or (keep_context and a == "context")]
        new_spec.append(tuple(new_axis))
      else:
        raise ValueError(f"Unsupported_axis_type: {type(axis)}")
    return P(*new_spec)

  return pps


def strip_pipeline_repeat_logical_axis(full_logical_spec):
  if full_logical_spec is None:
    return None

  def _remove_from_spec(spec):
    return jax.sharding.PartitionSpec(*[dim for dim in spec if dim != "circular_repeats"])

  return jax.tree.map(_remove_from_spec, full_logical_spec)


def create_gradient_accumulation_scan(
    model,
    length,
    deterministic=True,
    model_mode=None,
    logical_partition_spec=None,
):
  @jax.custom_vjp
  def run_single_microbatch_custom(lightweight_state, bsw, pos_arg, seg_arg):
    return run_single_microbatch_custom_fwd(lightweight_state, bsw, pos_arg, seg_arg)[0]

  def run_single_microbatch_custom_fwd(lightweight_state, bsw, pos_arg, seg_arg):
    def _run(l, b):
      out = model.run_one_iteration(
          l, b, pos_arg, seg_arg, deterministic, model_mode, logical_partition_spec=logical_partition_spec
      )
      return out

    _run_remat = jax.remat(_run, policy=model.get_pipeline_remat_policy())
    out, vjp_fun = jax.vjp(_run_remat, lightweight_state, bsw)
    return out, vjp_fun

  def run_single_microbatch_custom_bwd(res, g_out):
    vjp_fun = res
    d_l, d_b = vjp_fun(g_out)
    return d_l, d_b, None, None

  run_single_microbatch_custom.defvjp(run_single_microbatch_custom_fwd, run_single_microbatch_custom_bwd)

  @jax.custom_vjp
  def run_pipeline_microbatches_custom(loop_state, bsw, positions, segment_ids):
    return run_pipeline_microbatches_custom_fwd(loop_state, bsw, positions, segment_ids)[0]

  def run_pipeline_microbatches_custom_fwd(loop_state, bsw, positions, segment_ids):
    final_lightweight, scan_vjp_fun = jax.vjp(
        lambda l, b: jax.lax.scan(
            lambda carry, _: (run_single_microbatch_custom(carry, b, positions, segment_ids), None),
            l,
            None,
            length=length,
        )[0],
        loop_state,
        bsw,
    )
    return (final_lightweight, bsw), scan_vjp_fun

  def run_pipeline_microbatches_custom_bwd(residuals, g_final_state):
    scan_vjp_fun = residuals
    g_lightweight, g_bsw = g_final_state
    d_init_lightweight, d_init_bsw = scan_vjp_fun(g_lightweight)
    d_init_bsw = jax.tree.map(lambda d, g: d + g if hasattr(d, "shape") else d, d_init_bsw, g_bsw)
    return (d_init_lightweight, d_init_bsw, None, None)

  run_pipeline_microbatches_custom.defvjp(run_pipeline_microbatches_custom_fwd, run_pipeline_microbatches_custom_bwd)
  return run_pipeline_microbatches_custom


def is_static_param(path, v):
  return isinstance(v, nnx.Param) or type(v).__name__ in ("_overwrite_with_gradient", "MoEBiasVar")


def advance_rng_state(state, iteration):
  def _fold_if_rng(x):
    if isinstance(x, nnx.Variable) and issubclass(x.type, nnx.RngState):
      val = x.value
      if jax.dtypes.issubdtype(val.dtype, jax.dtypes.prng_key):
        def folded(k):
          return jax.random.fold_in(k, iteration)

        for _ in range(val.ndim):
          folded = jax.vmap(folded)
        return x.replace(value=folded(val))
    return x

  return jax.tree.map(_fold_if_rng, state, is_leaf=lambda x: isinstance(x, nnx.Variable))


def is_spec_leaf(x):
  return isinstance(x, P) or x is None


def flatten_nnx_state(state):
  def _is_var(x):
    return isinstance(x, nnx.Variable)

  leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(state, is_leaf=_is_var)
  arrays = []
  is_var_flags = []
  var_types = []
  var_metadata = []
  for _, leaf in leaves_with_path:
    if isinstance(leaf, nnx.Variable):
      arrays.append(leaf.value)
      is_var_flags.append(True)
      var_types.append(type(leaf))
      var_metadata.append(dict(leaf._var_metadata))
    else:
      arrays.append(leaf)
      is_var_flags.append(False)
      var_types.append(None)
      var_metadata.append({})
  return arrays, treedef, is_var_flags, var_types, var_metadata


def unflatten_nnx_state(arrays, treedef, is_var_flags, var_types, var_metadata):
  new_leaves = []
  for arr, is_var, vtype, meta in zip(arrays, is_var_flags, var_types, var_metadata):
    if is_var and vtype is not None:
      new_leaves.append(vtype(arr, **meta))
    else:
      new_leaves.append(arr)
  return treedef.unflatten(new_leaves)


SOWN_MICROBATCH_REDUCTIONS = {
    "moe_has_overflow": jnp.any,
    "moe_lb_loss": jnp.mean,
    "moe_bias_updates": jnp.sum,
    "max_logits": jnp.max,
    "te_moe_total_recv_tokens": jnp.max,
    "te_moe_recv_capacity_per_rank": jnp.min,
}
UNSUPPORTED_SOWN_METRICS = ("out_projection_activations",)


def _sown_leaf_names(treedef):
  template = jax.tree_util.tree_unflatten(treedef, list(range(treedef.num_leaves)))
  names = []
  for path, _ in jax.tree_util.tree_flatten_with_path(template)[0]:
    keys = [str(getattr(p, "key", getattr(p, "name", ""))) for p in path]
    keys = [k for k in keys if k and not k.isdigit()]
    names.append(keys[-1] if keys else "")
  return names


def reduce_sown_metrics_per_layer(
    stacked_arrays, treedef, num_repeats, num_microbatches, num_stages, forwarding_delay, squeeze_repeats=False
):
  valid_iteration = (
      jnp.arange(num_repeats)[:, None, None] * num_microbatches
      + jnp.arange(num_microbatches)[None, :, None]
      + forwarding_delay * jnp.arange(num_stages)[None, None, :]
  )
  stage_index = jnp.arange(num_stages)[None, None, :]

  def _reduce(name, stacked_leaf):
    if name in UNSUPPORTED_SOWN_METRICS:
      raise ValueError(f"Sown value '{name}' cannot be reduced over pipeline microbatches; unsupported with PP.")
    per_microbatch = stacked_leaf[valid_iteration, stage_index]
    reduction = SOWN_MICROBATCH_REDUCTIONS.get(name)
    if reduction is None:
      if per_microbatch.dtype == jnp.bool_:
        reduction = jnp.any
      elif jnp.issubdtype(per_microbatch.dtype, jnp.inexact):
        reduction = jnp.mean
      else:
        reduction = jnp.sum
    per_layer = reduction(per_microbatch, axis=1)
    return per_layer[0] if squeeze_repeats else per_layer

  return [
      jax.tree.map(functools.partial(_reduce, name), leaf)
      for name, leaf in zip(_sown_leaf_names(treedef), stacked_arrays)
  ]


def arrays_to_linen_collection(arrays, keys):
  return dict(zip(keys, arrays))


def linen_collection_to_arrays(collection, keys):
  return [collection[k] for k in keys]


class PipelineContext:
  __slots__ = (
      "weight_prefetching",
      "run_one_iteration",
      "from_all_variables_to_repeat_weights",
      "from_repeat_weights_to_bsw",
  )

  def __init__(self, pipeline_module):
    self.weight_prefetching = pipeline_module.weight_prefetching
    self.run_one_iteration = pipeline_module.run_one_iteration
    self.from_all_variables_to_repeat_weights = pipeline_module.from_all_variables_to_repeat_weights
    self.from_repeat_weights_to_bsw = pipeline_module.from_repeat_weights_to_bsw
