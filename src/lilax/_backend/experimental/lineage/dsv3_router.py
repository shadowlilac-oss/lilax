"""DeepSeek MoE router implementation."""

from collections.abc import Callable, Mapping
import dataclasses
import functools
import math
from typing import Any, Generic, TypeVar

import jax
import jax.experimental.compute_on
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from lilax._backend.experimental.lineage import dsv3_expert_shuffle
from lilax._backend.experimental.lineage import dsv3_types
from lilax._backend.experimental.lineage import ragged_gather
from lilax._backend.experimental.lineage import ragged_gather_reduce
from lilax._backend.experimental.lineage import ragged_gather_reduce_metadata
from lilax._backend.experimental.lineage import token_absmax
from lilax._backend.experimental.lineage.sc_reduce_scatter import reduce_scatter
from lilax._backend.experimental.lineage.sharding import axis_collectives
from lilax._backend.experimental.lineage.sharding import partition_specs

compute_on = jax.experimental.compute_on.compute_on
ArrayType = jax.Array
ShardingType = jax.sharding.PartitionSpec
T = TypeVar("T", ArrayType, ShardingType, Any)
_CAPACITY_ALIGN = 256
_SC_ALL_GATHER_MIN_BYTES = 500_000
MULTI_AXIS_TC_FALLBACK_TOKEN_THRESHOLD: int = 4096


def _get_axis_size(axis_name: str | tuple[str, ...]) -> int:
  if isinstance(axis_name, str):
    return jax.lax.axis_size(axis_name)
  return math.prod(jax.lax.axis_size(name) for name in axis_name)


def _get_axis_index(axis_name: str | tuple[str, ...]) -> jax.Array:
  if isinstance(axis_name, str):
    return jax.lax.axis_index(axis_name)
  idx = jnp.array(0, dtype=jnp.int32)
  for name in axis_name:
    idx = idx * jax.lax.axis_size(name) + jax.lax.axis_index(name)
  return idx


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_get_token_axis(x_pspec: jax.sharding.PartitionSpec, num_model_dims: int = 1) -> str | tuple[str, ...] | None:
  token_axes = []
  for ax in x_pspec[:-num_model_dims] if num_model_dims > 0 else x_pspec:
    if ax is not None:
      if isinstance(ax, tuple):
        token_axes.extend(ax)
      else:
        token_axes.append(ax)
  if not token_axes:
    return None
  if len(token_axes) == 1:
    return token_axes[0]
  return tuple(token_axes)


def _should_tc_fallback(axis_name: str | tuple[str, ...], num_tokens: int) -> bool:
  if not (isinstance(axis_name, (tuple, list)) and len(axis_name) > 1):
    return False
  return num_tokens >= MULTI_AXIS_TC_FALLBACK_TOKEN_THRESHOLD


def _all_gather_sc0(*operands: jax.Array, axis_name: str | tuple[str, ...]) -> tuple[jax.Array, ...]:
  axis_size = _get_axis_size(axis_name)
  if any(x.size * x.dtype.itemsize * axis_size < _SC_ALL_GATHER_MIN_BYTES for x in operands):
    return tuple(jax.lax.all_gather(x, axis_name=axis_name, axis=0, tiled=True) for x in operands)

  @compute_on(compute_type="tpu_sparsecore", out_memory_spaces=jax.memory.Space.Device,
              compiler_options={"sparse_core_config": {"core_ids": [0]}})
  def _ag(*xs: jax.Array) -> tuple[jax.Array, ...]:
    return tuple(jax.lax.all_gather(x, axis_name=axis_name, axis=0, tiled=True) for x in xs)
  return _ag(*operands)


def _psum_scatter_sc0(*operands: jax.Array, axis_name: str | tuple[str, ...]) -> tuple[jax.Array, ...]:
  @compute_on(compute_type="tpu_sparsecore", out_memory_spaces=jax.memory.Space.Device,
              compiler_options={"sparse_core_config": {"core_ids": [0]}})
  def _rs(*xs: jax.Array) -> tuple[jax.Array, ...]:
    return tuple(jax.lax.psum_scatter(x, axis_name=axis_name, scatter_dimension=0, tiled=True) for x in xs)
  return _rs(*operands)


def _quantized_reduce_scatter_sc0(
    y_ag: jax.Array, *, axis_name: str | tuple[str, ...], qtype: Any, per_hop_scaling: bool = False,
) -> jax.Array:
  if not (isinstance(axis_name, (tuple, list)) and len(axis_name) == 2):
    raise ValueError(
        "The quantized combine reduce-scatter needs a (ring_axis, core_axis)" f" expert axis mapping, got {axis_name!r}."
    )
  if qtype != jnp.float8_e4m3fn:
    raise ValueError(f"ops.reduce_scatter_sc0 only sends jnp.float8_e4m3fn, got {qtype!r}.")
  ring_axis, core_axis = axis_name
  token_amax = None if per_hop_scaling else token_absmax.token_absmax(y_ag)
  return reduce_scatter.reduce_scatter_sc0(
      y_ag, token_amax=token_amax, per_hop_scaling=per_hop_scaling, ring_axis=ring_axis, core_axis=core_axis,
  )


def _psum_scatter_and_reduce_sc0(
    *operands: jax.Array, axis_name: str | tuple[str, ...], reduce_fn: Callable[[Any], Any], reduce_operand: Any,
) -> tuple[tuple[jax.Array, ...], Any]:
  @compute_on(compute_type="tpu_sparsecore", out_memory_spaces=jax.memory.Space.Device,
              compiler_options={"sparse_core_config": {"core_ids": [0]}})
  def _rs_then_reduce(xs: tuple[jax.Array, ...], reduce_operand: Any) -> tuple[tuple[jax.Array, ...], Any]:
    scattered = tuple(jax.lax.psum_scatter(x, axis_name=axis_name, scatter_dimension=0, tiled=True) for x in xs)
    return scattered, reduce_fn(reduce_operand)
  return _rs_then_reduce(operands, reduce_operand)


def _all_gather_routing_metadata(x: jax.Array, *, axis_name: str | tuple[str, ...], num_tokens: int) -> jax.Array:
  if _should_tc_fallback(axis_name, num_tokens):
    return jax.lax.all_gather(x, axis_name=axis_name, axis=0, tiled=True)
  (x_ag,) = _all_gather_sc0(x, axis_name=axis_name)
  return x_ag


def _psum_scatter_routing_metadata(x: jax.Array, *, axis_name: str | tuple[str, ...], num_tokens: int) -> jax.Array:
  if _should_tc_fallback(axis_name, num_tokens):
    return jax.lax.psum_scatter(x, axis_name=axis_name, scatter_dimension=0, tiled=True)
  (x_rs,) = _psum_scatter_sc0(x, axis_name=axis_name)
  return x_rs


def _gather_sc0(x: jax.Array, indices: jax.Array) -> jax.Array:
  @compute_on(compute_type="tpu_sparsecore", out_memory_spaces=jax.memory.Space.Device,
              compiler_options={"sparse_core_config": {"core_ids": [0]}})
  def _gather(flat_x: jax.Array, flat_indices: jax.Array) -> jax.Array:
    return flat_x.at[flat_indices].get(mode="promise_in_bounds", wrap_negative_indices=False)
  out = _gather(jnp.ravel(x), jnp.ravel(indices))
  return jnp.reshape(out, indices.shape + x.shape[1:])


def _scatter_sc0(x: jax.Array, indices: jax.Array) -> jax.Array:
  @compute_on(compute_type="tpu_sparsecore", out_memory_spaces=jax.memory.Space.Device,
              compiler_options={"sparse_core_config": {"core_ids": [0]}})
  def _scatter(zeros: jax.Array, flat_x: jax.Array, flat_indices: jax.Array) -> jax.Array:
    return zeros.at[flat_indices].add(flat_x, mode="promise_in_bounds", wrap_negative_indices=False, unique_indices=True)
  sc_dtype = x.dtype if x.dtype in (jnp.float32, jnp.int32) else jnp.float32
  flat_x = jax.lax.bitcast_convert_type(jnp.ravel(x).astype(sc_dtype), jnp.int32)
  zeros = jnp.zeros_like(flat_x)
  out = _scatter(zeros, flat_x, jnp.ravel(indices))
  out = jax.lax.bitcast_convert_type(out, sc_dtype).astype(x.dtype)
  return jnp.reshape(out, x.shape)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RouterMetadata(Generic[T]):
  sort_indices: T
  group_sizes: T
  sorted_coeffs: T
  reduce_metadata: ragged_gather_reduce_metadata.RaggedGatherReduceMetadata | T
  selected_experts: T
  probs: T
  capacity: int = jax.tree.static()


def chunk_num_tokens(num_local_tokens: jt.Int[jax.Array, ""], start: jt.Int[jax.Array, ""] | int, capacity: int) -> jt.Int[jax.Array, ""]:
  remaining = num_local_tokens - start
  return jnp.where(remaining >= capacity, capacity, remaining)


def chunk_group_sizes(
    group_sizes: jt.Int[jax.Array, "E_local"], start: jt.Int[jax.Array, ""] | int, num_tokens: jt.Int[jax.Array, ""],
) -> jt.Int[jax.Array, "E_local"]:
  ends = jnp.cumsum(group_sizes)
  begins = ends - group_sizes
  stop = start + num_tokens
  return (jnp.clip(ends, start, stop) - jnp.clip(begins, start, stop)).astype(group_sizes.dtype)


def chunk_reduce_metadata(
    sort_indices: jt.Int[jax.Array, "kNBT"], start: jt.Int[jax.Array, ""] | int,
    num_tokens: jt.Int[jax.Array, ""], *, capacity: int, num_experts_per_tok: int,
) -> ragged_gather_reduce_metadata.RaggedGatherReduceMetadata:
  return ragged_gather_reduce_metadata.ragged_gather_reduce_tc_metadata(
      sort_indices // num_experts_per_tok, start=start, num_tokens=num_tokens,
      top_k=num_experts_per_tok, max_num_tokens=capacity,
  )


def permute_chunk(
    x_ag: jt.Num[jax.Array, "NBT D0 D1"], sort_indices: jt.Int[jax.Array, "kNBT"],
    start: jt.Int[jax.Array, ""] | int, num_tokens: jt.Int[jax.Array, ""],
    *, capacity: int, num_experts_per_tok: int, with_absmax: bool = False,
) -> Any:
  return ragged_gather.ragged_gather_tc(
      x_ag, sort_indices // num_experts_per_tok, start=start, num_tokens=num_tokens,
      max_out_tokens=capacity, with_absmax=with_absmax,
  )


def unpermute_chunk(
    x: jt.Num[jax.Array, "C D0 D1"], reduce_metadata: ragged_gather_reduce_metadata.RaggedGatherReduceMetadata,
    *, num_out_tokens: int, num_experts_per_tok: int, out: jt.Num[jax.Array, "NBT D0 D1"] | None = None,
    zero_initialized: bool | None = None,
) -> jt.Num[jax.Array, "NBT D0 D1"]:
  return ragged_gather_reduce.ragged_gather_reduce_tc(
      x, reduce_metadata, num_out_tokens=num_out_tokens, top_k=num_experts_per_tok,
      out=out, zero_initialized=zero_initialized,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _sort_coeffs(coeffs, sort_indices, group_sizes, num_experts_per_tok, axis_name):
  return _sort_coeffs_fwd(coeffs, sort_indices, group_sizes, num_experts_per_tok, axis_name)[0]


def _sort_coeffs_fwd(coeffs, sort_indices, group_sizes, num_experts_per_tok, axis_name):
  del num_experts_per_tok
  coeffs_ag = _all_gather_routing_metadata(coeffs, axis_name=axis_name, num_tokens=coeffs.shape[0])
  sorted_coeffs = _gather_sc0(jnp.reshape(coeffs_ag, (-1, 1)), sort_indices)
  return sorted_coeffs, (sort_indices, group_sizes)


def _sort_coeffs_bwd(num_experts_per_tok, axis_name, res, grad_sorted_coeffs):
  sort_indices, group_sizes = res
  num_local_tokens = jnp.sum(group_sizes, dtype=jnp.int32)
  valid = jnp.arange(grad_sorted_coeffs.shape[0]) < num_local_tokens
  masked_grad = jnp.where(valid[:, None], grad_sorted_coeffs, 0)
  unsorted = _scatter_sc0(masked_grad, sort_indices)
  grad_coeffs_ag = jnp.reshape(unsorted, (-1, num_experts_per_tok))
  num_tokens = grad_coeffs_ag.shape[0] // _get_axis_size(axis_name)
  grad_coeffs = _psum_scatter_routing_metadata(grad_coeffs_ag, axis_name=axis_name, num_tokens=num_tokens)
  return grad_coeffs, None, None


_sort_coeffs.defvjp(_sort_coeffs_fwd, _sort_coeffs_bwd)


def _cumsum_small_ints(x: jt.Shaped[jax.Array, "*B n"]) -> jt.Int[jax.Array, "*B n"]:
  n = x.shape[-1]
  upper = jnp.arange(n)[:, None] <= jnp.arange(n)[None, :]
  return jnp.matmul(x.astype(jnp.bfloat16), upper.astype(jnp.bfloat16), preferred_element_type=jnp.float32).astype(jnp.int32)


def _counting_argsort(keys: jt.Int[jax.Array, "N"], num_keys: int) -> jt.Int[jax.Array, "N"]:
  (n,) = keys.shape
  row = 128
  rows = jnp.reshape(jnp.pad(keys, (0, -n % (row * row)), constant_values=num_keys - 1), (-1, row))
  one_hot = rows[None] == jnp.arange(num_keys, dtype=keys.dtype)[:, None, None]
  rank_in_row = _cumsum_small_ints(one_hot)
  row_counts = jnp.reshape(rank_in_row[..., -1], (num_keys, -1, row))
  rows_in_block = _cumsum_small_ints(row_counts)
  block_counts = rows_in_block[..., -1]
  key_counts = jnp.sum(block_counts, axis=1)
  row_offsets = jnp.reshape(
      (jnp.cumsum(key_counts) - key_counts)[:, None, None]
      + (jnp.cumsum(block_counts, axis=1) - block_counts)[..., None] + rows_in_block - row_counts,
      (num_keys, -1),
  )
  dest = jnp.sum(jnp.where(one_hot, rank_in_row + row_offsets[..., None], 0), 0) - 1
  dest = jnp.ravel(dest)[:n]
  return _scatter_sc0(jnp.arange(n, dtype=jnp.int32), dest)


def _routing_metadata_impl(
    x: jt.Num[jax.Array, "B T D"], w: dsv3_types.DSv3MoERouterWeightsPytree,
    expert_permutations: jt.Int[jax.Array, "E"] | None = None, *, num_experts: int, num_experts_per_tok: int,
    routed_scaling_factor: float, n_routing_groups: int, topk_routing_group: int, topk_in_group: int,
    expert_axis_name: str, axis_mapping: Mapping[str, str | tuple[str, ...]], capacity: int,
    router_dtype: jax.typing.DTypeLike | None = None,
) -> tuple[RouterMetadata[jax.Array], jt.Num[jax.Array, "BT k"], dsv3_types.DSv3RouterAux[jax.Array]]:
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  x_flat = jnp.reshape(x, (-1, x.shape[-1]))
  selected_experts, coeffs, local_group_sizes, logits, probs = _expert_selection_impl(
      x_flat, w, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok,
      routed_scaling_factor=routed_scaling_factor, n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group, topk_in_group=topk_in_group, router_dtype=router_dtype,
  )
  if expert_permutations is not None:
    physical_selected_experts = dsv3_expert_shuffle.remap_selected_experts(selected_experts, expert_permutations)
    physical_group_sizes = dsv3_expert_shuffle.remap_group_sizes(local_group_sizes, expert_permutations)
  else:
    physical_selected_experts = selected_experts
    physical_group_sizes = local_group_sizes
  num_devices = _get_axis_size(physical_expert_axis_name)
  local_num_experts = num_experts // num_devices
  device_index = _get_axis_index(physical_expert_axis_name)
  selected_experts_flat = jnp.ravel(physical_selected_experts)
  all_selected_experts = _all_gather_routing_metadata(
      selected_experts_flat, axis_name=physical_expert_axis_name, num_tokens=physical_selected_experts.shape[0],
  )
  local_expert = all_selected_experts - device_index * local_num_experts
  is_local = (local_expert >= 0) & (local_expert < local_num_experts)
  sort_key = jnp.where(is_local, local_expert, local_num_experts)
  sort_indices = _counting_argsort(sort_key, local_num_experts + 1)
  all_group_sizes = jax.lax.all_gather(physical_group_sizes, axis_name=physical_expert_axis_name, axis=0)
  all_send_sizes_3d = jnp.reshape(all_group_sizes, (num_devices, num_devices, local_num_experts))
  group_sizes = jnp.sum(all_send_sizes_3d, axis=0)[device_index]
  sorted_coeffs = _sort_coeffs(coeffs, sort_indices, group_sizes, num_experts_per_tok, physical_expert_axis_name)
  reduce_metadata = chunk_reduce_metadata(
      sort_indices, 0, chunk_num_tokens(jnp.sum(group_sizes, dtype=jnp.int32), 0, capacity),
      capacity=capacity, num_experts_per_tok=num_experts_per_tok,
  )
  metadata = RouterMetadata[jax.Array](
      sort_indices=sort_indices, group_sizes=group_sizes, sorted_coeffs=sorted_coeffs,
      reduce_metadata=reduce_metadata, selected_experts=selected_experts, probs=jax.lax.stop_gradient(probs),
      capacity=capacity,
  )
  selected_experts = jnp.reshape(selected_experts, x.shape[:-1] + (num_experts_per_tok,))
  logits = jnp.reshape(logits, x.shape[:-1] + (num_experts,))
  aux = dsv3_types.DSv3RouterAux[jax.Array](
      group_sizes=local_group_sizes[None, :], selected_experts=selected_experts, logits=logits,
  )
  return metadata, coeffs, aux


def chunk_capacity(num_local_slots: int, num_expert_shards: int, capacity_factor: float) -> int:
  return min(num_local_slots * num_expert_shards, math.ceil(capacity_factor * num_local_slots / _CAPACITY_ALIGN) * _CAPACITY_ALIGN)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routing_metadata(
    x: jt.Num[jax.Array, "B T D"], w: dsv3_types.DSv3MoERouterWeightsPytree,
    *, num_experts: int, num_experts_per_tok: int, routed_scaling_factor: float, n_routing_groups: int,
    topk_routing_group: int, topk_in_group: int, expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]], mesh: jax.sharding.Mesh, capacity_factor: float,
    router_dtype: jax.typing.DTypeLike | None = None, expert_permutations: jt.Int[jax.Array, "E"] | None = None,
) -> tuple[RouterMetadata[jax.Array], jt.Num[jax.Array, "BT k"], dsv3_types.DSv3RouterAux[jax.Array]]:
  x_pspec = jax.typeof(x).sharding.spec
  token_axis = dsv3_get_token_axis(x_pspec)
  coeffs_spec = jax.sharding.PartitionSpec(token_axis, None)
  token_spec_1d = jax.sharding.PartitionSpec(token_axis)
  token_pspec = x_pspec[:-1]
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)

  def _num_shards(axis: str | tuple[str, ...] | None) -> int:
    if axis is None:
      return 1
    if isinstance(axis, str):
      return mesh.shape[axis]
    return math.prod(mesh.shape[a] for a in axis)

  num_local_slots = math.prod(x.shape[:-1]) // _num_shards(token_axis) * num_experts_per_tok
  capacity = chunk_capacity(num_local_slots, _num_shards(physical_expert_axis_name), capacity_factor)
  metadata_out_spec = RouterMetadata[jax.sharding.PartitionSpec](
      sort_indices=token_spec_1d, group_sizes=jax.sharding.PartitionSpec(physical_expert_axis_name),
      sorted_coeffs=coeffs_spec, reduce_metadata=token_spec_1d, selected_experts=coeffs_spec,
      probs=coeffs_spec, capacity=capacity,
  )
  aux_out_spec = dsv3_types.DSv3RouterAux[jax.sharding.PartitionSpec](
      group_sizes=jax.sharding.PartitionSpec(token_axis, None),
      selected_experts=jax.sharding.PartitionSpec(*token_pspec, None),
      logits=jax.sharding.PartitionSpec(*token_pspec, None),
  )
  if expert_permutations is not None:
    expert_permutations = jax.reshard(expert_permutations, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()))
  return jax.shard_map(
      functools.partial(
          _routing_metadata_impl, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor, n_routing_groups=n_routing_groups,
          topk_routing_group=topk_routing_group, topk_in_group=topk_in_group, expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping, capacity=capacity, router_dtype=router_dtype,
      ), mesh=mesh, out_specs=(metadata_out_spec, coeffs_spec, aux_out_spec), check_vma=False,
  )(x, w, expert_permutations)


@jax.custom_vjp
def _saved_router_probs(x: jt.Num[jax.Array, "BT D"], kernel: jt.Num[jax.Array, "D E"], probs: jt.Num[jax.Array, "BT E"]) -> jt.Num[jax.Array, "BT E"]:
  del x, kernel
  return probs


def _saved_router_probs_fwd(x, kernel, probs):
  return probs, (x, kernel, probs)


def _saved_router_probs_bwd(res, grad_probs):
  x, kernel, probs = res
  input_dtype = jnp.result_type(x, kernel)
  probs_f32 = probs.astype(jnp.float32)
  grad_z = (grad_probs.astype(jnp.float32) * probs_f32 * (1.0 - probs_f32)).astype(input_dtype)
  grad_x = jnp.dot(grad_z, kernel.astype(input_dtype).T, preferred_element_type=jnp.float32).astype(x.dtype)
  grad_kernel = jnp.dot(x.astype(input_dtype).T, grad_z, preferred_element_type=jnp.float32).astype(kernel.dtype)
  return grad_x, grad_kernel, None


_saved_router_probs.defvjp(_saved_router_probs_fwd, _saved_router_probs_bwd)


def _routing_replay_impl(
    x: jt.Num[jax.Array, "B T D"], w: dsv3_types.DSv3MoERouterWeightsPytree,
    selected_experts: jt.Int[jax.Array, "BT k"], probs: jt.Num[jax.Array, "BT E"],
    sort_indices: jt.Int[jax.Array, "kNBT"], group_sizes: jt.Int[jax.Array, "E_local"],
    *, num_experts_per_tok: int, routed_scaling_factor: float, expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> tuple[jt.Num[jax.Array, "kNBT 1"], jt.Num[jax.Array, "B T E"]]:
  assert w.kernel is not None
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  x_flat = jnp.reshape(x, (-1, x.shape[-1]))
  pre_bias_logits = _saved_router_probs(x_flat, w.kernel, probs)
  coeffs = selected_coeffs(pre_bias_logits, selected_experts, routed_scaling_factor=routed_scaling_factor).astype(x.dtype)
  sorted_coeffs = _sort_coeffs(coeffs, sort_indices, group_sizes, num_experts_per_tok, physical_expert_axis_name)
  logits = jnp.reshape(pre_bias_logits.astype(x.dtype), x.shape[:-1] + (-1,))
  return sorted_coeffs, logits


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routing_replay(
    x: jt.Num[jax.Array, "B T D"], w: dsv3_types.DSv3MoERouterWeightsPytree, metadata: RouterMetadata[jax.Array],
    *, num_experts_per_tok: int, routed_scaling_factor: float, expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]], mesh: jax.sharding.Mesh,
) -> tuple[jt.Num[jax.Array, "kNBT 1"], jt.Num[jax.Array, "B T E"]]:
  x_pspec = jax.typeof(x).sharding.spec
  token_axis = dsv3_get_token_axis(x_pspec)
  out_specs = (jax.sharding.PartitionSpec(token_axis, None), jax.sharding.PartitionSpec(*x_pspec[:-1], None))
  return jax.shard_map(
      functools.partial(
          _routing_replay_impl, num_experts_per_tok=num_experts_per_tok, routed_scaling_factor=routed_scaling_factor,
          expert_axis_name=expert_axis_name, axis_mapping=axis_mapping,
      ), mesh=mesh, out_specs=out_specs, check_vma=False,
  )(x, w, metadata.selected_experts, metadata.probs, metadata.sort_indices, metadata.group_sizes)


def _unsort_coeff_grads_impl(grad_sorted_coeffs, sort_indices, group_sizes, *, num_shards):
  num_local_tokens = jnp.sum(group_sizes, dtype=jnp.int32)
  valid = jnp.arange(grad_sorted_coeffs.shape[0]) < num_local_tokens
  masked = jnp.where(valid, grad_sorted_coeffs, 0).astype(grad_sorted_coeffs.dtype)
  unsorted = _scatter_sc0(masked, sort_indices)
  num_slots = unsorted.shape[0]
  if num_slots % (num_shards * 128) == 0:
    return jnp.reshape(unsorted, (num_slots // 128, 128))
  return jnp.reshape(unsorted, (num_shards, num_slots // num_shards))


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_unsort_coeff_grads(
    grad_sorted_coeffs: jt.Num[jax.Array, "kNBT"], metadata: RouterMetadata[jax.Array],
    *, expert_axis_name: str, axis_mapping: Mapping[str, str | tuple[str, ...]], mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "R C"]:
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  token_axis = jax.typeof(grad_sorted_coeffs).sharding.spec[0]

  def _impl(g, sort_indices, group_sizes):
    return _unsort_coeff_grads_impl(g, sort_indices, group_sizes, num_shards=_get_axis_size(physical_expert_axis_name))

  return jax.shard_map(_impl, mesh=mesh, out_specs=jax.sharding.PartitionSpec(token_axis, None), check_vma=False)(
      grad_sorted_coeffs, metadata.sort_indices, metadata.group_sizes,
  )


def _reshape_combined_grads(grad_x_local, grad_coeffs_local, *, local_seq_length, num_experts_per_tok):
  grad_x = jnp.reshape(grad_x_local, (-1, local_seq_length, grad_x_local.shape[-2] * grad_x_local.shape[-1]))
  return grad_x, jnp.reshape(grad_coeffs_local, (-1, num_experts_per_tok))


def _combine_with_coeff_grads_impl(
    grad_x_ag, grad_coeffs_ag, *, local_seq_length, num_experts_per_tok, expert_axis_name, axis_mapping,
):
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  grad_x_local, grad_coeffs_local = _psum_scatter_sc0(grad_x_ag, grad_coeffs_ag, axis_name=physical_expert_axis_name)
  return _reshape_combined_grads(
      grad_x_local, grad_coeffs_local, local_seq_length=local_seq_length, num_experts_per_tok=num_experts_per_tok,
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_combine_with_coeff_grads(
    grad_x_ag: jt.Num[jax.Array, "NBT D0 D1"], grad_coeffs_ag: jt.Num[jax.Array, "R C"],
    metadata: RouterMetadata[jax.Array], *, local_seq_length: int, num_experts_per_tok: int,
    expert_axis_name: str, axis_mapping: Mapping[str, str | tuple[str, ...]], mesh: jax.sharding.Mesh,
    out_specs: jax.sharding.PartitionSpec,
) -> tuple[jt.Num[jax.Array, "B T D"], jt.Num[jax.Array, "BT k"]]:
  out_specs_phys = partition_specs.physical_pspec(out_specs, axis_mapping)
  coeffs_spec = jax.typeof(metadata.selected_experts).sharding.spec
  return jax.shard_map(
      functools.partial(
          _combine_with_coeff_grads_impl, local_seq_length=local_seq_length, num_experts_per_tok=num_experts_per_tok,
          expert_axis_name=expert_axis_name, axis_mapping=axis_mapping,
      ), mesh=mesh, out_specs=(out_specs_phys, coeffs_spec), check_vma=False,
  )(grad_x_ag, grad_coeffs_ag)


def _combine_with_coeff_grads_and_reduce_impl(
    grad_x_ag, grad_coeffs_ag, reduce_operand, *, local_seq_length, num_experts_per_tok,
    expert_axis_name, axis_mapping, reduce_fn,
):
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  (grad_x_local, grad_coeffs_local), reduced = _psum_scatter_and_reduce_sc0(
      grad_x_ag, grad_coeffs_ag, axis_name=physical_expert_axis_name, reduce_fn=reduce_fn, reduce_operand=reduce_operand,
  )
  grad_x, grad_coeffs = _reshape_combined_grads(
      grad_x_local, grad_coeffs_local, local_seq_length=local_seq_length, num_experts_per_tok=num_experts_per_tok,
  )
  return grad_x, grad_coeffs, reduced


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_combine_with_coeff_grads_and_reduce(
    grad_x_ag: jt.Num[jax.Array, "NBT D0 D1"], grad_coeffs_ag: jt.Num[jax.Array, "R C"],
    metadata: RouterMetadata[jax.Array], deferred_reduce: axis_collectives.DeferredReduce | None,
    *, local_seq_length: int, num_experts_per_tok: int, expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]], mesh: jax.sharding.Mesh, out_specs: jax.sharding.PartitionSpec,
) -> tuple[jt.Num[jax.Array, "B T D"], jt.Num[jax.Array, "BT k"], Any]:
  if deferred_reduce is None:
    grad_x, grad_coeffs = dsv3_combine_with_coeff_grads(
        grad_x_ag, grad_coeffs_ag, metadata, local_seq_length=local_seq_length,
        num_experts_per_tok=num_experts_per_tok, expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping, mesh=mesh, out_specs=out_specs,
    )
    return grad_x, grad_coeffs, None
  out_specs_phys = partition_specs.physical_pspec(out_specs, axis_mapping)
  coeffs_spec = jax.typeof(metadata.selected_experts).sharding.spec
  grad_x, grad_coeffs, reduced = jax.shard_map(
      functools.partial(
          _combine_with_coeff_grads_and_reduce_impl, local_seq_length=local_seq_length,
          num_experts_per_tok=num_experts_per_tok, expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping, reduce_fn=deferred_reduce.local_fn,
      ), mesh=mesh, out_specs=(out_specs_phys, coeffs_spec, deferred_reduce.out_specs), check_vma=False,
  )(grad_x_ag, grad_coeffs_ag, deferred_reduce.operand)
  return grad_x, grad_coeffs, deferred_reduce.finish(reduced)


def _dense_selected_coeffs(pre_bias_logits, indices, *, routed_scaling_factor):
  num_experts = pre_bias_logits.shape[-1]
  one_hot = indices[..., None] == jnp.arange(num_experts, dtype=indices.dtype)
  coeffs = jnp.sum(jnp.where(one_hot, pre_bias_logits[:, None, :], 0), axis=-1, dtype=pre_bias_logits.dtype)
  return _normalize_coeffs(coeffs, routed_scaling_factor=routed_scaling_factor)


def _local_routing_impl(x, w, selected_experts, probs, *, routed_scaling_factor):
  assert w.kernel is not None
  x_flat = jnp.reshape(x, (-1, x.shape[-1]))
  pre_bias_logits = _saved_router_probs(x_flat, w.kernel, probs)
  coeffs = _dense_selected_coeffs(pre_bias_logits, selected_experts, routed_scaling_factor=routed_scaling_factor).astype(x.dtype)
  logits = jnp.reshape(pre_bias_logits.astype(x.dtype), x.shape[:-1] + (-1,))
  return coeffs, logits


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_local_routing(
    x: jt.Num[jax.Array, "B T D"], w: dsv3_types.DSv3MoERouterWeightsPytree,
    selected_experts: jt.Int[jax.Array, "BT k"], probs: jt.Num[jax.Array, "BT E"],
    *, routed_scaling_factor: float, mesh: jax.sharding.Mesh,
) -> tuple[jt.Num[jax.Array, "BT k"], jt.Num[jax.Array, "B T E"]]:
  x_pspec = jax.typeof(x).sharding.spec
  out_specs = (jax.typeof(selected_experts).sharding.spec, jax.sharding.PartitionSpec(*x_pspec[:-1], None))
  return jax.shard_map(
      functools.partial(_local_routing_impl, routed_scaling_factor=routed_scaling_factor),
      mesh=mesh, out_specs=out_specs, check_vma=False,
  )(x, w, selected_experts, probs)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routing_bwd(
    grad_coeffs: jt.Num[jax.Array, "BT k"], grad_logits: jt.Num[jax.Array, "B T E"] | None,
    x: jt.Num[jax.Array, "B T D"], w: dsv3_types.DSv3MoERouterWeightsPytree, metadata: RouterMetadata[jax.Array],
    *, routed_scaling_factor: float, mesh: jax.sharding.Mesh,
) -> tuple[jt.Num[jax.Array, "B T D"], dsv3_types.DSv3MoERouterWeightsPytree]:
  def _fwd(x, w):
    return dsv3_local_routing(
        x, w, metadata.selected_experts, metadata.probs, routed_scaling_factor=routed_scaling_factor, mesh=mesh,
    )
  (_, logits), vjp_fn = jax.vjp(_fwd, x, w)
  if grad_logits is None:
    grad_logits = jnp.zeros_like(logits)
  elif jax.typeof(grad_logits).sharding != jax.typeof(logits).sharding:
    grad_logits = jax.reshard(grad_logits, jax.typeof(logits).sharding)
  grad_x, grad_w = vjp_fn((grad_coeffs, grad_logits))
  return grad_x, grad_w


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _permute_impl(x_ag, metadata, num_experts_per_tok):
  return _permute_impl_fwd(x_ag, metadata, num_experts_per_tok)[0]


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def _permute_impl_fwd(x_ag: jt.Num[jax.Array, "NBT D0 D1"], metadata: RouterMetadata, num_experts_per_tok: int):
  num_local_tokens = jnp.sum(metadata.group_sizes, dtype=jnp.int32)
  out = permute_chunk(
      x_ag, metadata.sort_indices, 0, chunk_num_tokens(num_local_tokens, 0, metadata.capacity),
      capacity=metadata.capacity, num_experts_per_tok=num_experts_per_tok,
  )
  return out, metadata


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def _permute_impl_bwd(num_experts_per_tok: int, metadata: RouterMetadata, grads: jt.Num[jax.Array, "C D0 D1"]):
  grad_x_ag = unpermute_chunk(
      grads, metadata.reduce_metadata, num_out_tokens=metadata.sort_indices.shape[0] // num_experts_per_tok,
      num_experts_per_tok=num_experts_per_tok,
  )
  return grad_x_ag, None


_permute_impl.defvjp(_permute_impl_fwd, _permute_impl_bwd)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_permute(
    x_ag: jt.Num[jax.Array, "NBT D0 D1"], metadata: RouterMetadata,
    *, num_experts_per_tok: int, mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "C D0 D1"]:
  x_pspec = jax.typeof(x_ag).sharding.spec
  token_axis = dsv3_get_token_axis(x_pspec, num_model_dims=2)
  return jax.shard_map(
      functools.partial(_permute_impl, num_experts_per_tok=num_experts_per_tok),
      mesh=mesh, out_specs=jax.sharding.PartitionSpec(token_axis, None, None),
  )(x_ag, metadata)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_permute_bwd(
    grad_x_sorted: jt.Num[jax.Array, "C D0 D1"], metadata: RouterMetadata,
    *, num_experts_per_tok: int, mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "NBT D0 D1"]:
  x_pspec = jax.typeof(grad_x_sorted).sharding.spec
  token_axis = dsv3_get_token_axis(x_pspec, num_model_dims=2)

  def _bwd(grads, md):
    grad_x_ag, _ = _permute_impl_bwd(num_experts_per_tok, md, grads)
    return grad_x_ag

  return jax.shard_map(_bwd, mesh=mesh, out_specs=jax.sharding.PartitionSpec(token_axis, None, None))(
      grad_x_sorted, metadata,
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_dispatch_impl(
    x: jt.Num[jax.Array, "B T D"], *, expert_axis_name: str, axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> jt.Num[jax.Array, "NBT D0 D1"]:
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  x_flat = jnp.reshape(x, (-1, x.shape[-1] // 128, 128))
  (x_ag,) = _all_gather_sc0(x_flat, axis_name=physical_expert_axis_name)
  return x_ag


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_dispatch(
    x: jt.Num[jax.Array, "B T D"], *, expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]], mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "NBT D0 D1"]:
  x_pspec = jax.typeof(x).sharding.spec
  token_axis = dsv3_get_token_axis(x_pspec, num_model_dims=1)
  x_ag_spec = partition_specs.physical_pspec(jax.sharding.PartitionSpec(token_axis, None, None), axis_mapping)
  return jax.shard_map(
      functools.partial(dsv3_dispatch_impl, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping),
      mesh=mesh, out_specs=x_ag_spec,
  )(x)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_aggregated_dispatches_impl(
    moe_in: jt.Num[jax.Array, "B T D"], grad_out: jt.Num[jax.Array, "B T D"],
    *, expert_axis_name: str, axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> tuple[jt.Num[jax.Array, "NBT D0 D1"], jt.Num[jax.Array, "NBT D0 D1"]]:
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  moe_in_flat = jnp.reshape(moe_in, (-1, moe_in.shape[-1] // 128, 128))
  grad_out_flat = jnp.reshape(grad_out, (-1, grad_out.shape[-1] // 128, 128))
  x_ag, grad_y_ag = _all_gather_sc0(moe_in_flat, grad_out_flat, axis_name=physical_expert_axis_name)
  return x_ag, grad_y_ag


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_aggregated_dispatches(
    moe_in: jt.Num[jax.Array, "B T D"], grad_out: jt.Num[jax.Array, "B T D"],
    *, expert_axis_name: str, axis_mapping: Mapping[str, str | tuple[str, ...]], mesh: jax.sharding.Mesh,
) -> tuple[jt.Num[jax.Array, "NBT D0 D1"], jt.Num[jax.Array, "NBT D0 D1"]]:
  x1_pspec = jax.typeof(moe_in).sharding.spec
  x2_pspec = jax.typeof(grad_out).sharding.spec
  token_axis = dsv3_get_token_axis(x1_pspec, num_model_dims=1) or dsv3_get_token_axis(x2_pspec, num_model_dims=1)
  token_spec = partition_specs.physical_pspec(jax.sharding.PartitionSpec(token_axis, None, None), axis_mapping)
  return jax.shard_map(
      functools.partial(dsv3_aggregated_dispatches_impl, expert_axis_name=expert_axis_name, axis_mapping=axis_mapping),
      mesh=mesh, out_specs=(token_spec, token_spec),
  )(moe_in, grad_out)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_combine_impl(
    y_ag: jt.Num[jax.Array, "NBT D0 D1"], *, local_seq_length: int, expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]], combine_qtype: Any | None = None,
    combine_per_hop_scaling: bool = False,
) -> jt.Num[jax.Array, "B T D"]:
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  if combine_qtype is not None:
    y_local = _quantized_reduce_scatter_sc0(
        y_ag, axis_name=physical_expert_axis_name, qtype=combine_qtype, per_hop_scaling=combine_per_hop_scaling,
    )
  else:
    (y_local,) = _psum_scatter_sc0(y_ag, axis_name=physical_expert_axis_name)
  return jnp.reshape(y_local, (-1, local_seq_length, y_local.shape[-2] * y_local.shape[-1]))


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_combine(
    y_ag: jt.Num[jax.Array, "NBT D0 D1"], *, local_seq_length: int, expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]], mesh: jax.sharding.Mesh,
    out_specs: jax.sharding.PartitionSpec, combine_qtype: Any | None = None,
    combine_per_hop_scaling: bool = False,
) -> jt.Num[jax.Array, "B T D"]:
  out_specs_phys = partition_specs.physical_pspec(out_specs, axis_mapping)
  return jax.shard_map(
      functools.partial(
          dsv3_combine_impl, local_seq_length=local_seq_length, expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping, combine_qtype=combine_qtype, combine_per_hop_scaling=combine_per_hop_scaling,
      ), mesh=mesh, out_specs=out_specs_phys, check_vma=combine_qtype is None,
  )(y_ag)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _unpermute_impl(x, metadata, num_experts_per_tok):
  return _unpermute_impl_fwd(x, metadata, num_experts_per_tok)[0]


def _unpermute_impl_fwd(x, metadata, num_experts_per_tok):
  combined_x_ag, _ = _permute_impl_bwd(num_experts_per_tok, metadata, x)
  return combined_x_ag, metadata


def _unpermute_impl_bwd(num_experts_per_tok, metadata, cotangent_ag):
  grad_x, _ = _permute_impl_fwd(cotangent_ag, metadata, num_experts_per_tok)
  return grad_x, None


_unpermute_impl.defvjp(_unpermute_impl_fwd, _unpermute_impl_bwd)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_unpermute(
    x: jt.Num[jax.Array, "C D0 D1"], metadata: RouterMetadata[jax.Array],
    *, num_experts_per_tok: int, mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "NBT D0 D1"]:
  token_axis = dsv3_get_token_axis(jax.typeof(x).sharding.spec, num_model_dims=2)
  out_specs = jax.sharding.PartitionSpec(token_axis, None, None)
  return jax.shard_map(
      functools.partial(_unpermute_impl, num_experts_per_tok=num_experts_per_tok), mesh=mesh, out_specs=out_specs,
  )(x, metadata)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_unpermute_bwd(
    cotangent_ag: jt.Num[jax.Array, "NBT D0 D1"], metadata: RouterMetadata[jax.Array],
    *, num_experts_per_tok: int, mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "C D0 D1"]:
  return dsv3_permute(cotangent_ag, metadata, num_experts_per_tok=num_experts_per_tok, mesh=mesh)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def expert_group_mask(
    gate_logits: jt.Num[jax.Array, "BT E"], *, n_routing_groups: int, topk_routing_group: int, topk_in_group: int,
) -> jt.Num[jax.Array, "BT E"]:
  num_experts = gate_logits.shape[-1]
  scores_grouped_2d = jnp.reshape(gate_logits, (-1, num_experts // n_routing_groups))
  topk_in_group_vals, _ = jax.lax.top_k(scores_grouped_2d, k=topk_in_group)
  group_scores = jnp.reshape(
      jnp.sum(jnp.astype(topk_in_group_vals, jnp.float32), axis=-1), gate_logits.shape[:-1] + (n_routing_groups,),
  )
  _, group_idx = jax.lax.top_k(group_scores, k=topk_routing_group)
  group_mask = jax.nn.one_hot(group_idx, num_classes=n_routing_groups, dtype=jnp.float32)
  group_mask = jnp.sum(group_mask, axis=-2)
  score_mask_expanded = jnp.broadcast_to(group_mask[..., None], group_mask.shape + (num_experts // n_routing_groups,))
  return jnp.reshape(score_mask_expanded, score_mask_expanded.shape[:-2] + (num_experts,))


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def expert_indices_and_coeffs(
    gate_logits: jt.Num[jax.Array, "BT E"], pre_bias_logits: jt.Num[jax.Array, "BT E"],
    *, num_experts_per_tok: int, routed_scaling_factor: float, n_routing_groups: int,
    topk_routing_group: int, topk_in_group: int,
) -> tuple[jt.Num[jax.Array, "BT k"], jt.Num[jax.Array, "BT k"]]:
  expert_mask = expert_group_mask(
      gate_logits, n_routing_groups=n_routing_groups, topk_routing_group=topk_routing_group, topk_in_group=topk_in_group,
  )
  gate_logits = jnp.where(expert_mask > 0, gate_logits, -jnp.inf)
  _, indices = jax.lax.top_k(gate_logits, k=num_experts_per_tok)
  coeffs = selected_coeffs(pre_bias_logits, indices, routed_scaling_factor=routed_scaling_factor)
  return indices, coeffs


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def selected_coeffs(
    pre_bias_logits: jt.Num[jax.Array, "BT E"], indices: jt.Int[jax.Array, "BT k"], *, routed_scaling_factor: float,
) -> jt.Num[jax.Array, "BT k"]:
  num_experts_per_tok = indices.shape[-1]
  bt, num_e = pre_bias_logits.shape
  flat_logits = jnp.ravel(pre_bias_logits)
  flat_indices = jnp.ravel(indices + jnp.arange(bt, dtype=indices.dtype)[:, None] * num_e)
  if flat_indices.shape[0] % 1024 != 0:
    coeffs = flat_logits.at[flat_indices].get(mode="promise_in_bounds", wrap_negative_indices=False).reshape(bt, num_experts_per_tok)
  else:
    @compute_on(compute_type="tpu_sparsecore", out_memory_spaces=jax.memory.Space.Device,
                compiler_options={"sparse_core_config": {"core_ids": [0]}})
    def _sc_gather(flat_a: jax.Array, flat_idx: jax.Array) -> jax.Array:
      return flat_a.at[flat_idx].get(mode="promise_in_bounds", wrap_negative_indices=False)

    @compute_on(compute_type="tpu_sparsecore", out_memory_spaces=jax.memory.Space.Device,
                compiler_options={"sparse_core_config": {"core_ids": [0]}})
    def _sc_scatter_add_f32(zeros_f32: jax.Array, flat_idx: jax.Array, updates_f32: jax.Array) -> jax.Array:
      return zeros_f32.at[flat_idx].add(updates_f32, mode="promise_in_bounds", wrap_negative_indices=False)

    @jax.custom_vjp
    def _gather(flat_a: jax.Array, flat_idx: jax.Array) -> jax.Array:
      return _sc_gather(flat_a, flat_idx)

    def _gather_fwd(flat_a: jax.Array, flat_idx: jax.Array):
      return _sc_gather(flat_a, flat_idx), flat_idx

    def _gather_bwd(flat_idx: jax.Array, g: jax.Array):
      zeros_f32 = jnp.zeros(flat_logits.shape, dtype=jnp.float32)
      grad_a = _sc_scatter_add_f32(zeros_f32, flat_idx, g.astype(jnp.float32)).astype(g.dtype)
      return grad_a, None

    _gather.defvjp(_gather_fwd, _gather_bwd)
    coeffs = _gather(flat_logits, flat_indices).reshape(bt, num_experts_per_tok)
  return _normalize_coeffs(coeffs, routed_scaling_factor=routed_scaling_factor)


def _normalize_coeffs(coeffs: jt.Num[jax.Array, "BT k"], *, routed_scaling_factor: float) -> jt.Num[jax.Array, "BT k"]:
  coeffs_f32 = coeffs.astype(jnp.float32)
  coeffs_f32 = routed_scaling_factor * (coeffs_f32 / coeffs_f32.sum(-1, keepdims=True))
  return coeffs_f32.astype(coeffs.dtype)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def expert_selection(
    x: jt.Num[jax.Array, "BT D"], w: dsv3_types.DSv3MoERouterWeightsPytree, *, num_experts: int,
    num_experts_per_tok: int, routed_scaling_factor: float, n_routing_groups: int, topk_routing_group: int,
    topk_in_group: int, router_dtype: jax.typing.DTypeLike | None = None,
) -> tuple[jt.Num[jax.Array, "BT k"], jt.Num[jax.Array, "BT k"], jt.Num[jax.Array, "E"], jt.Num[jax.Array, "BT E"]]:
  return _expert_selection_impl(
      x, w, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok,
      routed_scaling_factor=routed_scaling_factor, n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group, topk_in_group=topk_in_group, router_dtype=router_dtype,
  )[:4]


def _expert_selection_impl(
    x: jt.Num[jax.Array, "BT D"], w: dsv3_types.DSv3MoERouterWeightsPytree, *, num_experts: int,
    num_experts_per_tok: int, routed_scaling_factor: float, n_routing_groups: int, topk_routing_group: int,
    topk_in_group: int, router_dtype: jax.typing.DTypeLike | None = None,
) -> tuple[
    jt.Num[jax.Array, "BT k"], jt.Num[jax.Array, "BT k"], jt.Num[jax.Array, "E"],
    jt.Num[jax.Array, "BT E"], jt.Num[jax.Array, "BT E"],
]:
  assert w.kernel is not None
  assert w.bias is not None
  input_dtype = jnp.result_type(x, w.kernel)
  compute_dtype = jnp.result_type(input_dtype, w.bias)
  if router_dtype is not None:
    compute_dtype = jnp.promote_types(compute_dtype, router_dtype)
  pre_bias_logits = jax.nn.sigmoid(
      jnp.tensordot(x.astype(input_dtype), w.kernel.astype(input_dtype), axes=1, preferred_element_type=compute_dtype)
  )
  if jnp.finfo(compute_dtype).bits < 32:
    finfo = jnp.finfo(compute_dtype)
    pre_bias_logits = jax.lax.reduce_precision(pre_bias_logits, exponent_bits=finfo.nexp, mantissa_bits=finfo.nmant)
  logits = pre_bias_logits + jax.lax.stop_gradient(w.bias.astype(compute_dtype))
  selected_experts, coeffs = expert_indices_and_coeffs(
      logits, pre_bias_logits, num_experts_per_tok=num_experts_per_tok, routed_scaling_factor=routed_scaling_factor,
      n_routing_groups=n_routing_groups, topk_routing_group=topk_routing_group, topk_in_group=topk_in_group,
  )
  coeffs = coeffs.astype(x.dtype)

  @compute_on(compute_type="tpu_sparsecore", out_memory_spaces=jax.memory.Space.Device,
              compiler_options={"sparse_core_config": {"core_ids": [0]}})
  def _bincount(x: jax.Array) -> jax.Array:
    return jnp.bincount(x, length=num_experts)

  group_sizes = _bincount(jnp.ravel(selected_experts))
  return selected_experts, coeffs, group_sizes, pre_bias_logits.astype(x.dtype), pre_bias_logits
