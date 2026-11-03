"""Per-layer expert shuffling for DSv3 routed experts."""

from collections.abc import Mapping
import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import typeguard

from lilax._backend.experimental.lineage import dsv3_types

P = jax.sharding.PartitionSpec


def physical_expert_axis(expert_axis_name: str, axis_mapping: Mapping[str, str | tuple[str, ...]]) -> tuple[str, ...]:
  """Returns physical mesh axes backing the logical expert axis."""
  axes = axis_mapping.get(expert_axis_name, expert_axis_name)
  return (axes,) if isinstance(axes, str) else tuple(axes)


def num_expert_shards(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> int:
  """Returns the number of shards along the physical expert axis."""
  return math.prod(mesh.shape[a] for a in physical_expert_axis(expert_axis_name, axis_mapping))


def inverse_permutation(perm: jt.Int[jax.Array, "*L E"]) -> jt.Int[jax.Array, "*L E"]:
  """Returns the inverse of each permutation."""
  return jnp.argsort(perm, axis=-1).astype(perm.dtype)


def _pack_permutation(perm: jax.Array) -> tuple[jax.Array, int, int]:
  num_experts = perm.shape[0]
  bits = max(1, (num_experts - 1).bit_length())
  per_word = 32 // bits
  num_words = -(-num_experts // per_word)
  table = jnp.pad(perm.astype(jnp.uint32), (0, num_words * per_word - num_experts)).reshape(num_words, per_word)
  shifts = (jnp.arange(per_word, dtype=jnp.uint32) * bits)[None, :]
  packed = jnp.bitwise_or.reduce(table << shifts, axis=1)
  return packed, bits, per_word


@jt.jaxtyped(typechecker=typeguard.typechecked)
def remap_selected_experts(
    selected_experts: jt.Int[jax.Array, "..."], perm: jt.Int[jax.Array, "E"],
) -> jt.Int[jax.Array, "..."]:
  """Maps logical expert IDs to physical positions using a packed lookup."""
  packed, bits, per_word = _pack_permutation(perm)
  sel = selected_experts.astype(jnp.int32)
  word_index = sel // per_word
  shift = ((sel % per_word) * bits).astype(jnp.uint32)
  word = jnp.zeros(sel.shape, jnp.uint32)
  for i in range(packed.shape[0]):
    word = jnp.where(word_index == i, packed[i], word)
  mask = jnp.uint32((1 << bits) - 1)
  return ((word >> shift) & mask).astype(selected_experts.dtype)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def remap_group_sizes(group_sizes: jt.Int[jax.Array, "E"], perm: jt.Int[jax.Array, "E"]) -> jt.Int[jax.Array, "E"]:
  """Reorders logical expert counts into physical position order."""
  return jnp.zeros_like(group_sizes).at[perm].set(group_sizes)


def local_slot_routing(
    perm: jt.Int[jax.Array, "E"], *, shard_index: jt.Int[jax.Array, ""], num_shards: int,
) -> tuple[jt.Int[jax.Array, "E"], jt.Int[jax.Array, "E"], jt.Int[jax.Array, "E"], jt.Int[jax.Array, "E"]]:
  """Computes ragged all-to-all offsets and sizes for local expert slots."""
  num_experts = perm.shape[0]
  local_experts = num_experts // num_shards
  dtype = jnp.int32
  perm = perm.astype(dtype)
  slots = jnp.arange(local_experts, dtype=dtype)
  dest = perm[shard_index * local_experts + slots]
  dest_shard = dest // local_experts
  dest_slot = dest % local_experts
  same_dest = dest_shard[:, None] == dest_shard[None, :]
  earlier = slots[None, :] < slots[:, None]
  rank = jnp.sum(same_dest & earlier, axis=1, dtype=dtype)
  update = dest_shard * local_experts + rank
  zeros = jnp.zeros((num_experts,), dtype=dtype)
  input_offsets = zeros.at[update].set(slots)
  send_sizes = zeros.at[update].set(1)
  output_offsets = zeros.at[update].set(dest_slot)
  counts = jnp.sum((perm // local_experts).reshape(num_shards, local_experts) == shard_index, axis=1, dtype=dtype)
  recv_sizes = (slots[None, :] < counts[:, None]).reshape(-1).astype(dtype)
  return input_offsets, send_sizes, output_offsets, recv_sizes


Routing = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


def _zeros_like_typed(x: jax.Array) -> jax.Array:
  zeros = jnp.zeros_like(x)
  want = jax.typeof(x).mat
  have = jax.typeof(zeros).mat
  for ax in sorted(want.reduced - have.reduced):
    zeros = jax.lax.pcast(zeros, ax, to="reduced")
  for ax in sorted(want.unreduced - have.unreduced):
    if ax not in jax.typeof(zeros).mat.varying:
      zeros = jax.lax.pcast(zeros, ax, to="varying")
    zeros = jax.lax.pcast(zeros, ax, to="unreduced")
  return zeros


def _local_ragged_all_to_all(x: jax.Array, routing: Routing, axis_name: tuple[str, ...]) -> jax.Array:
  input_offsets, send_sizes, output_offsets, recv_sizes = routing
  return jax.lax.ragged_all_to_all(
      x, _zeros_like_typed(x), input_offsets, send_sizes, output_offsets, recv_sizes, axis_name=axis_name,
  )


def _ragged_permute_shard(x: jax.Array, perm: jax.Array, *, axis_name: tuple[str, ...]) -> jax.Array:
  num_shards = math.prod(jax.lax.axis_size(a) for a in axis_name)
  shard_index = jax.lax.axis_index(axis_name)
  routing = local_slot_routing(perm, shard_index=shard_index, num_shards=num_shards)
  return _local_ragged_all_to_all(x, routing, axis_name)


def _ragged_permute(x: jax.Array, perm: jax.Array, *, axis_name: tuple[str, ...]) -> jax.Array:
  x_spec = jax.typeof(x).sharding.spec
  mesh = jax.typeof(x).sharding.mesh
  partitions = x_spec.partitions
  dim0 = partitions[0] if partitions else None
  dim0 = (dim0,) if isinstance(dim0, str) else tuple(dim0 or ())
  assert dim0[: len(axis_name)] == axis_name, (
      "routed expert weights must be sharded along the expert axis" f" {axis_name} on dim 0, got spec {x_spec}"
  )
  perm = jax.reshard(perm, jax.sharding.NamedSharding(mesh, P()))
  return jax.shard_map(
      functools.partial(_ragged_permute_shard, axis_name=dim0), mesh=mesh,
      in_specs=(x_spec, P()), out_specs=x_spec, check_vma=False,
  )(x, perm)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _permute_experts(x: jax.Array, perm: jax.Array, inv_perm: jax.Array, axis_name: tuple[str, ...]) -> jax.Array:
  del inv_perm
  return _ragged_permute(x, perm, axis_name=axis_name)


def _permute_experts_fwd(x, perm, inv_perm, axis_name):
  return _ragged_permute(x, perm, axis_name=axis_name), inv_perm


def _permute_experts_bwd(axis_name, inv_perm, g):
  return _ragged_permute(g, inv_perm, axis_name=axis_name), None, None


_permute_experts.defvjp(_permute_experts_fwd, _permute_experts_bwd)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def shuffle_experts(
    xs: jt.PyTree[jt.Num[jax.Array, "..."]], perm: jt.Int[jax.Array, "E"],
    *, expert_axis_name: str, axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> jt.PyTree[jt.Num[jax.Array, "..."]]:
  """Moves expert-sharded arrays from logical to physical positions."""
  axis_name = physical_expert_axis(expert_axis_name, axis_mapping)
  inv_perm = inverse_permutation(perm)
  return jax.tree.map(lambda x: _permute_experts(x, perm, inv_perm, axis_name), xs)


def validate_expert_permutations(perm: Any, *, num_experts: int, num_layers: int | None = None) -> None:
  """Checks shape and concrete values of expert permutations."""
  if perm is None:
    return
  expected_rank = 1 if num_layers is None else 2
  if perm.ndim != expected_rank:
    raise ValueError(f"expert_permutations must have rank {expected_rank}, got {perm.shape}")
  if perm.shape[-1] != num_experts:
    raise ValueError("expert_permutations last dim must equal num_experts" f" ({num_experts}), got {perm.shape}")
  if num_layers is not None and perm.shape[0] < num_layers:
    raise ValueError(f"expert_permutations must have at least {num_layers} rows, got" f" {perm.shape}")
  if not jnp.issubdtype(perm.dtype, jnp.integer):
    raise ValueError(f"expert_permutations must be integer, got {perm.dtype}")
  if isinstance(perm, jax.core.Tracer):
    return
  rows = np.asarray(perm).reshape(-1, num_experts)
  expected = np.arange(num_experts)
  for i, row in enumerate(rows):
    if not np.array_equal(np.sort(row), expected):
      raise ValueError(f"expert_permutations row {i} is not a permutation of" f" range({num_experts}): {row}")


def _lpt_permutation(group_sizes: jax.Array, num_shards: int) -> jax.Array:
  num_experts = group_sizes.shape[-1]
  local_experts = num_experts // num_shards
  load = jnp.sum(group_sizes, axis=0, dtype=jnp.int32)
  order = jnp.argsort(-load, stable=True)
  full = jnp.iinfo(jnp.int32).max

  def body(i, carry):
    perm, shard_load, shard_count = carry
    e = order[i]
    open_load = jnp.where(shard_count < local_experts, shard_load, full)
    d = jnp.argmin(open_load)
    perm = perm.at[e].set(d * local_experts + shard_count[d])
    return perm, shard_load.at[d].add(load[e]), shard_count.at[d].add(1)

  init = (jnp.zeros((num_experts,), jnp.int32), jnp.zeros((num_shards,), jnp.int32), jnp.zeros((num_shards,), jnp.int32))
  return jax.lax.fori_loop(0, num_experts, body, init)[0]


@functools.partial(jax.jit, static_argnames=("num_shards",))
def lpt_expert_permutation(aux: dsv3_types.DSv3RouterAux[Any], *, num_shards: int) -> jt.Int[jax.Array, "*L E"]:
  """Balances expert load across shards using longest processing time first."""
  group_sizes = aux.group_sizes
  num_experts = group_sizes.shape[-1]
  if num_shards <= 0 or num_experts % num_shards:
    raise ValueError(f"num_experts ({num_experts}) must be divisible by num_shards" f" ({num_shards})")
  if group_sizes.ndim == 3:
    return jax.vmap(functools.partial(_lpt_permutation, num_shards=num_shards))(group_sizes)
  if group_sizes.ndim != 2:
    raise ValueError(
        "aux.group_sizes must have shape (num_token_shards, num_experts) or"
        f" (num_layers, num_token_shards, num_experts), got {group_sizes.shape}"
    )
  return _lpt_permutation(group_sizes, num_shards)


EXPERT_PERMUTATION_ALGORITHMS = ("none", "lpt")


def next_expert_permutation(
    algorithm: str, aux: dsv3_types.DSv3RouterAux[Any], *, num_shards: int,
) -> jax.Array | None:
  """Returns the next expert permutation for the selected algorithm."""
  if algorithm == "none":
    return None
  if algorithm == "lpt":
    return lpt_expert_permutation(aux, num_shards=num_shards)
  raise ValueError(
      f"Unknown expert_permutation algorithm {algorithm!r}; expected one of" f" {EXPERT_PERMUTATION_ALGORITHMS}"
  )
