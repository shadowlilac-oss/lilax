"""Splitting a batch into two micro-batches and merging them back."""

import jax
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

AnyJaxArray = jt.Num[jax.Array, "..."]
AnyJaxArrayOrPyTree = AnyJaxArray | jt.PyTree[AnyJaxArray]


@jt.jaxtyped(typechecker=typeguard.typechecked)
def split_microbatches(
    x: AnyJaxArrayOrPyTree, *, mesh: jax.sharding.Mesh,
) -> tuple[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree]:
  """Splits the batch into two micro-batches locally in a shard map."""
  if x is None:
    return None, None

  def _split_leaf(arr):
    s = jax.typeof(arr).sharding
    arr_pspec = getattr(s, "spec", jax.sharding.PartitionSpec())

    def _local_split(local_arr):
      return tuple(jnp.split(local_arr, 2, axis=0))

    return jax.shard_map(
        _local_split, mesh=mesh, in_specs=arr_pspec, out_specs=(arr_pspec, arr_pspec),
    )(arr)

  leaves, treedef = jax.tree.flatten(x)
  splits = [_split_leaf(leaf) for leaf in leaves]
  mb0 = jax.tree.unflatten(treedef, [s[0] for s in splits])
  mb1 = jax.tree.unflatten(treedef, [s[1] for s in splits])
  return mb0, mb1


@jt.jaxtyped(typechecker=typeguard.typechecked)
def merge_microbatches(mb0: AnyJaxArrayOrPyTree, mb1: AnyJaxArrayOrPyTree, *, mesh: jax.sharding.Mesh) -> AnyJaxArrayOrPyTree:
  """Merges two micro-batches back into a single batch in a shard map."""
  if mb0 is None:
    return None

  def _merge_leaf(arr0, arr1):
    s0 = jax.typeof(arr0).sharding
    s1 = jax.typeof(arr1).sharding
    spec0 = getattr(s0, "spec", jax.sharding.PartitionSpec())
    spec1 = getattr(s1, "spec", jax.sharding.PartitionSpec())
    arr_pspec = spec0 if spec0 is not None else spec1

    def _local_merge(local_arr0, local_arr1):
      return jnp.concatenate([local_arr0, local_arr1], axis=0)

    return jax.shard_map(
        _local_merge, mesh=mesh, in_specs=(arr_pspec, arr_pspec), out_specs=arr_pspec,
    )(arr0, arr1)

  return jax.tree.map(_merge_leaf, mb0, mb1)
