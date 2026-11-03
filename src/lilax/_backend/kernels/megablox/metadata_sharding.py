"""Shard selection for grouped matrix multiplication tile metadata."""

import jax.numpy as jnp


def select_shard(group_ids, m_tile_ids, group_tiles, start_group, end_group, num_groups):
  """Move a shard's tile IDs to the front and count its active tiles."""
  first_tile_in_shard = (group_ids < start_group).sum()
  group_ids = jnp.roll(group_ids, shift=-first_tile_in_shard, axis=0)
  m_tile_ids = jnp.roll(m_tile_ids, shift=-first_tile_in_shard, axis=0)
  iota = jnp.arange(num_groups, dtype=jnp.int32)
  active_group_mask = jnp.logical_and(iota <= end_group, iota >= start_group)
  group_tiles = jnp.where(active_group_mask, group_tiles, 0)
  return group_ids, m_tile_ids, group_tiles.sum()
