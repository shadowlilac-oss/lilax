"""Block-ordered patch coordinates for padded Qwen vision grids."""

import jax.numpy as jnp

from lilax._backend.common.common_types import Array


def generate_block_coords(video_grid_thw: Array, max_patches: int, merge_size: int) -> tuple[Array, Array]:
  """Generate row and col coordinates in block-based order for padded video.

  Args:
    video_grid_thw: Actual video grid with shape (batch, 3), in Qwen grid units.
    max_patches: Maximum number of patches (static).
    merge_size: Spatial merge block size (static).

  Returns:
    Tuple of (row, col) each of shape [batch, max_patches]
  """
  V_T = video_grid_thw[:, 0:1]  # [B, 1]
  V_H = video_grid_thw[:, 1:2]
  V_W = video_grid_thw[:, 2:3]

  merged_w = V_W // merge_size
  V_len = V_T * V_H * V_W

  idx = jnp.arange(max_patches, dtype=jnp.int32)[None, :]  # [1, max_patches]
  is_valid = idx < V_len  # [B, max_patches]

  stride_lh = V_H * V_W
  safe_stride_lh = jnp.maximum(stride_lh, 1)
  s_idx = idx % safe_stride_lh

  intra_col = s_idx % merge_size
  intra_row = (s_idx // merge_size) % merge_size

  safe_merged_w = jnp.maximum(merged_w, 1)
  block_elements = merge_size * merge_size
  block_col = (s_idx // block_elements) % safe_merged_w
  block_row = s_idx // (safe_merged_w * block_elements)

  row = jnp.where(is_valid, block_row * merge_size + intra_row, 0)
  col = jnp.where(is_valid, block_col * merge_size + intra_col, 0)

  return row, col
