"""Bilinearly interpolated learned 2D positional embeddings for the Qwen3OmniMoe vision encoder."""

import jax.numpy as jnp

from flax import nnx

from lilax._backend.common.common_types import Array, DType
from lilax._backend.layers.position_embeddings.vision_block_coords import generate_block_coords


class Qwen3OmniMoeVisionPosEmbedInterpolate(nnx.Module):
  """Bilinear interpolation of learned 2D positional embeddings for Qwen3OmniMoe vision.

  This module maintains a fixed grid of learned positional embeddings and interpolates
  them to match dynamic input dimensions using bilinear interpolation. This allows
  the model to handle images/videos of varying sizes while using a fixed embedding table.

  Attributes:
    num_position_embeddings: Number of position embeddings in the fixed grid
    hidden_size: Hidden dimension size
    spatial_merge_size: Spatial merge block size
    dtype: Data type for embeddings
    cast_as_fprop_dtype: Whether to cast to fprop dtype
    fprop_dtype: Output dtype
    rngs: RNG state passed in by nnx.bridge.to_linen
  """

  def __init__(
      self,
      num_position_embeddings: int,
      hidden_size: int,
      spatial_merge_size: int,
      dtype: DType = jnp.float32,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      rngs: nnx.Rngs = None,
  ):
    """Initializes the Qwen3OmniMoe vision position embedding interpolation module.

    Args:
      num_position_embeddings: Number of position embeddings in the fixed grid
      hidden_size: Hidden dimension size
      spatial_merge_size: Spatial merge block size
      dtype: Data type for embeddings
      cast_as_fprop_dtype: Whether to cast to fprop dtype
      fprop_dtype: Output dtype
      rngs: RNG state passed in by nnx.bridge.to_linen
    """
    self.num_position_embeddings = num_position_embeddings
    self.hidden_size = hidden_size
    self.spatial_merge_size = spatial_merge_size
    self.dtype = dtype
    self.cast_as_fprop_dtype = cast_as_fprop_dtype
    self.fprop_dtype = fprop_dtype
    self.rngs = rngs

    # Initialize the learned position embedding table
    if self.rngs is not None:
      # Initialize with normal distribution scaled by hidden_size^(-0.5)
      init_fn = nnx.initializers.normal(stddev=self.hidden_size**-0.5)
      self.pos_embed = nnx.Param(
          init_fn(
              self.rngs.params(),
              (self.num_position_embeddings, self.hidden_size),
              self.dtype,
          ),
      )
    self.num_grid_per_side = int(self.num_position_embeddings**0.5)

  def _interpolate_single(self, t: int, h: int, w: int) -> tuple[Array, Array]:
    """Compute bilinear interpolation indices and weights for a single image/video.

    Args:
      t: Number of temporal frames
      h: Target height in patches
      w: Target width in patches

    Returns:
      Tuple of (indices, weights) where:
        - indices: [4, h*w] indices into pos_embed for 4 corners
        - weights: [4, h*w] bilinear weights for 4 corners
    """
    N = self.num_grid_per_side

    # Create interpolation coordinates
    h_idxs = jnp.linspace(0, N - 1, h)
    w_idxs = jnp.linspace(0, N - 1, w)

    # Floor and ceiling indices
    h_idxs_floor = jnp.floor(h_idxs).astype(jnp.int32)
    w_idxs_floor = jnp.floor(w_idxs).astype(jnp.int32)
    h_idxs_ceil = jnp.minimum(h_idxs_floor + 1, N - 1)
    w_idxs_ceil = jnp.minimum(w_idxs_floor + 1, N - 1)

    # Fractional parts for interpolation weights
    dh = h_idxs - h_idxs_floor
    dw = w_idxs - w_idxs_floor

    # Compute flat indices for 2D grid
    base_h = h_idxs_floor * N
    base_h_ceil = h_idxs_ceil * N

    # 4 corner indices: (floor_h, floor_w), (floor_h, ceil_w), (ceil_h, floor_w), (ceil_h, ceil_w)
    indices = jnp.stack(
        [
            (base_h[:, None] + w_idxs_floor[None, :]).reshape(-1),
            (base_h[:, None] + w_idxs_ceil[None, :]).reshape(-1),
            (base_h_ceil[:, None] + w_idxs_floor[None, :]).reshape(-1),
            (base_h_ceil[:, None] + w_idxs_ceil[None, :]).reshape(-1),
        ],
        axis=0,
    )  # [4, h*w]

    # Bilinear weights
    weights = jnp.stack(
        [
            ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1),
            ((1 - dh)[:, None] * dw[None, :]).reshape(-1),
            (dh[:, None] * (1 - dw)[None, :]).reshape(-1),
            (dh[:, None] * dw[None, :]).reshape(-1),
        ],
        axis=0,
    )  # [4, h*w]

    return indices, weights

  def __call__(
      self,
      num_frames: int,
      height: int,
      width: int,
      video_grid_thw: Array | None = None,
      attention_mask: Array | None = None,
  ) -> Array:
    """Interpolate positional embeddings for given static grid dimensions.

    Args:
      num_frames: Number of temporal frames (static)
      height: Height in patches (static)
      width: Width in patches (static)
      video_grid_thw: Optional actual video grid with shape (batch, 3) (dynamic)
      attention_mask: Optional attention mask with shape (batch, max_patches) (dynamic)

    Returns:
      Interpolated positional embeddings of shape [batch, num_frames * height * width, hidden_size]
    """
    if video_grid_thw is not None:
      if isinstance(video_grid_thw, (tuple, list)):
        video_grid_thw = jnp.array([video_grid_thw], dtype=jnp.int32)
      elif video_grid_thw.ndim == 1:
        video_grid_thw = video_grid_thw[None, :]
      batch_size = video_grid_thw.shape[0]
    elif attention_mask is not None:
      batch_size = attention_mask.shape[0]
    else:
      batch_size = 1

    max_patches = num_frames * height * width

    if video_grid_thw is None:
      video_grid_thw = jnp.tile(
          jnp.array([[num_frames, height, width]], dtype=jnp.int32),
          (batch_size, 1),
      )
    if attention_mask is None:
      attention_mask = jnp.ones((batch_size, max_patches), dtype=jnp.int32)

    # Generate coordinates in block-based order
    row, col = generate_block_coords(video_grid_thw, max_patches, self.spatial_merge_size)

    # Normalize coordinates to [0, 1] range based on valid grid sizes
    V_H = video_grid_thw[:, 1:2]
    V_W = video_grid_thw[:, 2:3]
    row_norm = row / jnp.maximum(V_H - 1, 1)
    col_norm = col / jnp.maximum(V_W - 1, 1)

    # Interpolate from N x N grid
    N = self.num_grid_per_side
    table = self.pos_embed.value.reshape(N, N, self.hidden_size)

    y = row_norm * (N - 1)
    x = col_norm * (N - 1)

    y0 = jnp.floor(y).astype(jnp.int32)
    x0 = jnp.floor(x).astype(jnp.int32)
    y1 = jnp.minimum(y0 + 1, N - 1)
    x1 = jnp.minimum(x0 + 1, N - 1)

    dy = y - y0
    dx = x - x0

    # Gather 4 corners
    embed_00 = table[y0, x0]
    embed_01 = table[y0, x1]
    embed_10 = table[y1, x0]
    embed_11 = table[y1, x1]

    # Apply bilinear weights
    dy = dy[:, :, None]
    dx = dx[:, :, None]

    interpolated = (
        (1.0 - dy) * (1.0 - dx) * embed_00 + (1.0 - dy) * dx * embed_01 + dy * (1.0 - dx) * embed_10 + dy * dx * embed_11
    )

    # Mask out invalid (padded) tokens by setting their embeddings to zero
    interpolated = interpolated * attention_mask[:, :, None]

    if self.cast_as_fprop_dtype:
      interpolated = interpolated.astype(self.fprop_dtype)

    return interpolated
