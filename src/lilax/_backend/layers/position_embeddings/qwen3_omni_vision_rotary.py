"""2D rotary position embedding for the Qwen3OmniMoe vision encoder."""

import jax.numpy as jnp

from flax import nnx

from lilax._backend.common.common_types import Array, DType
from lilax._backend.layers.position_embeddings.vision_block_coords import generate_block_coords


class Qwen3OmniMoeVisionRotaryEmbedding(nnx.Module):
  """Rotary position embedding for Qwen3OmniMoe vision encoder.

  Attributes:
    hidden_size: Hidden dimension size
    num_attention_heads: Number of attention heads
    spatial_merge_size: Spatial merge block size (e.g., 2 for 2x2 blocks)
    rope_theta: Base theta for frequency computation (default 10000.0)
    cast_as_fprop_dtype: Whether to cast to fprop dtype
    fprop_dtype: Output dtype
    rngs: RNG state passed in by nnx.bridge.to_linen, not used in this module
  """

  def __init__(
      self,
      hidden_size: int,
      num_attention_heads: int,
      spatial_merge_size: int,
      rope_theta: float = 10000.0,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      rngs: nnx.Rngs = None,
  ):
    """Initializes the Qwen3OmniMoe vision rotary embedding.

    Args:
      hidden_size: Hidden dimension size
      num_attention_heads: Number of attention heads
      spatial_merge_size: Spatial merge block size (e.g., 2 for 2x2 blocks)
      rope_theta: Base theta for frequency computation (default 10000.0)
      cast_as_fprop_dtype: Whether to cast to fprop dtype
      fprop_dtype: Output dtype
      rngs: RNG state passed in by nnx.bridge.to_linen, not used in this module
    """
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.spatial_merge_size = spatial_merge_size
    self.rope_theta = rope_theta
    self.cast_as_fprop_dtype = cast_as_fprop_dtype
    self.fprop_dtype = fprop_dtype
    self.rngs = rngs
    self.head_dim = self.hidden_size // self.num_attention_heads

  def _compute_freq_table(self, max_hw: int) -> Array:
    """Precompute frequency table for positions up to max_hw.

    Args:
      max_hw: Maximum height or width dimension

    Returns:
      Array of shape [max_hw, head_dim//4] containing frequencies for each position
    """

    inv_freq = 1.0 / (self.rope_theta ** (jnp.arange(0, self.head_dim // 2, 2, dtype=jnp.float32) / (self.head_dim // 2)))
    # Compute for all positions [0, max_hw)
    positions = jnp.arange(max_hw, dtype=jnp.float32)
    freqs = jnp.outer(positions, inv_freq)  # [max_hw, head_dim//4]
    return freqs

  def _generate_position_ids_single(self, num_frames: int, height: int, width: int) -> Array:
    """Generate 2D position IDs for a single image or video.

    Args:
      num_frames: Number of temporal frames (1 for images, >1 for videos)
      height: Height in patches
      width: Width in patches

    Returns:
      Array of shape [num_frames * height * width, 2] with (row_id, col_id)
    """
    merge_size = self.spatial_merge_size
    merged_h = height // merge_size
    merged_w = width // merge_size

    # Block indices
    block_rows = jnp.arange(merged_h)  # [merged_h]
    block_cols = jnp.arange(merged_w)  # [merged_w]

    # Intra-block offsets
    intra_row = jnp.arange(merge_size)  # [merge_size]
    intra_col = jnp.arange(merge_size)  # [merge_size]

    # Full resolution positions using broadcasting
    # Shape: [merged_h, 1, merge_size, 1]
    row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
    # Shape: [1, merged_w, 1, merge_size]
    col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

    # Expand to full grid and flatten
    row_idx = jnp.broadcast_to(row_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)
    col_idx = jnp.broadcast_to(col_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)

    coords = jnp.stack([row_idx, col_idx], axis=-1)  # [h*w, 2]

    # Repeat for video frames
    if num_frames > 1:
      coords = jnp.tile(coords, (num_frames, 1))

    return coords

  def compute_cos_sin(self, num_frames: int, height: int, width: int) -> tuple[Array, Array]:
    """Compute cos and sin embeddings for given static grid dimensions.

    Args:
      num_frames: Number of temporal frames
      height: Height in patches
      width: Width in patches

    Returns:
      Tuple of (cos_emb, sin_emb) each of shape [num_frames * height * width, head_dim]
    """
    max_hw = max(height, width)
    freq_table = self._compute_freq_table(max_hw)  # [max_hw, head_dim//4]
    coords = self._generate_position_ids_single(num_frames, height, width)  # [T*H*W, 2]

    row_freqs = freq_table[coords[:, 0]]  # [T*H*W, head_dim//4]
    col_freqs = freq_table[coords[:, 1]]  # [T*H*W, head_dim//4]

    # Concatenate row and column frequencies
    embeddings = jnp.concatenate([row_freqs, col_freqs], axis=-1)  # [T*H*W, head_dim//2]

    # Double the embeddings to match head_dim
    embeddings = jnp.concatenate([embeddings, embeddings], axis=-1)  # [T*H*W, head_dim]

    cos_emb = jnp.cos(embeddings)
    sin_emb = jnp.sin(embeddings)

    if self.cast_as_fprop_dtype:
      cos_emb = cos_emb.astype(self.fprop_dtype)
      sin_emb = sin_emb.astype(self.fprop_dtype)

    return cos_emb, sin_emb

  def _rotate_half(self, x: Array) -> Array:
    """Rotates half the hidden dims of the input.

    Args:
      x: Input tensor of any shape with last dimension divisible by 2

    Returns:
      Rotated tensor where (x1, x2) -> (-x2, x1)
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)

  def __call__(
      self,
      inputs: Array,
      num_frames: int,
      height: int,
      width: int,
      token_mask: Array | None = None,
      valid_grid: Array | None = None,
  ) -> Array:
    """Apply rotary position embeddings directly to inputs (Q or K tensors).

    Args:
      inputs: Input tensor of shape [B, S, N, head_dim] (batch, sequence, heads, head_dim)
             where sequence length S = num_frames * height * width (static maximum).
      num_frames: Number of temporal frames (static)
      height: Height in patches (static)
      width: Width in patches (static)
      token_mask: Optional mask identifying valid tokens in the padded sequence.
      valid_grid: Optional actual video grid with shape (batch, 3) (dynamic)

    Returns:
      Rotated inputs with same shape [B, S, N, head_dim]
    """
    is_3d = inputs.ndim == 3
    if is_3d:
      inputs = inputs[None, :, :, :]

    batch_size = inputs.shape[0]
    max_patches = num_frames * height * width

    if valid_grid is not None:
      if isinstance(valid_grid, (tuple, list)):
        valid_grid = jnp.array([valid_grid], dtype=jnp.int32)
      elif valid_grid.ndim == 1:
        valid_grid = valid_grid[None, :]
    else:
      valid_grid = jnp.tile(
          jnp.array([[num_frames, height, width]], dtype=jnp.int32),
          (batch_size, 1),
      )

    # Generate coordinates in block-based order
    row, col = generate_block_coords(valid_grid, max_patches, self.spatial_merge_size)

    # Compute frequencies dynamically using coordinates
    max_hw = max(height, width)
    freq_table = self._compute_freq_table(max_hw)  # [max_hw, head_dim//4]

    row_freqs = freq_table[row]  # [B, max_patches, head_dim//4]
    col_freqs = freq_table[col]  # [B, max_patches, head_dim//4]

    # Concatenate row and column frequencies
    embeddings = jnp.concatenate([row_freqs, col_freqs], axis=-1)  # [B, max_patches, head_dim//2]

    # Double the embeddings to match head_dim
    embeddings = jnp.concatenate([embeddings, embeddings], axis=-1)  # [B, max_patches, head_dim]

    cos_emb = jnp.cos(embeddings)
    sin_emb = jnp.sin(embeddings)

    # Mask out invalid (padded) tokens by setting cos to 1 and sin to 0
    if token_mask is not None:
      is_valid = token_mask[:, :, None]  # [B, max_patches, 1]
      cos_emb = jnp.where(is_valid, cos_emb, 1.0)
      sin_emb = jnp.where(is_valid, sin_emb, 0.0)

    if self.cast_as_fprop_dtype:
      cos_emb = cos_emb.astype(self.fprop_dtype)
      sin_emb = sin_emb.astype(self.fprop_dtype)

    # Reshape for broadcasting to inputs [B, S, N, H]
    cos_emb = cos_emb[:, :, None, :]  # [B, S, 1, H]
    sin_emb = sin_emb[:, :, None, :]

    rotated = inputs * cos_emb + self._rotate_half(inputs) * sin_emb

    if is_3d:
      rotated = rotated[0]

    return rotated
