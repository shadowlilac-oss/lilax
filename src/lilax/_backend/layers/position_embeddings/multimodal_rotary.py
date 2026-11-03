"""Multi-dimensional rotary position embedding (MRoPE) for multimodal text sequences."""

import jax
import jax.numpy as jnp

from flax import nnx

from lilax._backend.common.common_types import DType
from lilax._backend.layers.position_embeddings.rotary import RotaryEmbedding


class Qwen3OmniMoeThinkerTextRotaryEmbedding(RotaryEmbedding):
  """Multi-dimensional Rotary Position Embedding (MRoPE) for Qwen3-Omni Thinker.

  This implements MRoPE which extends standard RoPE to handle 3D position IDs
  (temporal, height, width) for multimodal sequences containing text and vision tokens.

  For text-only sequences, it uses standard 2D position IDs.
  For sequences with vision tokens, it uses 3D position IDs where:
    - Dimension 0: Temporal position
    - Dimension 1: Height position (spatial)
    - Dimension 2: Width position (spatial)

  The implementation uses an interleaved pattern that reorganizes frequency
  components from chunked [TTT...HHH...WWW] to interleaved [THTHWHTHW...].
  """

  def __init__(
      self,
      min_timescale: int,
      max_timescale: int,
      embedding_dims: int = 0,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      mrope_section: tuple[int, int, int] | None = None,
      partial_rotary_factor: float = 1.0,
      attention_scaling: float = 1.0,
      rngs: nnx.Rngs = None,
  ):
    """Initializes the Qwen3OmniMoeThinkerTextRotaryEmbedding module.

    Args:
      min_timescale: Start of the geometric index (typically 1).
      max_timescale: End of the geometric index (rope_theta, e.g., 1000000).
      embedding_dims: Dimension of the attention head.
      cast_as_fprop_dtype: Whether to cast output to fprop dtype.
      fprop_dtype: The dtype of the output.
      mrope_section: Tuple of (temporal_dim, height_dim, width_dim) for MRoPE.
                     Defaults to [24, 20, 20] if None.
      partial_rotary_factor: Fraction of the head dimensions to rotate. The
        remaining suffix is passed through unchanged.
      attention_scaling: Scaling factor applied to cos/sin embeddings. Defaults to 1.0.
      rngs: rng keys passed in by nnx.bridge.to_linen.
    """
    if partial_rotary_factor is None or not 0.0 < partial_rotary_factor <= 1.0:
      raise ValueError(f"partial_rotary_factor must be in (0, 1], got {partial_rotary_factor}.")

    self.head_dim = embedding_dims
    self.partial_rotary_factor = partial_rotary_factor
    self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
    if self.rotary_dim <= 0 or self.rotary_dim % 2:
      raise ValueError("Rotary dim for rotary position embedding must be a positive multiple of 2.")

    super().__init__(
        min_timescale=min_timescale,
        max_timescale=max_timescale,
        mesh=None,
        embedding_dims=self.rotary_dim,
        cast_as_fprop_dtype=cast_as_fprop_dtype,
        fprop_dtype=fprop_dtype,
        rngs=rngs,
    )
    self.mrope_section = mrope_section if mrope_section is not None else (24, 20, 20)
    self.attention_scaling = attention_scaling

    if sum(self.mrope_section) != self.rotary_dim // 2:
      raise ValueError(
          f"mrope_section must describe rotary_dim / 2 frequencies; got {self.mrope_section} "
          f"for rotary_dim={self.rotary_dim}."
      )

  def _apply_interleaved_mrope(self, freqs: jax.Array) -> jax.Array:
    """Apply interleaved MRoPE pattern to 3D rotary embeddings.

    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THTHWHTHW...], preserving frequency continuity.

    Args:
      freqs: Shape (batch, seq_len, 3, head_dim // 2)
        Dimension -2 index 0: temporal frequencies
        Dimension -2 index 1: height frequencies
        Dimension -2 index 2: width frequencies

    Returns:
      freqs_t: Shape (batch, seq_len, rotary_dim // 2) with interleaved pattern
    """
    # Start with temporal frequencies
    freqs_t = freqs[..., 0, :]  # (batch, seq_len, head_dim // 2)

    # Create interleaved pattern
    # For each spatial dimension (H, W), place frequencies at positions:
    # offset=1 for H, offset=2 for W, with stride=3
    for dim_idx, offset in enumerate([1, 2], start=1):  # H=1, W=2
      section_size = self.mrope_section[dim_idx] * 3  # Total positions for this dimension
      # Select positions with stride 3, starting at offset
      # Use slice syntax to match PyTorch behavior
      idx = slice(offset, section_size, 3)
      # Replace those positions with the corresponding spatial frequencies
      freqs_t = freqs_t.at[..., idx].set(freqs[..., dim_idx, idx])

    return freqs_t

  def __call__(
      self,
      inputs: jax.Array,
      position: jax.Array,
  ) -> jax.Array:
    """Generates rotary position embeddings for multimodal sequences.

    Args:
      inputs: Input tensor of shape [batch, sequence, heads, head_dim]. MRoPE
        is applied to the first ``rotary_dim`` features and the rest are
        returned unchanged.
      position: Position IDs with shape:
        - [batch, sequence] for text-only (2D)
        - [batch, sequence, 3] for multimodal with vision (3D)
          where the last dim is (temporal, height, width)

    Returns:
      Tensor of shape [batch, sequence, heads, head_dim] with RoPE applied.
    """
    if len(inputs.shape) != 4:
      raise ValueError("Input is assumed to be a rank 4 tensor of shape [batch, sequence, heads, head_dim].")
    if self.head_dim != inputs.shape[3]:
      raise ValueError("The head dim of the rotary position embedding must match the hidden dimension of the inputs.")

    # Normalize trailing singleton: (batch, seq, 1) -> (batch, seq)
    if position.ndim == 3 and position.shape[-1] == 1:
      position = jnp.squeeze(position, axis=-1)

    # Handle both 2D (text-only) and 3D (multimodal) position IDs
    if position.ndim == 2:
      # Text-only: expand (batch, seq) -> (batch, seq, 3) with same positions
      position = jnp.broadcast_to(position[..., jnp.newaxis], position.shape + (3,))
    elif position.ndim != 3 or position.shape[-1] != 3:
      raise ValueError(f"Position IDs must be 2D (batch, seq) or 3D (batch, seq, 3), got shape {position.shape}")

    # Compute frequencies: (batch, seq, 3, 1) * (1, 1, 1, head_dim//2) -> (batch, seq, 3, head_dim//2)
    inv_freq_expanded = (1.0 / self.timescale)[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]  # (1, 1, 1, head_dim//2)
    position_expanded = position[..., jnp.newaxis]  # (batch, seq, 3, 1)
    freqs = position_expanded * inv_freq_expanded  # (batch, seq, 3, head_dim//2)

    # Apply interleaved MRoPE pattern for 3D positions
    freqs = self._apply_interleaved_mrope(freqs)  # (batch, seq, rotary_dim // 2)

    # Compute sin and cos
    # Duplicate frequencies for the two halves of the rotated prefix.
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    cos_emb = jnp.cos(emb) * self.attention_scaling
    sin_emb = jnp.sin(emb) * self.attention_scaling

    # Expand for the heads dimension.
    cos_emb = cos_emb[:, :, jnp.newaxis, :]
    sin_emb = sin_emb[:, :, jnp.newaxis, :]

    inputs_rotary, inputs_pass = jnp.split(inputs, [self.rotary_dim], axis=-1)
    rotated = self.apply_rotary(inputs_rotary, cos_emb, sin_emb)
    x_out = jnp.concatenate([rotated, inputs_pass], axis=-1)

    if self.cast_as_fprop_dtype:
      x_out = x_out.astype(self.fprop_dtype)

    return x_out
