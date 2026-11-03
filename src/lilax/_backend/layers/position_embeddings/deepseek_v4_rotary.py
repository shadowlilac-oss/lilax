"""DeepSeek-V4 interleaved partial rotary position embedding."""

from typing import Any

import jax
import jax.numpy as jnp

from lilax._backend.layers.position_embeddings.rotary import RotaryEmbedding
from lilax._backend.layers.position_embeddings.yarn_rotary import YarnRotaryEmbedding


class DeepSeekV4RotaryEmbedding(RotaryEmbedding):
  """DeepSeek-V4 partial rotary embedding with interleaved frequencies.

  DeepSeek-V4 uses an interleaved positional encoding where consecutive channels
  are paired together. Unlike standard rotary models that split dimensions globally
  into first and second halves, this implementation pairs each even channel 2i
  with the corresponding odd channel 2i + 1.

  This results in two specific mathematical properties:
  1. Inverse frequencies are computed for (dim // 2) unique theta angles.
  2. Sinusoidal components are expanded consecutively (e.g., [f0, f0, f1, f1])
     prior to application.
  """

  def __init__(
      self,
      head_dim: int,
      partial_rotary_factor: float = 64.0 / 512.0,
      rope_theta: float = 10000.0,
      fprop_dtype: Any = jnp.float32,
      min_timescale: int = 10000,
      max_timescale: int = 10000,
      mesh: Any = None,
      rope_type: str = "default",
      rope_factor: float = 16.0,
      beta_fast: float = 32.0,
      beta_slow: float = 1.0,
      original_max_position_embeddings: int = 65536,
      truncate: bool = True,
      **kwargs,
  ):
    super().__init__(
        min_timescale=min_timescale,
        max_timescale=max_timescale,
        mesh=mesh,
        fprop_dtype=fprop_dtype,
        **kwargs,
    )
    self.head_dim = head_dim
    self.partial_rotary_factor = partial_rotary_factor
    self.rope_theta = rope_theta
    self.fprop_dtype = fprop_dtype
    self.rope_type = rope_type
    self.rope_factor = rope_factor
    self.beta_fast = beta_fast
    self.beta_slow = beta_slow
    self.original_max_position_embeddings = original_max_position_embeddings
    self.truncate = truncate

    # Compute the partial rotary dimension (rope_head_dim)
    self.dim = int(head_dim * partial_rotary_factor)

  @property
  def inv_freq(self):
    """Computes inverse frequencies with optional YaRN scaling."""
    # Compute base inverse frequencies for half of self.dim
    half_dim = self.dim // 2
    fraction = 2 * jnp.arange(0, half_dim, dtype=jnp.float32) / self.dim
    freqs = 1.0 / (self.rope_theta**fraction)
    if self.rope_type != "yarn":
      return freqs

    # pylint: disable=protected-access
    low, high = YarnRotaryEmbedding._find_correction_range(
        self.beta_fast,
        self.beta_slow,
        self.dim,
        self.rope_theta,
        self.original_max_position_embeddings,
        self.truncate,
    )
    smooth = 1 - YarnRotaryEmbedding._linear_ramp_factor(low, high, half_dim)
    return freqs / self.rope_factor * (1 - smooth) + freqs * smooth

  def get_freqs(self, position_ids: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # position_ids: [B, S]
    inv_freq_expanded = self.inv_freq[jnp.newaxis, jnp.newaxis, :]
    position_ids_expanded = position_ids[:, :, jnp.newaxis].astype(jnp.float32)
    freqs = position_ids_expanded * inv_freq_expanded

    cos = jnp.cos(freqs).astype(jnp.float32)  # [B, S, dim/2]
    sin = jnp.sin(freqs).astype(jnp.float32)  # [B, S, dim/2]

    return cos, sin

  def __call__(
      self,
      inputs: jnp.ndarray,
      position: jnp.ndarray,
      unsqueeze_dim: int | None = 1,
      reverse: bool = False,
  ) -> jnp.ndarray:
    """Applies interleaved Rotary Position Embedding to the inputs.

    Args:
      inputs: The input sequence on which to apply the RoPE.
      position: The positions of the tokens in the sequence.
      unsqueeze_dim: The axis at which to insert an additional dimension into `cos`/`sin`
        for broadcasting across attention heads. If None, no dimension is inserted.
      reverse: If True, applies reverse rotation (-sin).

    Returns:
      The inputs with the trailing `rope_dim` rotated.
    """
    cos, sin = self.get_freqs(position)
    if reverse:
      sin = -sin
    return _apply_rotary_pos_emb(inputs, cos, sin, unsqueeze_dim=unsqueeze_dim)


def _rotate_half(x: jax.Array) -> jax.Array:
  """Performs an interleaved half-rotation on the input tensor.

  Pairs adjacent elements in the last dimension and negates the second element
  of each pair. This transforms `[x0, x1, x2, x3]` into `[-x1, x0, -x3, x2]`.

  Args:
    x: Input tensor to be rotated.

  Returns:
    A rotated `jax.Array` of the exact same shape as the input.
  """
  x1 = x[..., 0::2]
  x2 = x[..., 1::2]
  return jnp.stack((-x2, x1), axis=-1).reshape(x.shape)


def _apply_rotary_pos_emb(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    unsqueeze_dim: int | None = 1,
) -> jax.Array:
  """Applies DeepSeek-V4 interleaved Rotary Positional Embedding to the input.

  The embedding is applied only to the trailing portion of the hidden dimension,
  leaving the leading portion (the 'nope' dimension) untouched. The trigonometric
  frequencies are provided in half-size and are duplicated consecutively to match
  the interleaved rotation structure.

  Args:
    x: Input tensor of shape `[..., seq_len, head_dim]` or similar, where the last
      dimension contains the combined nope and rope features.
    cos: Cosine frequencies of shape `[..., seq_len, rope_dim // 2]`.
    sin: Sine frequencies of shape `[..., seq_len, rope_dim // 2]`.
    unsqueeze_dim: The axis at which to insert an additional dimension into `cos` and `sin`
      for broadcasting across the attention heads. If None, no dimension is inserted. Default is 1.

  Returns:
    A `jax.Array` of the same shape and dtype as `x` with the trailing `rope_dim`
    rotated and the leading `nope_dim` unmodified.
  """
  cos = jnp.repeat(cos, 2, axis=-1)
  sin = jnp.repeat(sin, 2, axis=-1)

  # Insert an expansion dimension to align the frequency tensors (e.g., [B, S, D])
  # with the attention head axes of the input tensor (e.g., [B, S, H, D]).
  if unsqueeze_dim is not None:
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)

  rope_dim = cos.shape[-1]
  nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]

  rope_f32 = rope.astype(jnp.float32)
  rotated = ((rope_f32 * cos) + (_rotate_half(rope_f32) * sin)).astype(x.dtype)

  return jnp.concatenate([nope, rotated], axis=-1)
