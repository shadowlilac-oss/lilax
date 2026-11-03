"""Sinusoidal absolute positional embeddings."""

import dataclasses

import jax.numpy as jnp

from flax import nnx

from lilax._backend.common.common_types import Array, DType

_MAX_WAVELENGTH = 10_000


@dataclasses.dataclass(repr=False)
class PositionalEmbedding(nnx.Module):
  """Sinusoidal positional embeddings supporting both uniform and per-batch positions.

  This module computes sinusoidal positional embeddings and supports two use cases:

  1. Uniform positions across batch: All batch elements share the same position sequence.
     Pass position as 1D array (seq_len,) or None for sequential [0,1,2,...].
     Returns (seq_len, embedding_dims), caller broadcasts to batch.
     Example: pos_emb = layer(seq_len)  # Sequential positions
              pos_emb = layer(seq_len, position_1d)  # Custom 1D positions

  2. Per-batch positions (packed sequences): Each batch element has different positions.
     Pass position as 2D array (batch, seq_len).
     Returns (batch, seq_len, embedding_dims).
     Example: pos_emb = layer(seq_len, position_2d)

  As a side effect, the uniform case is more efficient since sin/cos are computed once
  and broadcasted, rather than per batch element.
  """

  #: The dimension of the embeddings.
  embedding_dims: int
  #: The maximum wavelength for the sinusoidal positional embeddings.
  max_wavelength: int = _MAX_WAVELENGTH
  #: Whether to cast output to fprop_dtype.
  cast_as_fprop_dtype: bool = False
  #: The dtype of the output when cast_as_fprop_dtype is True.
  fprop_dtype: DType = jnp.bfloat16
  #: RNG state passed in by nnx.bridge.to_linen, not used in this module.
  rngs: nnx.Rngs = None  # Not used in PositionalEmbedding but passed in by nnx.bridge.to_linen

  def _compute_embeddings(self, position: Array) -> Array:
    """Compute sinusoidal embeddings for given positions.

    Args:
      position: Either (seq_len,) for efficient path or (batch, seq_len) for full path.

    Returns:
      Embeddings of shape (seq_len, embedding_dims) or (batch, seq_len, embedding_dims).
    """
    num_timescales = self.embedding_dims // 2
    log_timescale_increment = jnp.log(float(self.max_wavelength)) / jnp.maximum(
        jnp.asarray(num_timescales, dtype=jnp.float32) - 1, 1
    )
    inv_timescales = jnp.exp(jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)

    if position.ndim == 1:
      # use the same position for the whole batch when position is (seq_len,)
      scaled_time = position[:, jnp.newaxis] * inv_timescales[jnp.newaxis, :]
    else:
      # when position is (batch, seq_len)
      position = position[:, :, jnp.newaxis]
      inv_timescales = inv_timescales[jnp.newaxis, jnp.newaxis, :]
      scaled_time = position * inv_timescales

    signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=-1)

    if self.cast_as_fprop_dtype:
      return signal.astype(self.fprop_dtype)
    else:
      return signal.astype(jnp.float32)

  def __call__(
      self,
      seq_len: int,
      position: Array | None = None,
  ) -> Array:
    """Compute positional embeddings.

    Args:
      seq_len: Sequence length for computing embeddings.
      position: Optional position array. If None, uses sequential [0,1,2,...].
        Shape can be (seq_len,) or (batch, seq_len) for packed sequences.

    Returns:
      Positional embeddings of shape (seq_len, embedding_dims) or
      (batch, seq_len, embedding_dims) if position has batch dimension.
    """
    if position is None:
      position = jnp.arange(seq_len, dtype=jnp.float32)

    return self._compute_embeddings(position)
