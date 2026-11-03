"""YaRN rotary position embedding."""

import math

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from flax import nnx

from lilax._backend.common.common_types import ShardMode, Array, DType
from lilax._backend.utils.sharding import create_sharding


def _cis(theta: Array) -> Array:
  """`exp(1j * theta)` for real `theta`, written as `cos(theta) + 1j*sin(theta)`.

  Bit-identical to `jnp.exp(1j * theta)` for every angle RoPE produces (the two
  differ only in the sign of a zero imaginary part at `theta == -0.0`), but it
  does not go through XLA's overflow-safe complex `exponential` expansion, which
  evaluates `exp(real(1j*theta)) == exp(0)` over the whole tensor. Under
  `shard_mode: explicit` it evaluates it *twice*: the `Sharding` custom-call on
  the broadcast `1j` keeps the simplifier from commuting the constant to the
  right, so the reassociation that lets CSE merge the two `exponential`s never
  fires.
  """
  return jax.lax.complex(jnp.cos(theta), jnp.sin(theta))


class YarnRotaryEmbedding(nnx.Module):
  """Yarn rotary embedding.

  Based on https://arxiv.org/abs/2309.00071
  This implementation uses DeepSeek-v3 PyTorch as reference
  https://github.com/deepseek-ai/DeepSeek-V3/blob/2f7b80eecebf3d1c84da5a0d465f6639ea175012/inference/model.py#L294

  Implementation Notes:
  - YaRN vs. Standard RoPE:
    1. Frequency Initialization: YaRN modifies how frequencies are computed.
    2. Attention Scaling: YaRN typically scales embeddings by `0.1 * ln(rope_factor) + 1.0`
       when `rope_factor > 1`. This scaling can be applied within this layer (if `attention_scaling=True`)
       or externally.
  - RoPE Implementation Details (General):
    - Arithmetic: Uses complex number arithmetic by default. `pairwise=True` uses real number arithmetic
      on (real, imag) pairs instead; the resulting embeddings are equivalent.
    - Input Layout: Supports both interleaved (`interleave=True`, e.g., [real1, img1, real2, img2]) and
      concatenated (`interleave=False`, e.g., [real1, real2, img1, img2]) formats.
    - Output Layout: Returns concatenated format ([real, imag]) by default, or interleaved format when
      `pairwise=True`. While the embedding is different, attention scores are invariant, as long as we
      apply the same output layout for Q and K.

  Attributes:
    embedding_dims: Dimension of the embedding to be generated.
    max_position_embeddings: The maximum sequence length that will be encountered.
    original_max_position_embeddings: The sequence length for which the base frequencies were defined.
    beta_fast: Lower bound parameter for correction.
    beta_slow: Upper bound parameter for correction.
    rope_theta: The base theta value for the frequency computation.
    rope_factor: Factor applied to adjust the frequencies.
    cast_as_fprop_dtype: Whether to cast the output to `fprop_dtype`.
    fprop_dtype: The forward pass dtype.
    rope_interleave: Whether complex representation is interleaved or concatenated.
    rope_truncate: Whether or not to floor lower bound and ceil upper bound for correction range.
    rope_attention_scaling: Whether or not to scale the rotary embedding output.
    pairwise: Whether to rotate interleaved inputs as real (real, imag) pairs and return interleaved output.
      Requires `interleave=True`.
    direct_position_freqs: Whether to compute angles directly as `position * inv_freqs` instead of
      gathering from a precomputed [max_position_embeddings, half_dim] table. Outputs match for positions
      < max_position_embeddings. Beyond that, the table gather clamps every position to the last row, while
      the direct computation keeps using the true angle.
    dot_pairwise: Whether to keep the pairwise rotation in rank 4 by applying the pair swap as a block-skew
      matmul instead of reshaping to rank 5. Requires `pairwise=True`.
    rngs: rng keys passed in by nnx.bridge.to_linen.
  """

  def __init__(
      self,
      embedding_dims: int,
      mesh: Mesh,
      max_position_embeddings: int = 4096 * 4,
      original_max_position_embeddings: int = 4096,
      beta_fast: float = 32,
      beta_slow: float = 1,
      rope_theta: float = 10000.0,
      rope_factor: float = 40,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      shard_mode: ShardMode = ShardMode.AUTO,
      interleave=True,
      truncate=True,
      attention_scaling=False,
      pairwise=False,
      direct_position_freqs=False,
      dot_pairwise=False,
      # Not used in YarnRotaryEmbedding but passed in by nnx.bridge.to_linen.
      # TODO: Remove when bridge no longer needed
      rngs: nnx.Rngs = None,
  ):
    """Initializes the YarnRotaryEmbedding module."""
    self.embedding_dims = embedding_dims
    self.max_position_embeddings = max_position_embeddings
    self.original_max_position_embeddings = original_max_position_embeddings
    self.beta_fast = beta_fast
    self.beta_slow = beta_slow
    self.rope_theta = rope_theta
    self.rope_factor = rope_factor
    self.cast_as_fprop_dtype = cast_as_fprop_dtype
    self.fprop_dtype = fprop_dtype
    self.interleave = interleave
    self.truncate = truncate
    self.mesh = mesh
    self.shard_mode = shard_mode
    self.attention_scaling = attention_scaling
    self.pairwise = pairwise
    self.direct_position_freqs = direct_position_freqs
    self.dot_pairwise = dot_pairwise

    if self.pairwise and not self.interleave:
      raise ValueError("rope_pairwise=True requires rope_interleave=True.")
    if self.dot_pairwise and not self.pairwise:
      raise ValueError("rope_dot_pairwise=True requires rope_pairwise=True.")

    # freqs are [batch, length, half_dim]; the trailing head-dim slice is never sharded.
    self.freqs_sharding = (
        create_sharding(mesh, ("activation_batch", "activation_length", None))
        if shard_mode == ShardMode.EXPLICIT
        else None
    )

    if self.embedding_dims % 2:
      raise ValueError("Embedding dim for rotary position embedding must be a multiple of 2.")

  @property
  def inv_freqs(self) -> Array:
    """1D base inverse frequencies of shape [half_dim] in float32."""
    half_dim = self.embedding_dims // 2
    # Compute base frequencies for each (even-indexed) dimension.
    # (Note: We use jnp.arange with float32 for precision.)
    freqs = 1.0 / (self.rope_theta ** (2.0 * jnp.arange(0, half_dim, dtype=jnp.float32) / self.embedding_dims))

    low, high = self._find_correction_range(
        self.beta_fast,
        self.beta_slow,
        self.embedding_dims,
        self.rope_theta,
        self.original_max_position_embeddings,
        self.truncate,
    )
    smooth = 1 - self._linear_ramp_factor(low, high, half_dim)
    # The corrected frequency is a weighted mix of the scaled and base values.
    return freqs / self.rope_factor * (1 - smooth) + freqs * smooth

  @property
  def freqs_cis(self):
    """Frequencies for every position, shape [max_position_embeddings, half_dim]."""
    # Precompute frequencies for all positions by taking the outer product.
    t = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)  # shape [max_position_embeddings]
    # This gives a [max_position_embeddings, half_dim] tensor with rows as time steps.
    return _cis(jnp.outer(t, self.inv_freqs))

  @staticmethod
  def _find_correction_dim(num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
    """Compute the correction dimension for a given number of rotations."""
    return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

  @staticmethod
  def _find_correction_range(
      low_rot: float,
      high_rot: float,
      dim: int,
      base: float,
      max_position_embeddings: int,
      truncate: bool,
  ):
    """Computes the range of correction dimensions for rotary positional embeddings.

    Args:
        low_rot (float): Lower bound for the number of rotations.
        high_rot (float): Upper bound for the number of rotations.
        dim (int): Dimensionality of the embedding space.
        base (float): Base value for the exponential computation.
        max_position_embeddings (int): Maximum sequence length.
        truncate (bool): Whether to floor lower bound and ceil upper bound.

    Returns:
        tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
    """
    low = YarnRotaryEmbedding._find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = YarnRotaryEmbedding._find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
      low = math.floor(low)
      high = math.ceil(high)
    low = max(low, 0)
    high = min(high, dim - 1)
    return low, high

  @staticmethod
  def _linear_ramp_factor(min_val: float, max_val: float, dim: int) -> Array:
    """Computes a linear ramp over the dimension.

    Returns a jax.Array of shape (dim,) with values between 0 and 1.
    """
    if min_val == max_val:
      max_val += 0.001  # Avoid division by zero.
    linear_func = (jnp.arange(dim, dtype=jnp.float32) - min_val) / (max_val - min_val)
    return jnp.clip(linear_func, 0, 1)

  def _position_angles(self, position: Array, inv_freqs: Array) -> Array:
    """Computes float32 RoPE angles position * inv_freqs of shape [B, S, 1, len(inv_freqs)]."""
    return (
        position.astype(jnp.float32)[:, :, jnp.newaxis, jnp.newaxis] * inv_freqs[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    )

  def _gather_freqs_cis(self, position: Array) -> Array:
    """Looks up precomputed complex frequencies, returning shape [B, S, 1, half_dim]."""
    freqs = self.freqs_cis.at[position].get(out_sharding=self.freqs_sharding)  # shape: [B, S, half_dim]
    return freqs[:, :, jnp.newaxis, :]

  def _cos_sin(self, position: Array, repeat_interleave: bool = False) -> tuple[Array, Array]:
    """Returns float32 (cos, sin) of the RoPE angles, each of shape [B, S, 1, half_dim].

    With repeat_interleave=True, each frequency is repeated twice ([f_0, f_0, f_1, f_1, ...]) to match the
    interleaved layout, giving shape [B, S, 1, dim].
    """
    if self.direct_position_freqs:
      inv_freqs = jnp.repeat(self.inv_freqs, 2) if repeat_interleave else self.inv_freqs
      angles = self._position_angles(position, inv_freqs)
      return jnp.cos(angles), jnp.sin(angles)
    freqs = self._gather_freqs_cis(position)
    cos, sin = jnp.real(freqs), jnp.imag(freqs)
    if repeat_interleave:
      cos = jnp.repeat(cos, 2, axis=-1)
      sin = jnp.repeat(sin, 2, axis=-1)
    return cos, sin

  def __call__(self, inputs: Array, position: None | Array = None) -> Array:
    """Applies the rotary positional embedding using the precomputed complex frequencies.

    Args:
      inputs: jax.Array of shape [B, S, N, H]. (H must equal self.embedding_dims.)
      position: jax.Array of shape [B, S] with integer positions (indexes into precomputed freqs).

    Returns:
      jax.Array of shape [B, S, N, H] with the rotary embedding applied.
    """
    if len(inputs.shape) != 4:
      raise ValueError("Input is assumed to be a rank 4 tensor of shape [batch, sequence, heads, dims].")
    if self.embedding_dims != inputs.shape[3]:
      raise ValueError(
          "The embedding dims of the rotary position embedding must match the hidden dimension of the inputs."
      )

    # Determine positions if not provided
    if position is None:
      seq_length = inputs.shape[1]
      position = jnp.arange(seq_length, dtype=jnp.int32)[jnp.newaxis, :]
    else:
      position = position.astype(jnp.int32)

    if self.interleave and self.pairwise:
      with jax.named_scope("rope_pairwise"):
        b, s, n, h = inputs.shape
        half_dim = h // 2
        if self.dot_pairwise:
          cos, sin = self._cos_sin(position, repeat_interleave=True)  # shape: [B, S, 1, dim]
          # Swap-and-sign [x_0, x_1, ...] -> [-x_1, x_0, ...] as a matmul against the block-skew
          # matrix I_{half_dim} (x) [[0, 1], [-1, 0]], avoiding the rank-5 reshape relayout copies.
          p_swap = jnp.kron(jnp.eye(half_dim, dtype=inputs.dtype), jnp.asarray([[0, 1], [-1, 0]], dtype=inputs.dtype))
          swapped_signed = jax.lax.dot_general(
              inputs,
              p_swap,
              (((3,), (0,)), ((), ())),
              # The default precision rounds non-bf16 inputs to bf16 on the TPU MXU; p_swap is exact in any dtype.
              precision=None if inputs.dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST,
              preferred_element_type=jnp.float32,
          )
          output = inputs.astype(jnp.float32) * cos + swapped_signed * sin
        else:
          pairs = inputs.reshape(b, s, n, half_dim, 2)
          pairs = pairs.astype(jnp.float32)
          cos, sin = self._cos_sin(position)  # shape: [B, S, 1, half_dim]
          cos = cos[..., jnp.newaxis]
          sin = sin[..., jnp.newaxis]
          swapped = jnp.flip(pairs, axis=-1)
          sign = jnp.asarray([-1.0, 1.0], dtype=jnp.float32)
          rotated_pairs = pairs * cos + swapped * sin * sign
          output = rotated_pairs.reshape(b, s, n, h)
    else:
      if self.direct_position_freqs:
        freqs = _cis(self._position_angles(position, self.inv_freqs))  # shape: [B, S, 1, half_dim]
      else:
        freqs = self._gather_freqs_cis(position)  # shape: [B, S, 1, half_dim]
      if self.interleave:
        # Inputs with interleaved format [real1, img1, real2, img2, ...] at last dimension
        # Convert the last dimension into a complex representation.
        # First reshape so that each pair of numbers represents the real and imaginary parts.
        b, s, n, h = inputs.shape
        half_dim = h // 2
        inputs_reshaped = inputs.reshape(b, s, n, half_dim, 2)
        first_half, second_half = inputs_reshaped[..., 0], inputs_reshaped[..., 1]
      else:
        # Inputs with concatenated format [real1, real2, ..., img1, img2, ...] at last dimension
        first_half, second_half = jnp.split(inputs, 2, axis=-1)

      inputs_complex = first_half + 1j * second_half  # shape: [b, s, n, half_dim]
      # Apply the rotary transformation via complex multiplication.
      rotated = jnp.multiply(inputs_complex, freqs)  # shape: [b, s, n, half_dim]

      # Convert the complex result back to a real tensor.
      # Split the complex number into its real and imaginary parts.
      # [real1, real2, ..., img1, img2, ...]
      output = jnp.concatenate([jnp.real(rotated), jnp.imag(rotated)], axis=-1)

    if self.attention_scaling:
      attention_scaling = 1.0 if self.rope_factor <= 1 else (0.1 * math.log(self.rope_factor) + 1.0)
      output = output * attention_scaling

    if self.cast_as_fprop_dtype:
      output = output.astype(self.fprop_dtype)
    return output
