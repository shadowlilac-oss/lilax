"""Rotary position embeddings over 1D token positions."""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from flax import nnx

from lilax._backend.common.common_types import ShardMode, DType


class RotaryEmbedding(nnx.Module):
  """Rotary Position Embedding."""

  def __init__(
      self,
      min_timescale: int,
      max_timescale: int,
      mesh: Mesh,
      embedding_dims: int = 0,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      shard_mode: ShardMode = ShardMode.AUTO,
      # Not used in RotaryEmbedding but passed in by nnx.bridge.to_linen.
      # TODO: Remove when bridge no longer needed
      rope_linear_scaling_factor: float = 1.0,
      rngs: nnx.Rngs = None,
  ):
    """Initializes the RotaryEmbedding module.

    Args:
      min_timescale: Start of the geometric index. Determines the periodicity of
        the added signal.
      max_timescale: End of the geometric index. Determines the frequency of the
        added signal.
      embedding_dims: Dimension of the embedding to be generated.
      cast_as_fprop_dtype: Whether to cast the output to the fprop dtype.
      fprop_dtype: The dtype of the output.
      rngs: rng keys passed in by nnx.bridge.to_linen.
    """
    self.min_timescale = min_timescale
    self.max_timescale = max_timescale
    self.mesh = mesh
    self.embedding_dims = embedding_dims
    self.cast_as_fprop_dtype = cast_as_fprop_dtype
    self.fprop_dtype = fprop_dtype
    self.shard_mode = shard_mode
    self.rope_linear_scaling_factor = rope_linear_scaling_factor

    if self.embedding_dims % 2:
      raise ValueError("Embedding dim for rotary position embedding must be a multiple of 2.")

  @property
  def timescale(self):
    """Returns the timescale for the rotary embedding."""
    half_embedding_dim = self.embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
    timescale = self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
    if self.rope_linear_scaling_factor != 1.0:
      timescale = timescale * self.rope_linear_scaling_factor
    return timescale

  def _rotate_half(self, x: jax.Array) -> jax.Array:
    """Rotates half the hidden dims of the input: (x1, x2) -> (-x2, x1)."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)

  def apply_rotary(self, inputs: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Applies the rotary transformation logic."""
    return (inputs * cos) + (self._rotate_half(inputs) * sin)

  def __call__(
      self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      inputs: jax.Array,
      position: None | jax.Array = None,
  ) -> jax.Array:
    """Generates a jax.Array of sinusoids with different frequencies.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. Since rotary position embeddings are applied to query and
        keys after projection, it is assumed of shape [B, S, N, H].
      position: Optional position jax.Array which denotes the position of each
        token in the sequence. This only needs to be supplied when the sequence
        is packed. It is of shape [B, S].

    Returns:
      a jax.Array of shape [B, S, N, H] which includes the inputs together with
      the rotary position embedding incorporated in it.
    """
    assert position is not None
    if len(inputs.shape) != 4:
      raise ValueError("Input is assumed to be a rank 4 tensor of shape" "[batch, sequence, heads, dims].")
    if self.embedding_dims != inputs.shape[3]:
      raise ValueError(
          "The embedding dims of the rotary position embedding" "must match the hidden dimension of the inputs."
      )

    position = position[:, :, jnp.newaxis, jnp.newaxis]
    sinusoid_inp = position / self.timescale
    sin_half = jnp.sin(sinusoid_inp).astype(inputs.dtype)
    cos_half = jnp.cos(sinusoid_inp).astype(inputs.dtype)

    sin = jnp.concatenate([sin_half, sin_half], axis=-1)
    cos = jnp.concatenate([cos_half, cos_half], axis=-1)

    x_out = self.apply_rotary(inputs, cos, sin)

    if self.cast_as_fprop_dtype:
      x_out = x_out.astype(self.fprop_dtype)
    return x_out


class PartialRotaryEmbedding(RotaryEmbedding):
  """Rotary Position Embedding applied to a partial fraction of dimensions."""

  def __init__(
      self,
      min_timescale: int,
      max_timescale: int,
      mesh: Mesh,
      embedding_dims: int = 0,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      partial_rotary_factor: float = 0.25,
      shard_mode: ShardMode = ShardMode.AUTO,
      rngs: nnx.Rngs = None,
  ):
    """Initializes the PartialRotaryEmbedding module.

    Args:
      min_timescale: Start of the geometric index. Determines the periodicity of
        the added signal.
      max_timescale: End of the geometric index. Determines the frequency of the
        added signal.
      embedding_dims: Dimension of the embedding to be generated.
      partial_rotary_factor: Ratio of dimensions to apply ROPE to
      rngs: rng keys passed in by nnx.bridge.to_linen.
    """
    self.head_dim = embedding_dims
    self.partial_rotary_factor = partial_rotary_factor
    self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)

    # Initialize the base class with only the rotary_dim
    super().__init__(
        min_timescale=min_timescale,
        max_timescale=max_timescale,
        mesh=mesh,
        embedding_dims=self.rotary_dim,
        cast_as_fprop_dtype=cast_as_fprop_dtype,
        fprop_dtype=fprop_dtype,
        shard_mode=shard_mode,
        rngs=rngs,
    )

  def __call__(self, inputs: jax.Array, position: None | jax.Array = None) -> jax.Array:
    """Applies Partial variant of rotary position embedding.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. It is assumed of shape [B, S, H, D].
      position: Optional position array [B, S]. Only needed when the sequence
        is packed.

    Returns:
      A jax.Array of shape [B, S, H, D - rotary_dim] with rotary position embeddings applied.
    """
    # Split, apply base RoPE to the first fraction, and concatenate
    inputs_rot, inputs_pass = jnp.split(inputs, [self.rotary_dim], axis=-1)
    inputs_rot = super().__call__(inputs_rot, position)
    inputs = jnp.concatenate([inputs_rot, inputs_pass], axis=-1)
    return inputs


class Gemma4PartialRotaryEmbedding(RotaryEmbedding):
  """Gemma 4 Rotary Position Embedding applied to a partial fraction of dimensions.

  Unlike standard PartialRotaryEmbedding which physically splits and concatenates
  features (resulting in a [Rotated, Unrotated] layout), Gemma 4 computes frequencies
  using the full embedding dimension denominator and pads the unrotated timescales
  with infinity.

  Because x / inf = 0, applying RoPE mathematically acts as an identity function
  on those unrotated dimensions. Because the base Rotary class splits the full tensor
  in half, this creates an interleaved feature layout in memory:
  [Rotated Half 1, Unrotated Half 1, Rotated Half 2, Unrotated Half 2].
  """

  def __init__(
      self,
      min_timescale: int,
      max_timescale: int,
      mesh: Mesh,
      embedding_dims: int = 0,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      partial_rotary_factor: float = 0.25,
      shard_mode: ShardMode = ShardMode.AUTO,
      rngs: nnx.Rngs = None,
  ):
    """Initializes the instance."""
    self.head_dim = embedding_dims
    self.partial_rotary_factor = partial_rotary_factor
    self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)

    # Pass the full head_dim to the base class so it splits at head_dim / 2,
    # ensuring the unrotated dimensions get correctly mixed into the center.
    super().__init__(
        min_timescale=min_timescale,
        max_timescale=max_timescale,
        mesh=mesh,
        embedding_dims=self.head_dim,
        cast_as_fprop_dtype=cast_as_fprop_dtype,
        fprop_dtype=fprop_dtype,
        shard_mode=shard_mode,
        rngs=rngs,
    )

  @property
  def timescale(self) -> jax.Array:
    """The inf-padded timescale for Gemma 4 rotary embedding."""
    half_rotary_dim = self.rotary_dim // 2

    # Gemma 4 uniquely uses the full head_dim as the denominator
    fraction = 2 * jnp.arange(0, half_rotary_dim) / self.head_dim
    timescale = self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction

    if getattr(self, "rope_linear_scaling_factor", 1.0) != 1.0:
      timescale = timescale * self.rope_linear_scaling_factor

    # Pad the remaining angles with jnp.inf.
    # When position is divided by inf, the angle becomes 0.
    # sin(0)=0 and cos(0)=1, which acts as a passthrough for unrotated dims.
    nope_angles = (self.head_dim // 2) - half_rotary_dim

    return jnp.pad(
        timescale,
        pad_width=(0, nope_angles),
        mode="constant",
        constant_values=(0.0, jnp.inf),
    )

  # Note: No __call__ override is required. The base RotaryEmbedding.__call__
  # handles the rotation perfectly using the padded self.timescale.


class LLaMARotaryEmbedding(RotaryEmbedding):
  """LLaMA variant of ROPE."""

  def __init__(
      self,
      min_timescale: int,
      max_timescale: int,
      mesh: Mesh,
      embedding_dims: int = 0,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      use_scale: bool = True,
      shard_mode: ShardMode = ShardMode.AUTO,
      # Not used in LLaMARotaryEmbedding but passed in by nnx.bridge.to_linen.
      # TODO: Remove when bridge no longer needed
      rngs: nnx.Rngs = None,
  ):
    """Initializes the LLaMARotaryEmbedding module.

    Args:
      min_timescale: Start of the geometric index. Determines the periodicity of
        the added signal.
      max_timescale: End of the geometric index. Determines the frequency of the
        added signal.
      embedding_dims: Dimension of the embedding to be generated.
      cast_as_fprop_dtype: Whether to cast the output to the fprop dtype.
      fprop_dtype: The dtype of the output.
      use_scale: Whether to apply LLaMA3.1 scaling factor.
      rngs: rng keys passed in by nnx.bridge.to_linen.
    """
    super().__init__(
        min_timescale=min_timescale,
        max_timescale=max_timescale,
        mesh=mesh,
        embedding_dims=embedding_dims,
        cast_as_fprop_dtype=cast_as_fprop_dtype,
        fprop_dtype=fprop_dtype,
        shard_mode=shard_mode,
        rngs=rngs,
    )

    # LLaMA3.1 ROPE scaling, see the original pytorch implementation:
    # https://github.com/meta-llama/llama-models/blob/301ca3a2b3b10e94ddcd1fdd2c57e52f812e1cac/models/llama3/reference_impl/model.py#L45C5-L45C18
    self.use_scale = use_scale

  @property
  def timescale(self):
    half_embedding_dim = self.embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
    fraction = jnp.repeat(fraction, 2)
    timescale = self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction

    # Apply scaling factor if enabled
    if self.use_scale:
      timescale = 1.0 / jax.vmap(self._apply_scaling_factor)(1.0 / timescale)

    # Expand timescale dimensions for broadcasting
    return timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]

  def _apply_scaling_factor(self, freq):
    """apply scaling factor to rotary position embedding."""
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * jnp.pi / freq

    def lower_wavelen(freq):
      return freq

    def bigger_or_equal_wavelen(freq):
      def bigger_wavelen(freq):
        return freq / scale_factor

      def equal_wavelen(freq):
        smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        return (1 - smooth) * freq / scale_factor + smooth * freq

      bigger_wavelen_cond = wavelen > low_freq_wavelen
      return jax.lax.cond(bigger_wavelen_cond, bigger_wavelen, equal_wavelen, freq)

    lower_wavelen_cond = wavelen < high_freq_wavelen
    return jax.lax.cond(lower_wavelen_cond, lower_wavelen, bigger_or_equal_wavelen, freq)

  def __call__(self, inputs: jax.Array, position: None | jax.Array = None) -> jax.Array:
    """Applies LLaMA variant of rotary position embedding.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. It is assumed of shape [B, S, N, H].
      position: Optional position array [B, S]. Only needed when the sequence
        is packed.

    Returns:
      A jax.Array of shape [B, S, N, H] with rotary position embeddings applied.
    """
    # Ensure input is 4D
    if len(inputs.shape) != 4:
      raise ValueError("Input is assumed to be a rank 4 tensor of shape [B, S, N, H].")
    if self.embedding_dims != inputs.shape[3]:
      raise ValueError(
          "The embedding dims of the rotary position embedding must match the hidden dimension of the inputs."
      )

    # Shift the inputs left and right as per LLaMA's specific behavior
    inputs_shifted_left = jnp.concatenate([inputs[..., 1:], inputs[..., :1]], axis=-1)
    inputs_shifted_right = jnp.concatenate([inputs[..., -1:], inputs[..., :-1]], axis=-1)
    # `lax.select` requires the predicate to match the cases in shape and, under
    # ShardMode.EXPLICIT, in sharding too. Tiling the 1-D mask to the full input shape yields a
    # replicated [B, S, N, H] predicate, which is rejected once the inputs are sharded.
    # Broadcasting the 1-D mask through `jnp.where` selects exactly the same elements -- verified
    # bit-for-bit in both the forward pass and the gradient -- and carries no sharding of its own.
    # It is used only under explicit sharding so that ShardMode.AUTO keeps the XLA fusion, and
    # hence the exact floating-point rounding, it has today.
    is_odd_dim = jnp.mod(jnp.arange(self.embedding_dims, dtype=jnp.int32), 2)
    if self.shard_mode == ShardMode.EXPLICIT:
      inputs_shifted = jnp.where(is_odd_dim.astype(bool), inputs_shifted_right, inputs_shifted_left)
    else:
      inputs_shifted = jax.lax.select(
          jnp.tile(is_odd_dim, inputs.shape[:-1] + (1,)),
          inputs_shifted_right,
          inputs_shifted_left,
      )

    # Determine positions if not provided
    if position is None:
      seq_length = inputs.shape[1]
      position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]

    # Calculate sinusoidal input
    position = position[:, :, jnp.newaxis, jnp.newaxis]
    sinusoid_inp = position / self.timescale

    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    # Apply alternating sign
    sign = jnp.tile(jnp.array([-1, 1]), self.embedding_dims // 2)

    # Combine original inputs with sinusoidal information
    outputs = inputs * cos + inputs_shifted * sin * sign

    if self.cast_as_fprop_dtype:
      outputs = outputs.astype(self.fprop_dtype)

    return outputs
