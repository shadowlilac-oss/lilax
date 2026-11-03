"""2D rotary position embedding for the Llama4 vision encoder."""

import dataclasses

import jax.numpy as jnp

from flax import nnx

from lilax._backend.common.common_types import Array, DType


@dataclasses.dataclass(repr=False)
class LlamaVisionRotaryEmbedding(nnx.Module):
  """Rotary position embedding for Llama4 vision encoder.

  Based on Pytorch Reference
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py
  This implementation follows the Llama4 vision encoder's rotary embedding approach,
  which uses 2D coordinates (x, y) to generate rotary position embeddings.
  """

  #: size of the input image
  image_size: int
  #: size of the image patches
  patch_size: int
  #: size of the hidden dimension
  hidden_size: int
  #: number of attention heads
  num_attention_heads: int
  #: base theta value for the frequency computation
  rope_theta: float = 10000.0
  #: whether to cast the output to the fprop dtype
  cast_as_fprop_dtype: bool = True
  #: the dtype of the output
  fprop_dtype: DType = jnp.bfloat16
  # Not used in LlamaVisionRotaryEmbedding but passed in by nnx.bridge.to_linen.
  # TODO: Remove when bridge no longer needed
  #: RNG state passed in by nnx.bridge.to_linen, not used in this module
  rngs: nnx.Rngs = None

  @property
  def freqs_cis(self):
    """Frequencies for rotary embedding."""
    idx = self.image_size // self.patch_size
    img_idx = jnp.arange(idx**2, dtype=jnp.int32).reshape(idx**2, 1)
    img_idx = jnp.concatenate([img_idx, img_idx[:1]], axis=0)
    img_idx = img_idx.at[-1, -1].set(-2)  # ID_CLS_TOKEN

    # Get 2D coordinates
    frequencies_x = img_idx % idx  # x coordinates
    frequencies_y = img_idx // idx  # y coordinates

    # Compute frequency dimensions
    freq_dim = self.hidden_size // self.num_attention_heads // 2
    rope_freq = 1.0 / (self.rope_theta ** (jnp.arange(0, freq_dim, 2)[: (freq_dim // 2)].astype(jnp.float32) / freq_dim))

    # Compute frequencies for x and y coordinates
    freqs_x = (frequencies_x + 1)[..., None] * rope_freq[None, None, :]
    freqs_y = (frequencies_y + 1)[..., None] * rope_freq[None, None, :]

    # Interleave x and y frequencies
    freqs_x = jnp.repeat(freqs_x, 2, axis=-1)
    freqs_y = jnp.repeat(freqs_y, 2, axis=-1)

    # Combine frequencies
    freqs = jnp.concatenate([freqs_x, freqs_y], axis=-1).astype(jnp.float32)
    freqs = freqs[..., ::2]

    # Mask out invalid positions
    freqs = jnp.where(img_idx.reshape(-1, 1, 1) < 0, 0, freqs)
    # Convert to complex representation
    return jnp.exp(1j * freqs)

  def __call__(self, inputs: Array, position: None | Array = None) -> Array:
    """Applies rotary embeddings to the input tensor for Llama4 vision encoder.

    Args:
      inputs: Input tensor of shape [batch_size_times_tiles, num_patches_incl_cls, num_heads, head_dim]

    Returns:
      Tensor with rotary embeddings applied, maintaining the same shape as input.
    """
    if len(inputs.shape) != 4:
      raise ValueError(
          """Input is assumed to be a rank 4 tensor of shape [batch_size_times_tiles, num_patches_incl_cls,
          num_heads, head_dim]."""
      )

    # Reshape inputs to complex representation
    B, S, N, H = inputs.shape
    half_dim = H // 2

    # Convert the last dimension into a complex representation.
    # First reshape so that each pair of numbers represents the real and imaginary parts.
    inputs_reshaped = inputs.reshape(B, S, N, half_dim, 2)
    inputs_complex = inputs_reshaped[..., 0] + 1j * inputs_reshaped[..., 1]

    # Reshape freqs_ci for broadcasting
    freqs_ci = self.freqs_cis[jnp.newaxis, :, :, :]

    # Apply rotary transformation
    rotated = inputs_complex * freqs_ci

    # Convert the complex result back to a real tensor.
    # Split the complex number into its real and imaginary parts.
    rotated_real = jnp.stack([jnp.real(rotated), jnp.imag(rotated)], axis=-1)
    output = rotated_real.reshape(B, S, N, H)

    if self.cast_as_fprop_dtype:
      output = output.astype(self.fprop_dtype)

    return output
