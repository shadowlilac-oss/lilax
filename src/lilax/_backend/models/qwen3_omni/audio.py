"""Qwen3 Omni audio encoder and projector."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import math

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from lilax._backend.common.common_types import Array, AttentionType, Config, MODEL_MODE_TRAIN
from lilax._backend.layers import initializers as max_initializers
from lilax._backend.layers.attentions import Attention
from lilax._backend.layers.position_embeddings.sinusoidal import PositionalEmbedding
from lilax._backend.layers.initializers import nd_dense_init
from lilax._backend.layers.linears import DenseGeneral, MlpBlock


class Qwen3OmniAudioEncoderLayer(nnx.Module):
  """Transformer encoder layer for audio model."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs = None):  # pyrefly: ignore[bad-function-definition]
    self.config = config
    self.mesh = mesh
    self.rngs = rngs

    self.hidden_states_shape = (
        self.config.per_device_batch_size,
        self.config.max_source_positions_for_audio,
        self.config.d_model_for_audio,
    )

    self.input_layer_norm = nnx.LayerNorm(
        num_features=self.config.d_model_for_audio,
        epsilon=1e-5,
        dtype=self.config.dtype_mm,
        rngs=self.rngs,
    )

    self.self_attention_audio = Attention(
        config=self.config,
        num_query_heads=self.config.encoder_attention_heads_for_audio,
        num_kv_heads=self.config.encoder_attention_heads_for_audio,
        head_dim=self.config.d_model_for_audio // self.config.encoder_attention_heads_for_audio,
        max_target_length=self.config.max_source_positions_for_audio,
        attention_kernel="dot_product",
        inputs_q_shape=self.hidden_states_shape,
        inputs_kv_shape=self.hidden_states_shape,
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        mesh=self.mesh,
        dropout_rate=self.config.attention_dropout_for_audio,
        name="self_attention_audio",
        attention_type=AttentionType.FULL,
        is_nope_layer=True,  # No rotary position embeddings for audio
        use_bias_in_projections=True,
        use_qk_norm=False,
        query_pre_attn_scalar=1
        / math.sqrt(self.config.d_model_for_audio // self.config.encoder_attention_heads_for_audio),
        model_mode=MODEL_MODE_TRAIN,
        rngs=self.rngs,
    )

    self.post_attention_layer_norm = nnx.LayerNorm(
        num_features=self.config.d_model_for_audio,
        epsilon=1e-5,
        dtype=self.config.dtype_mm,
        rngs=self.rngs,
    )

    self.AudioMLP = MlpBlock(
        config=self.config,
        mesh=self.mesh,
        in_features=self.config.d_model_for_audio,
        intermediate_dim=self.config.encoder_ffn_dim_for_audio,
        activations=("gelu",),  # Single GELU activation
        kernel_init=max_initializers.nd_dense_init(self.config.dense_init_scale, "fan_in", "truncated_normal"),
        intermediate_dropout_rate=0.0,  # No dropout to match AudioMLP
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        use_bias=True,  # AudioMLP uses bias
        use_pre_norm=False,  # Norm is handled outside
        quant=None,  # No quantization
        model_mode=None,  # Not needed for encoder
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: Array,
      deterministic: bool = False,
  ):
    """Apply transformer encoder layer to audio hidden states.

    Args:
        hidden_states: Input tensor of shape (batch, seq_len, d_model_for_audio)
        deterministic: Whether to use deterministic mode (disable dropout)

    Returns:
        Output tensor of shape (batch, seq_len, d_model_for_audio)
    """
    residual = hidden_states
    hidden_states = self.input_layer_norm(hidden_states)
    hidden_states, _ = self.self_attention_audio(
        inputs_q=hidden_states,
        inputs_kv=hidden_states,
        deterministic=deterministic,
    )
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layer_norm(hidden_states)
    hidden_states = self.AudioMLP(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


class Qwen3OmniAudioEncoder(nnx.Module):
  """Full audio encoder with convs, positional embeddings, and transformer layers.

  Attributes:
      config: Config containing model parameters
      mesh: Mesh, JAX device mesh (used for sharding)
  """

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs = None):  # pyrefly: ignore[bad-function-definition]
    self.config = config
    self.mesh = mesh
    self.rngs = rngs

    self.positional_embedding = PositionalEmbedding(
        embedding_dims=self.config.d_model_for_audio,
        max_wavelength=self.config.max_timescale_for_audio,
        cast_as_fprop_dtype=True,
        fprop_dtype=self.config.dtype_mm,
    )

    self.layernorm_post = nnx.LayerNorm(
        num_features=self.config.d_model_for_audio,
        epsilon=1e-5,
        dtype=self.config.dtype_mm,
        rngs=self.rngs,
    )

    # Convolutional downsampling layers
    self.conv2d1 = nnx.Conv(
        in_features=1,
        out_features=self.config.downsample_hidden_size_for_audio,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=((1, 1), (1, 1)),
        use_bias=True,
        dtype=self.config.dtype_mm,
        param_dtype=self.config.weight_dtype,
        precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

    self.conv2d2 = nnx.Conv(
        in_features=self.config.downsample_hidden_size_for_audio,
        out_features=self.config.downsample_hidden_size_for_audio,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=((1, 1), (1, 1)),
        use_bias=True,
        dtype=self.config.dtype_mm,
        param_dtype=self.config.weight_dtype,
        precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

    self.conv2d3 = nnx.Conv(
        in_features=self.config.downsample_hidden_size_for_audio,
        out_features=self.config.downsample_hidden_size_for_audio,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=((1, 1), (1, 1)),
        use_bias=True,
        dtype=self.config.dtype_mm,
        param_dtype=self.config.weight_dtype,
        precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

    conv_out_dim = self.config.downsample_hidden_size_for_audio * (
        (((self.config.num_mel_bins_for_audio + 1) // 2 + 1) // 2 + 1) // 2
    )
    self.conv_out = DenseGeneral(
        in_features_shape=conv_out_dim,
        out_features_shape=self.config.d_model_for_audio,
        use_bias=False,
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        kernel_init=nd_dense_init(self.config.dense_init_scale, "fan_in", "normal"),
        matmul_precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

    # Transformer encoder layers
    for lyr in range(self.config.encoder_layers_for_audio):
      layer_name = f"layers_{lyr}"
      layer = Qwen3OmniAudioEncoderLayer(
          config=self.config,
          mesh=self.mesh,
          rngs=self.rngs,
      )
      setattr(self, layer_name, layer)

  def __call__(
      self,
      audio_features: Array,
      deterministic: bool = False,
  ):
    """Process audio features through convs + transformer encoder.

    Args:
        audio_features: Input of shape (batch, num_mel_bins, audio_length)
        deterministic: Whether to use deterministic mode

    Returns:
        Encoded features of shape (batch, seq_len, d_model_for_audio)
    """
    batch_size, num_mel_bins, audio_length = audio_features.shape
    chunk_size = self.config.n_window_for_audio * 2

    # Reshape to chunks
    num_chunks = audio_length // chunk_size
    audio_chunks = audio_features.reshape(batch_size, num_mel_bins, num_chunks, chunk_size)
    audio_chunks = audio_chunks.transpose(0, 2, 1, 3)
    audio_chunks = audio_chunks.reshape(batch_size * num_chunks, num_mel_bins, chunk_size)

    # Add channel dimension
    hidden_states = audio_chunks[:, :, :, jnp.newaxis]

    # Apply convolutional layers
    hidden_states = self.conv2d1(hidden_states)
    hidden_states = jax.nn.gelu(hidden_states)
    hidden_states = self.conv2d2(hidden_states)
    hidden_states = jax.nn.gelu(hidden_states)
    hidden_states = self.conv2d3(hidden_states)
    hidden_states = jax.nn.gelu(hidden_states)

    # Reshape conv output
    bc, f, t, c = hidden_states.shape
    hidden_states = hidden_states.transpose(0, 2, 3, 1)
    hidden_states = hidden_states.reshape(bc, t, c * f)
    hidden_states = self.conv_out(hidden_states)

    # Add positional embeddings
    seq_len_per_chunk = hidden_states.shape[1]
    pos_emb = self.positional_embedding(seq_len_per_chunk)
    pos_emb = jnp.broadcast_to(
        pos_emb[None, :, :], (batch_size * num_chunks, seq_len_per_chunk, self.config.d_model_for_audio)
    )
    hidden_states = hidden_states + pos_emb

    # Apply transformer encoder layers
    for lyr in range(self.config.encoder_layers_for_audio):
      layer_name = f"layers_{lyr}"
      layer = getattr(self, layer_name)
      hidden_states = layer(
          hidden_states,
          deterministic=deterministic,
      )

    hidden_states = self.layernorm_post(hidden_states)

    # Reshape back: (batch*chunks, seq_len_per_chunk, d_model) -> (batch, chunks*seq_len_per_chunk, d_model)
    hidden_states = hidden_states.reshape(batch_size, num_chunks * seq_len_per_chunk, self.config.d_model_for_audio)

    return hidden_states


class Qwen3OmniAudioProjector(nnx.Module):
  """Projection layer that converts audio encoder output to model embedding space."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs = None):  # pyrefly: ignore[bad-function-definition]
    self.config = config
    self.proj1 = DenseGeneral(
        in_features_shape=config.d_model_for_audio,
        out_features_shape=config.d_model_for_audio,
        use_bias=True,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        kernel_init=nd_dense_init(config.dense_init_scale, "fan_in", "normal"),
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

    self.proj2 = DenseGeneral(
        in_features_shape=config.d_model_for_audio,
        out_features_shape=config.output_dim_for_audio,
        use_bias=True,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        kernel_init=nd_dense_init(config.dense_init_scale, "fan_in", "normal"),
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(self, hidden_states: Array) -> Array:
    """
    Args:
        hidden_states: Encoder output of shape (num_chunks, seq_len, d_model_for_audio)

    Returns:
        Projected output of shape (num_chunks, seq_len, output_dim_for_audio)
    """
    hidden_states = self.proj1(hidden_states)
    hidden_states = jax.nn.gelu(hidden_states)
    hidden_states = self.proj2(hidden_states)
    return hidden_states
