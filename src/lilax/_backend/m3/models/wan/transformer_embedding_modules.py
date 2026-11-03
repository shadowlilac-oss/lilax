"""Wan transformer embedding modules."""

from typing import Any, Optional, Tuple

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from lilax._backend.m3.core.sharding import Sharding
from lilax._backend.common import common_types
from lilax._backend.utils import max_logging
from .transformer_embeddings import get_1d_rotary_pos_embed, get_sinusoidal_embeddings
from .transformer_timestep_embedding import TimestepEmbedding
from .utils_wan import TpuType, get_activation, get_tpu_type, param_metadata, safe_getattr

Array = common_types.Array


class FP32LayerNorm(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, dim: int, eps: float, elementwise_affine: bool):
    self.layer_norm = nnx.LayerNorm(rngs=rngs, num_features=dim, epsilon=eps, use_bias=elementwise_affine,
                                    use_scale=elementwise_affine, param_dtype=jnp.float32, dtype=jnp.float32)

  def __call__(self, inputs: jax.Array) -> jax.Array:
    origin_dtype = inputs.dtype
    return self.layer_norm(inputs.astype(dtype=jnp.float32)).astype(dtype=origin_dtype)


class SimpleFeedForward(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, dim: int, dim_out: Optional[int] = None, mult: int = 4,
               activation_fn: str = "gelu", dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32,
               precision: Optional[jax.lax.Precision] = None, sharding_specs: Optional[Any] = None,
               sharding: Sharding | None = None):
    inner_dim = int(dim * mult)
    dim_out = dim_out if dim_out is not None else dim
    net_0_kernel = safe_getattr(sharding_specs, "net_0_kernel", ("embed", "mlp"))
    net_0_bias = safe_getattr(sharding_specs, "net_0_bias", ("mlp",))
    net_2_kernel = safe_getattr(sharding_specs, "net_2_kernel", ("mlp", "embed"))
    net_2_bias = safe_getattr(sharding_specs, "net_2_bias", ("embed",))
    self.net_0 = nnx.Linear(dim, inner_dim, rngs=rngs, use_bias=True, dtype=dtype, param_dtype=weights_dtype,
                            precision=precision, kernel_init=nnx.initializers.lecun_normal(),
                            kernel_metadata=param_metadata(sharding, "ffn_net_0_kernel", net_0_kernel),
                            bias_init=nnx.initializers.zeros,
                            bias_metadata=param_metadata(sharding, "ffn_net_0_bias", net_0_bias))
    self.act = get_activation(activation_fn)
    self.net_2 = nnx.Linear(inner_dim, dim_out, rngs=rngs, use_bias=True, dtype=dtype, param_dtype=weights_dtype,
                            precision=precision, kernel_init=nnx.initializers.lecun_normal(),
                            kernel_metadata=param_metadata(sharding, "ffn_net_2_kernel", net_2_kernel),
                            bias_init=nnx.initializers.zeros,
                            bias_metadata=param_metadata(sharding, "ffn_net_2_bias", net_2_bias))

  def __call__(self, hidden_states: Array) -> Array:
    hidden_states = self.net_0(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.net_2(hidden_states)
    return hidden_states


class Timesteps(nnx.Module):
  def __init__(self, dim: int = 32, flip_sin_to_cos: bool = False, freq_shift: float = 1.0, scale: int = 1):
    self.dim, self.flip_sin_to_cos, self.freq_shift, self.scale = dim, flip_sin_to_cos, freq_shift, scale

  def __call__(self, timesteps):
    return get_sinusoidal_embeddings(timesteps, embedding_dim=self.dim, flip_sin_to_cos=self.flip_sin_to_cos,
                                     freq_shift=self.freq_shift)


class WanImageEmbedding(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, in_features: int, out_features: int, dtype: jnp.dtype,
               weights_dtype: jnp.dtype, precision: jax.lax.Precision, pos_embed_seq_len=None,
               alignment: Optional[int] = None, flash_min_seq_length: int = 4096, sharding: Sharding | None = None):
    self.norm1 = FP32LayerNorm(rngs=rngs, dim=in_features, elementwise_affine=True, eps=1e-6)
    self.ff = SimpleFeedForward(rngs=rngs, dim=in_features, dim_out=out_features, mult=1, activation_fn="gelu",
                                dtype=dtype, weights_dtype=weights_dtype, precision=precision, sharding=sharding)
    self.norm2 = FP32LayerNorm(rngs=rngs, dim=out_features, elementwise_affine=True, eps=1e-6)
    if alignment is None:
      tpu_type = get_tpu_type()
      self.alignment = 256 if tpu_type in [TpuType.TPU_V6_LITE, TpuType.TPU_7X] else 128
    else:
      self.alignment = alignment
    self.flash_min_seq_length = flash_min_seq_length
    if pos_embed_seq_len is not None:
      self.pos_embed = nnx.Param(jnp.zeros((1, pos_embed_seq_len, in_features), dtype=dtype))
    else:
      self.pos_embed = nnx.data(None)

  def __call__(self, encoder_hidden_states_image: jax.Array) -> tuple[jax.Array, jax.Array]:
    hidden_states = encoder_hidden_states_image
    B, current_seq_len, _ = hidden_states.shape
    if self.pos_embed is not None:
      pe_len = self.pos_embed.value.shape[1]
      add_len = min(current_seq_len, pe_len)
      hidden_states = hidden_states.at[:, :add_len, :].add(self.pos_embed.value[:, :add_len, :])
      if current_seq_len > pe_len:
        max_logging.log(f"[WARN] Input seq_len {current_seq_len} > pos_embed len {pe_len}")
    hidden_states = self.norm1(hidden_states)
    hidden_states = self.ff(hidden_states)
    hidden_states = self.norm2(hidden_states)
    B, current_seq_len, D_out = hidden_states.shape
    use_flash_attn = current_seq_len >= self.flash_min_seq_length
    if use_flash_attn:
      num_blocks = (current_seq_len + self.alignment - 1) // self.alignment
      target_seq_len = num_blocks * self.alignment
    else:
      target_seq_len = current_seq_len
    attention_mask = jnp.ones((B, current_seq_len), dtype=jnp.int32)
    if current_seq_len < target_seq_len:
      padding_size = target_seq_len - current_seq_len
      padding = jnp.zeros((B, padding_size, D_out), dtype=hidden_states.dtype)
      hidden_states = jnp.concatenate([hidden_states, padding], axis=1)
      padding_mask = jnp.zeros((B, padding_size), dtype=jnp.int32)
      attention_mask = jnp.concatenate([attention_mask, padding_mask], axis=1)
    if not use_flash_attn:
      attention_mask = None
    return hidden_states, attention_mask


class PixArtAlphaTextProjection(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, in_features: int, hidden_size: int, out_features: int = None,
               act_fn: str = "gelu_tanh", dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32,
               precision: jax.lax.Precision = None, sharding_specs: Optional[Any] = None,
               sharding: Sharding | None = None):
    linear_1_kernel = safe_getattr(sharding_specs, "emb_linear_1_kernel", ("embed", "mlp"))
    linear_1_bias = safe_getattr(sharding_specs, "emb_linear_1_bias", ("mlp",))
    linear_2_kernel = safe_getattr(sharding_specs, "emb_linear_2_kernel", ("mlp", "embed"))
    linear_2_bias = safe_getattr(sharding_specs, "emb_linear_2_bias", ("embed",))
    if out_features is None:
      out_features = hidden_size
    self.linear_1 = nnx.Linear(rngs=rngs, in_features=in_features, out_features=hidden_size, use_bias=True,
                               dtype=dtype, param_dtype=weights_dtype, precision=precision,
                               kernel_init=nnx.initializers.xavier_uniform(),
                               kernel_metadata=param_metadata(sharding, "text_embed_linear_1_kernel", linear_1_kernel),
                               bias_init=nnx.initializers.zeros,
                               bias_metadata=param_metadata(sharding, "text_embed_linear_1_bias", linear_1_bias))
    self.act_1 = get_activation(act_fn)
    self.linear_2 = nnx.Linear(rngs=rngs, in_features=hidden_size, out_features=out_features, use_bias=True,
                               dtype=dtype, param_dtype=weights_dtype, precision=precision,
                               kernel_init=nnx.initializers.xavier_uniform(),
                               kernel_metadata=param_metadata(sharding, "text_embed_linear_2_kernel", linear_2_kernel),
                               bias_init=nnx.initializers.zeros,
                               bias_metadata=param_metadata(sharding, "text_embed_linear_2_bias", linear_2_bias))

  def __call__(self, caption):
    hidden_states = self.linear_1(caption)
    hidden_states = self.act_1(hidden_states)
    return self.linear_2(hidden_states)


def get_frequencies(max_seq_len: int, theta: int, attention_head_dim: int):
  h_dim = w_dim = 2 * (attention_head_dim // 6)
  t_dim = attention_head_dim - h_dim - w_dim
  freqs = []
  for dim in [t_dim, h_dim, w_dim]:
    freqs.append(get_1d_rotary_pos_embed(dim, max_seq_len, theta, freqs_dtype=jnp.float32))
  freqs = jnp.concatenate(freqs, axis=1)
  t_size = attention_head_dim // 2 - 2 * (attention_head_dim // 6)
  hw_size = attention_head_dim // 6
  cumulative_sizes = np.cumsum([t_size, hw_size, hw_size])
  return jnp.split(freqs, cumulative_sizes[:-1].tolist(), axis=1)


class WanRotaryPosEmbed(nnx.Module):
  def __init__(self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0):
    self.attention_head_dim, self.patch_size, self.max_seq_len, self.theta = attention_head_dim, patch_size, max_seq_len, theta

  def __call__(self, hidden_states: jax.Array) -> jax.Array:
    _, num_frames, height, width, _ = hidden_states.shape
    p_t, p_h, p_w = self.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
    freqs_split = get_frequencies(self.max_seq_len, self.theta, self.attention_head_dim)
    freqs_f = jnp.expand_dims(jnp.expand_dims(freqs_split[0][:ppf], axis=1), axis=1)
    freqs_f = jnp.broadcast_to(freqs_f, (ppf, pph, ppw, freqs_split[0].shape[-1]))
    freqs_h = jnp.expand_dims(jnp.expand_dims(freqs_split[1][:pph], axis=0), axis=2)
    freqs_h = jnp.broadcast_to(freqs_h, (ppf, pph, ppw, freqs_split[1].shape[-1]))
    freqs_w = jnp.expand_dims(jnp.expand_dims(freqs_split[2][:ppw], axis=0), axis=1)
    freqs_w = jnp.broadcast_to(freqs_w, (ppf, pph, ppw, freqs_split[2].shape[-1]))
    freqs_concat = jnp.concatenate([freqs_f, freqs_h, freqs_w], axis=-1)
    return jnp.reshape(freqs_concat, (1, 1, ppf * pph * ppw, -1))


class WanTimeTextImageEmbedding(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, dim: int, time_freq_dim: int, time_proj_dim: int, text_embed_dim: int,
               image_embed_dim: Optional[int] = None, pos_embed_seq_len: Optional[int] = None,
               dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32,
               precision: jax.lax.Precision = None, flash_min_seq_length: int = 4096, sharding: Sharding | None = None):
    self.timesteps_proj = Timesteps(dim=time_freq_dim, flip_sin_to_cos=True, freq_shift=0)
    self.time_embedder = TimestepEmbedding(rngs=rngs, in_channels=time_freq_dim, time_embed_dim=dim, dtype=dtype,
                                           weights_dtype=weights_dtype, precision=precision, sharding=sharding)
    self.act_fn = get_activation("silu")
    self.time_proj = nnx.Linear(rngs=rngs, in_features=dim, out_features=time_proj_dim, dtype=jnp.float32,
                                param_dtype=weights_dtype, precision=precision, kernel_init=nnx.initializers.xavier_uniform(),
                                kernel_metadata=param_metadata(sharding, "time_proj_kernel", ("embed", "mlp")),
                                bias_init=nnx.initializers.zeros,
                                bias_metadata=param_metadata(sharding, "time_proj_bias", ("mlp",)))
    self.text_embedder = PixArtAlphaTextProjection(rngs=rngs, in_features=text_embed_dim, hidden_size=dim,
                                                  act_fn="gelu_tanh", sharding=sharding)
    self.image_embedder = nnx.data(None)
    if image_embed_dim is not None:
      self.image_embedder = WanImageEmbedding(rngs=rngs, in_features=image_embed_dim, out_features=dim,
                                              pos_embed_seq_len=pos_embed_seq_len, dtype=dtype,
                                              weights_dtype=weights_dtype, precision=precision,
                                              flash_min_seq_length=flash_min_seq_length, sharding=sharding)

  def __call__(self, timestep: jax.Array, encoder_hidden_states: jax.Array,
               encoder_hidden_states_image: Optional[jax.Array] = None, skip_embeddings: bool = False):
    timestep = self.timesteps_proj(timestep)
    temb = self.time_embedder(timestep)
    with jax.named_scope("time_proj"):
      timestep_proj = self.time_proj(self.act_fn(temb))
    if not skip_embeddings:
      encoder_hidden_states = self.text_embedder(encoder_hidden_states)
      encoder_attention_mask = None
      if encoder_hidden_states_image is not None:
        encoder_hidden_states_image, encoder_attention_mask = self.image_embedder(encoder_hidden_states_image)
    else:
      encoder_attention_mask = None
      if encoder_hidden_states_image is not None and encoder_hidden_states_image.shape[-1] != encoder_hidden_states.shape[-1]:
        img_dim = encoder_hidden_states_image.shape[-1]
        text_dim = encoder_hidden_states.shape[-1]
        if img_dim < text_dim:
          pad_shape = (encoder_hidden_states_image.shape[0], encoder_hidden_states_image.shape[1], text_dim - img_dim)
          encoder_hidden_states_image = jnp.concatenate([encoder_hidden_states_image,
                                                         jnp.zeros(pad_shape, dtype=encoder_hidden_states_image.dtype)], axis=-1)
        else:
          pad_shape = (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1], img_dim - text_dim)
          encoder_hidden_states = jnp.concatenate([encoder_hidden_states,
                                                   jnp.zeros(pad_shape, dtype=encoder_hidden_states.dtype)], axis=-1)
    return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image, encoder_attention_mask
