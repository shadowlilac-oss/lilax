"""Wan 3D causal variational autoencoder components."""

from typing import List, Optional, Sequence, Tuple, Union

from diffusers.utils import BaseOutput
import flax
from flax import nnx
import jax
from jax import tree_util
import jax.numpy as jnp

from lilax._backend.m3.core.sharding import Sharding, TensorType
from lilax._backend.utils import max_logging
from .autoencoder_posterior_wan import (DiagonalGaussianDistribution, WanDiagonalGaussianDistribution,
                                        _wan_diag_gauss_dist_flatten, _wan_diag_gauss_dist_unflatten)
from .utils_wan import constrain, get_activation, param_metadata

_SPATIAL_NAME = "vae_spatial_activation"
_SPATIAL_AXES = ("activation_batch", None, None, "activation_length", None)


def _spatial_spec(sharding: Sharding | None):
  return None if sharding is None else sharding(_SPATIAL_NAME, _SPATIAL_AXES, TensorType.ACTIVATION)


@flax.struct.dataclass
class DecoderOutput(BaseOutput):
  sample: jnp.ndarray


@flax.struct.dataclass
class AutoencoderKLOutput(BaseOutput):
  latent_dist: "DiagonalGaussianDistribution"


try:
  flax.config.update("flax_always_shard_variable", False)
except LookupError:
  pass


from .autoencoder_cache_wan import (CACHE_T, RepSentinel, AutoencoderKLWanCache, _update_cache,
                                    _wan_cache_flatten, _wan_cache_unflatten)
from .autoencoder_causal_conv_wan import WanCausalConv3d, _canonicalize_tuple
from .autoencoder_norm_wan import WanRMS_norm
from .autoencoder_resample_wan import Identity, WanResample, WanUpsample, ZeroPaddedConv2D
from .autoencoder_block_wan import WanAttentionBlock, WanResidualBlock
from .autoencoder_stage_wan import WanMidBlock, WanUpBlock
from .autoencoder_encoder_wan import WanEncoder3d
from .autoencoder_decoder_wan import WanDecoder3d


_DEFAULT_LATENTS_MEAN = (-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                         0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921)
_DEFAULT_LATENTS_STD = (2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160)


class AutoencoderKLWan(nnx.Module):
  @staticmethod
  def latent_shape(vae_config: dict, batch: int, num_frames: int, height: int, width: int) -> tuple:
    downsample = vae_config.get("temperal_downsample") or [False, True, True]
    temporal, spatial = 2 ** sum(bool(d) for d in downsample), 2 ** len(downsample)
    return (batch, vae_config.get("z_dim", 16), (num_frames - 1) // temporal + 1, height // spatial, width // spatial)

  def __init__(self, rngs: nnx.Rngs, base_dim: int = 96, z_dim: int = 16, dim_mult: Optional[List[int]] = None,
               num_res_blocks: int = 2, attn_scales: Optional[List[float]] = None,
               temperal_downsample: Optional[List[bool]] = None, dropout: float = 0.0,
               latents_mean: Optional[List[float]] = None, latents_std: Optional[List[float]] = None,
               sharding: Sharding | None = None, dtype: jnp.dtype = jnp.float32,
               weights_dtype: jnp.dtype = jnp.float32, precision: jax.lax.Precision = None,
               vae_decode_chunk: int = 1, vae_encode_chunk: int = 4):
    dim_mult = [1, 2, 4, 4] if dim_mult is None else dim_mult
    attn_scales = [] if attn_scales is None else attn_scales
    temperal_downsample = [False, True, True] if temperal_downsample is None else temperal_downsample
    latents_mean = list(_DEFAULT_LATENTS_MEAN) if latents_mean is None else latents_mean
    latents_std = list(_DEFAULT_LATENTS_STD) if latents_std is None else latents_std
    self.z_dim = z_dim
    assert vae_decode_chunk >= 1 or vae_decode_chunk == -1, f"vae_decode_chunk must be >= 1 or -1, got {vae_decode_chunk}"
    assert vae_encode_chunk >= 1 or vae_encode_chunk == -1, f"vae_encode_chunk must be >= 1 or -1, got {vae_encode_chunk}"
    self.vae_decode_chunk = vae_decode_chunk
    self.vae_encode_chunk = vae_encode_chunk
    self.temperal_downsample = temperal_downsample
    self.temporal_upsample = temperal_downsample[::-1]
    self.temporal_downsample_factor = 2 ** sum(temperal_downsample)
    if self.vae_encode_chunk != -1:
      assert self.vae_encode_chunk % self.temporal_downsample_factor == 0, (
          f"vae_encode_chunk ({self.vae_encode_chunk}) must be a multiple of the temporal downsampling factor"
          f" ({self.temporal_downsample_factor})."
      )
    self.latents_mean = latents_mean
    self.latents_std = latents_std
    self.encoder = WanEncoder3d(rngs=rngs, dim=base_dim, z_dim=z_dim * 2, dim_mult=dim_mult,
                                num_res_blocks=num_res_blocks, attn_scales=attn_scales,
                                temperal_downsample=temperal_downsample, dropout=dropout, sharding=sharding,
                                dtype=dtype, weights_dtype=weights_dtype, precision=precision)
    self.quant_conv = WanCausalConv3d(rngs=rngs, in_channels=z_dim * 2, out_channels=z_dim * 2, kernel_size=1,
                                      sharding=sharding, dtype=dtype, weights_dtype=weights_dtype, precision=precision)
    self.post_quant_conv = WanCausalConv3d(rngs=rngs, in_channels=z_dim, out_channels=z_dim, kernel_size=1,
                                           sharding=sharding, dtype=dtype, weights_dtype=weights_dtype,
                                           precision=precision)
    self.decoder = WanDecoder3d(rngs=rngs, dim=base_dim, z_dim=z_dim, dim_mult=dim_mult,
                                num_res_blocks=num_res_blocks, attn_scales=attn_scales,
                                temperal_upsample=self.temporal_upsample, dropout=dropout, sharding=sharding,
                                dtype=dtype, weights_dtype=weights_dtype, precision=precision)
    self.sharding = sharding

  def _encode(self, x: jax.Array, feat_cache: AutoencoderKLWanCache):
    feat_cache.init_cache()
    if x.shape[-1] != 3:
      x = jnp.transpose(x, (0, 2, 3, 4, 1))
      assert x.shape[-1] == 3, f"Expected input shape (N, D, H, W, 3), got {x.shape}"
    x = x.astype(self.encoder.conv_in.conv.dtype)
    t = x.shape[1]
    CHUNK_SIZE = self.vae_encode_chunk
    if CHUNK_SIZE == -1:
      T_rest = max(1, t - 1)
      factor = self.temporal_downsample_factor
      CHUNK_SIZE = ((T_rest + factor - 1) // factor) * factor
    iter_ = 1 + ((t - 1 + CHUNK_SIZE - 1) // CHUNK_SIZE) if t > 1 else 1
    enc_feat_map = feat_cache._enc_feat_map
    spatial_sharding = _spatial_spec(self.sharding)

    def finalize(out, enc_feat_map):
      feat_cache._enc_feat_map = enc_feat_map
      enc = self.quant_conv(out)
      return enc

    with jax.named_scope("AutoencoderKLWan_encode_chunk_0"):
      chunk_0 = x[:, :1, ...]
      out_0, enc_feat_map, _ = self.encoder(chunk_0, feat_cache=enc_feat_map, feat_idx=0)
      if spatial_sharding is not None:
        out_0 = self.sharding.constrain(out_0, _SPATIAL_NAME, _SPATIAL_AXES)
    if iter_ <= 1:
      return finalize(out_0, enc_feat_map)
    with jax.named_scope("AutoencoderKLWan_encode_chunk_1"):
      chunk_1 = x[:, 1 : (1 + CHUNK_SIZE), ...]
      out_1, enc_feat_map, _ = self.encoder(chunk_1, feat_cache=enc_feat_map, feat_idx=0)
      if spatial_sharding is not None:
        out_1 = self.sharding.constrain(out_1, _SPATIAL_NAME, _SPATIAL_AXES)
    if iter_ <= 2:
      out = jnp.concatenate([out_0, out_1], axis=1)
      if spatial_sharding is not None:
        out = self.sharding.constrain(out, _SPATIAL_NAME, _SPATIAL_AXES)
      return finalize(out, enc_feat_map)
    x_rest = x[:, 1 + CHUNK_SIZE :, ...]
    T_rest = x_rest.shape[1]
    pad_amount = (-T_rest) % CHUNK_SIZE
    if pad_amount > 0:
      pad_shape = (x_rest.shape[0], pad_amount, *x_rest.shape[2:])
      x_rest_padded = jnp.concatenate([x_rest, jnp.zeros(pad_shape, dtype=x_rest.dtype)], axis=1)
    else:
      x_rest_padded = x_rest
    T_padded = T_rest + pad_amount
    num_scan_iters = T_padded // CHUNK_SIZE
    x_scannable = x_rest_padded.reshape(x_rest_padded.shape[0], num_scan_iters, CHUNK_SIZE,
                                         x_rest_padded.shape[2], x_rest_padded.shape[3], x_rest_padded.shape[4])
    x_scannable = jnp.transpose(x_scannable, (1, 0, 2, 3, 4, 5))
    graphdef, state = nnx.split(self.encoder)

    @jax.named_scope("AutoencoderKLWan_encode_scan_body")
    def scan_fn(carry, chunk):
      current_feat_map = carry
      local_encoder = nnx.merge(graphdef, state)
      if spatial_sharding is not None:
        chunk = self.sharding.constrain(chunk, _SPATIAL_NAME, _SPATIAL_AXES)
      out_chunk, next_feat_map, _ = local_encoder(chunk, feat_cache=current_feat_map, feat_idx=0)
      if spatial_sharding is not None:
        out_chunk = self.sharding.constrain(out_chunk, _SPATIAL_NAME, _SPATIAL_AXES)
      next_feat_map = jax.tree_util.tree_map(
          lambda x: self.sharding.constrain(x, _SPATIAL_NAME, _SPATIAL_AXES)
          if spatial_sharding is not None and hasattr(x, "shape") and x.ndim == len(_SPATIAL_AXES) else x,
          next_feat_map,
      )
      return next_feat_map, out_chunk

    enc_feat_map, out_rest = jax.lax.scan(scan_fn, enc_feat_map, x_scannable)
    out_rest = jnp.transpose(out_rest, (1, 0, 2, 3, 4, 5))
    out_rest = out_rest.reshape(out_rest.shape[0], -1, out_rest.shape[3], out_rest.shape[4], out_rest.shape[5])
    out_rest = out_rest[:, : T_rest // self.temporal_downsample_factor, ...]
    out = jnp.concatenate([out_0, out_1, out_rest], axis=1)
    if spatial_sharding is not None:
      out = self.sharding.constrain(out, _SPATIAL_NAME, _SPATIAL_AXES)
    return finalize(out, enc_feat_map)

  @jax.named_scope("AutoencoderKLWan_encode")
  def encode(self, x: jax.Array, feat_cache: AutoencoderKLWanCache,
             return_dict: bool = True) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
    h = self._encode(x, feat_cache)
    posterior = WanDiagonalGaussianDistribution(h)
    if not return_dict:
      return (posterior,)
    return AutoencoderKLOutput(latent_dist=posterior)

  def _decode(self, z: jax.Array, feat_cache: AutoencoderKLWanCache,
              return_dict: bool = True) -> Union[DecoderOutput, jax.Array]:
    feat_cache.init_cache()
    iter_ = z.shape[1]
    z = z.astype(self.post_quant_conv.conv.dtype)
    x = self.post_quant_conv(z)
    dec_feat_map = feat_cache._feat_map
    spatial_sharding = _spatial_spec(self.sharding)
    with jax.named_scope("AutoencoderKLWan_decode_chunk_0"):
      chunk_in_0 = x[:, 0:1, ...]
      if spatial_sharding is not None:
        chunk_in_0 = self.sharding.constrain(chunk_in_0, _SPATIAL_NAME, _SPATIAL_AXES)
      out_0, dec_feat_map, _ = self.decoder(chunk_in_0, feat_cache=dec_feat_map, feat_idx=0)
      if spatial_sharding is not None:
        out_0 = self.sharding.constrain(out_0, _SPATIAL_NAME, _SPATIAL_AXES)
    if iter_ > 1:
      with jax.named_scope("AutoencoderKLWan_decode_chunk_1"):
        chunk_in_1 = x[:, 1:2, ...]
        if spatial_sharding is not None:
          chunk_in_1 = self.sharding.constrain(chunk_in_1, _SPATIAL_NAME, _SPATIAL_AXES)
        out_chunk_1, dec_feat_map, _ = self.decoder(chunk_in_1, feat_cache=dec_feat_map, feat_idx=0)
        if spatial_sharding is not None:
          out_chunk_1 = self.sharding.constrain(out_chunk_1, _SPATIAL_NAME, _SPATIAL_AXES)
      out_1 = out_chunk_1
      out_list = [out_0, out_1]
      if iter_ > 2:
        x_rest = x[:, 2:, ...]
        T_rest = x_rest.shape[1]
        K = self.vae_decode_chunk
        if K == -1:
          K = max(1, T_rest)
        pad_amount = (-T_rest) % K
        if pad_amount > 0:
          pad_shape = (x_rest.shape[0], pad_amount, *x_rest.shape[2:])
          x_rest_padded = jnp.concatenate([x_rest, jnp.zeros(pad_shape, dtype=x_rest.dtype)], axis=1)
        else:
          x_rest_padded = x_rest
        T_padded = T_rest + pad_amount
        num_chunks = T_padded // K
        x_scannable = x_rest_padded.reshape(x_rest_padded.shape[0], num_chunks, K, x_rest_padded.shape[2],
                                            x_rest_padded.shape[3], x_rest_padded.shape[4])
        x_scannable = jnp.transpose(x_scannable, (1, 0, 2, 3, 4, 5))
        graphdef, state = nnx.split(self.decoder)

        @jax.named_scope("AutoencoderKLWan_decode_scan_body")
        def scan_fn(carry, chunk_in):
          current_feat_map = carry
          local_decoder = nnx.merge(graphdef, state)
          if spatial_sharding is not None:
            chunk_in = self.sharding.constrain(chunk_in, _SPATIAL_NAME, _SPATIAL_AXES)
          out_chunk, next_feat_map, _ = local_decoder(chunk_in, feat_cache=current_feat_map, feat_idx=0)
          if spatial_sharding is not None:
            out_chunk = self.sharding.constrain(out_chunk, _SPATIAL_NAME, _SPATIAL_AXES)
          next_feat_map = jax.tree_util.tree_map(
              lambda x: self.sharding.constrain(x, _SPATIAL_NAME, _SPATIAL_AXES)
              if spatial_sharding is not None and hasattr(x, "shape") and x.ndim == len(_SPATIAL_AXES) else x,
              next_feat_map,
          )
          return next_feat_map, out_chunk

        dec_feat_map, out_rest = jax.lax.scan(scan_fn, dec_feat_map, x_scannable)
        out_rest = jnp.transpose(out_rest, (1, 0, 2, 3, 4, 5))
        out_rest = out_rest.reshape(out_rest.shape[0], -1, out_rest.shape[3], out_rest.shape[4], out_rest.shape[5])
        out_rest = out_rest[:, : T_rest * self.temporal_downsample_factor, ...]
        out_list.append(out_rest)
      out = jnp.concatenate(out_list, axis=1)
      if spatial_sharding is not None:
        out = self.sharding.constrain(out, _SPATIAL_NAME, _SPATIAL_AXES)
    else:
      out = out_0
    feat_cache._feat_map = dec_feat_map
    out = jnp.clip(out, min=-1.0, max=1.0)
    if not return_dict:
      return (out,)
    return DecoderOutput(sample=out)

  @jax.named_scope("AutoencoderKLWan_decode")
  def decode(self, z: jax.Array, feat_cache: AutoencoderKLWanCache,
             return_dict: bool = True) -> Union[DecoderOutput, jax.Array]:
    if z.shape[-1] != self.z_dim:
      z = jnp.transpose(z, (0, 2, 3, 4, 1))
      assert z.shape[-1] == self.z_dim, f"Expected input shape (N, D, H, W, {self.z_dim}, got {z.shape}"
    decoded = self._decode(z, feat_cache).sample
    if not return_dict:
      return (decoded,)
    return DecoderOutput(sample=decoded)
