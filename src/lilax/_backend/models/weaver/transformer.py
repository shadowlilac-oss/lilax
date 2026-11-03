# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``WeaverOmniTransformer`` joint multimodal diffusion backbone.

Integrates ``embed_tokens``, ``proj_in``, ``time_embedder``, stacked
``WeaverMoTDecoderLayer`` blocks (``scan_layers=False`` and
``scan_layers=True``), ``norm``, ``norm_moe_gen``, and ``proj_out``.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import Any, Literal, overload

from flax import nnx
import jax
from jax import sharding
import jax.numpy as jnp
from lilax._backend.layers import nnx_scan
from lilax._backend.models.weaver.config import WeaverConfig
from lilax._backend.models.weaver.decoder_layer import WeaverMoTDecoderLayer
from lilax._backend.models.weaver.latents import patchify_latents, unpatchify_latents
from lilax._backend.models.weaver.leaf_layers import _make_embed, _make_linear, _make_rmsnorm
from lilax._backend.models.weaver.packing import WeaverPackingMetadata, build_weaver_packing_metadata
from lilax._backend.models.weaver.rotary import build_weaver_3d_position_ids, compute_3d_mrope_cos_sin
from lilax._backend.models.weaver.streams import reinterleave_streams
from lilax._backend.models.weaver.timestep import WeaverTimeEmbedder


class WeaverOmniTransformer(nnx.Module):
  """Complete Weaver joint Mixture-of-Transformers diffusion backbone."""

  def __init__(
      self, config: WeaverConfig | Any | int = 4096, *, head_dim: int | None = None,
      num_attention_heads: int | None = None, num_key_value_heads: int | None = None,
      intermediate_size: int | None = None, num_hidden_layers: int | None = None, vocab_size: int | None = None,
      latent_channels: int | None = None, patch_size: int | None = None, time_embed_in_channels: int | None = None,
      timestep_scale: float | None = None, timestep_max_period: int | None = None, rms_norm_eps: float | None = None,
      hidden_act: str | None = None, attention_bias: bool | None = None, qk_norm_for_text: bool | None = None,
      qk_norm_for_diffusion: bool | None = None, use_und_k_norm_for_gen: bool | None = None,
      attention_kernel: str | None = None, use_ragged_ops: bool | None = None, block_q: int | None = None,
      block_kv: int | None = None, mrope_section: tuple[int, int, int] | None = None,
      rope_theta: float | None = None, scan_layers: bool | None = None, param_scan_axis: int | None = None,
      remat_policy: str | None = None, dtype: Any | None = None, weight_dtype: Any | None = None,
      sharding_hook: Callable[[Any, str], Any] | None = None, mesh: sharding.Mesh | None = None,
      kernel_init: Callable[..., jax.Array] | None = None, rngs: nnx.Rngs,
  ):
    del mesh
    if isinstance(config, WeaverConfig):
      cfg = dataclasses.replace(config)
    elif isinstance(config, int):
      cfg = WeaverConfig(hidden_size=config)
    else:
      cfg = WeaverConfig.from_maxtext_config(config)
    local_vars = locals()
    overrides = {
        f.name: local_vars[f.name]
        for f in dataclasses.fields(WeaverConfig)
        if f.name in local_vars and local_vars[f.name] is not None
    }
    if overrides:
      cfg = dataclasses.replace(cfg, **overrides)

    self.config = cfg
    self.hidden_size = cfg.hidden_size
    self.head_dim = cfg.head_dim
    self.num_attention_heads = cfg.num_attention_heads
    self.num_key_value_heads = cfg.num_key_value_heads
    self.intermediate_size = cfg.intermediate_size
    self.num_hidden_layers = cfg.num_hidden_layers
    self.vocab_size = cfg.vocab_size
    self.latent_channels = cfg.latent_channels
    self.patch_size = cfg.patch_size
    self.patch_dim = cfg.patch_size * cfg.patch_size * cfg.latent_channels
    self.time_embed_in_channels = cfg.time_embed_in_channels
    self.timestep_scale = cfg.timestep_scale
    self.timestep_max_period = cfg.timestep_max_period
    self.rms_norm_eps = cfg.rms_norm_eps
    self.hidden_act = cfg.hidden_act
    self.mrope_section = cfg.mrope_section
    self.rope_theta = cfg.rope_theta
    self.scan_layers = cfg.scan_layers
    self.param_scan_axis = cfg.param_scan_axis
    self.remat_policy = cfg.remat_policy
    self.use_ragged_ops = cfg.use_ragged_ops
    self.dtype = cfg.dtype
    self.weight_dtype = cfg.weight_dtype
    self.sharding_hook = sharding_hook or (lambda x, name: x)
    self.embed_tokens = _make_embed(
        num_embeddings=cfg.vocab_size, features=cfg.hidden_size, dtype=cfg.dtype, weight_dtype=cfg.weight_dtype,
        kernel_axes=cfg.embed_kernel_axes, sharding_hook=sharding_hook, hook_name="embed_tokens_kernel_init", rngs=rngs,
    )
    self.proj_in = _make_linear(
        self.patch_dim, cfg.hidden_size, use_bias=True, dtype=cfg.dtype, weight_dtype=cfg.weight_dtype,
        kernel_init=kernel_init, kernel_axes=cfg.proj_in_kernel_axes, sharding_hook=sharding_hook,
        hook_name="proj_in_kernel_init", rngs=rngs,
    )
    self.time_embedder = WeaverTimeEmbedder(
        in_channels=cfg.time_embed_in_channels, time_embed_dim=cfg.hidden_size, timestep_scale=cfg.timestep_scale,
        max_period=cfg.timestep_max_period, in_kernel_axes=cfg.time_mlp_in_kernel_axes,
        out_kernel_axes=cfg.time_mlp_out_kernel_axes, sharding_hook=sharding_hook, dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype, kernel_init=kernel_init, rngs=rngs,
    )
    if cfg.scan_layers:
      self.layers = None
      self.scanned_layers = nnx_scan.create_scanned_layers(
          lambda layer_rngs: WeaverMoTDecoderLayer(cfg, sharding_hook=sharding_hook, kernel_init=kernel_init, rngs=layer_rngs),
          length=cfg.num_hidden_layers, param_scan_axis=cfg.param_scan_axis, metadata_axis_name="layers", rngs=rngs,
      )
    else:
      self.layers = nnx.List(
          [WeaverMoTDecoderLayer(cfg, sharding_hook=sharding_hook, kernel_init=kernel_init, rngs=rngs)
           for _ in range(cfg.num_hidden_layers)]
      )
      self.scanned_layers = None
    self.norm = _make_rmsnorm(
        cfg.hidden_size, epsilon=cfg.rms_norm_eps, dtype=cfg.dtype, weight_dtype=cfg.weight_dtype,
        kernel_axes=cfg.norm_kernel_axes, sharding_hook=sharding_hook, hook_name="norm_scale_init", rngs=rngs,
    )
    self.norm_moe_gen = _make_rmsnorm(
        cfg.hidden_size, epsilon=cfg.rms_norm_eps, dtype=cfg.dtype, weight_dtype=cfg.weight_dtype,
        kernel_axes=cfg.norm_kernel_axes, sharding_hook=sharding_hook, hook_name="norm_moe_gen_scale_init", rngs=rngs,
    )
    self.proj_out = _make_linear(
        cfg.hidden_size, self.patch_dim, use_bias=True, dtype=cfg.dtype, weight_dtype=cfg.weight_dtype,
        kernel_init=kernel_init, kernel_axes=cfg.proj_out_kernel_axes, sharding_hook=sharding_hook,
        hook_name="proj_out_kernel_init", rngs=rngs,
    )

  @classmethod
  def from_config(
      cls, config: WeaverConfig | Any, *, rngs: nnx.Rngs, mesh: sharding.Mesh | None = None,
      sharding_hook: Callable[[Any, str], Any] | None = None,
      kernel_init: Callable[..., jax.Array] | None = None, **overrides: Any,
  ) -> WeaverOmniTransformer:
    return cls(config, mesh=mesh, sharding_hook=sharding_hook, kernel_init=kernel_init, rngs=rngs, **overrides)

  @property
  def vae2llm(self) -> nnx.Linear:
    return self.proj_in

  @property
  def llm2vae(self) -> nnx.Linear:
    return self.proj_out

  @overload
  def __call__(
      self, input_ids: jax.Array, latents: jax.Array, timesteps: jax.Array, *,
      position_ids: jax.Array | None = None, metadata: WeaverPackingMetadata | None = None,
      return_hidden_states: Literal[False] = False,
  ) -> jax.Array: ...

  @overload
  def __call__(
      self, input_ids: jax.Array, latents: jax.Array, timesteps: jax.Array, *,
      position_ids: jax.Array | None = None, metadata: WeaverPackingMetadata | None = None,
      return_hidden_states: Literal[True],
  ) -> tuple[jax.Array, jax.Array]: ...

  def __call__(
      self, input_ids: jax.Array, latents: jax.Array, timesteps: jax.Array, *,
      position_ids: jax.Array | None = None, metadata: WeaverPackingMetadata | None = None,
      return_hidden_states: bool = False,
  ) -> jax.Array | tuple[jax.Array, jax.Array]:
    if input_ids.ndim != 2:
      raise ValueError(f"input_ids must be 2D [B, S_und], got shape {input_ids.shape}.")
    if latents.ndim != 5:
      raise ValueError(f"latents must be 5D [B, C, T, H, W], got shape {latents.shape}.")
    bsz, s_und = input_ids.shape
    bsz_lat, c_lat, t_lat, h_lat, w_lat = latents.shape
    if bsz != bsz_lat:
      raise ValueError(f"Batch size mismatch between input_ids ({bsz}) and latents ({bsz_lat}).")
    if c_lat != self.latent_channels:
      raise ValueError(f"Expected latent_channels={self.latent_channels}, got {c_lat}.")
    p = self.patch_size
    h_patch = h_lat // p
    w_patch = w_lat // p
    s_gen = t_lat * h_patch * w_patch
    text_embeds = self.embed_tokens(input_ids).astype(self.dtype)
    und_seq = text_embeds.reshape(bsz * s_und, self.hidden_size)
    patches = patchify_latents(latents.astype(self.dtype), patch_size=p)
    vision_embeds = self.proj_in(patches)
    time_embeds = self.time_embedder(timesteps).astype(self.dtype)
    if time_embeds.ndim == 2:
      time_embeds = time_embeds[:, None, :]
    elif time_embeds.ndim == 3 and time_embeds.shape[1] == t_lat and s_gen != t_lat:
      time_embeds = jnp.repeat(time_embeds, h_patch * w_patch, axis=1)
    gen_tokens = vision_embeds + time_embeds
    gen_seq = gen_tokens.reshape(bsz * s_gen, self.hidden_size)
    if metadata is None:
      metadata = build_weaver_packing_metadata([s_und] * bsz, [s_gen] * bsz)
    if position_ids is None:
      pos_3d = build_weaver_3d_position_ids(batch_size=bsz, s_und=s_und, t_lat=t_lat, h_patch=h_patch, w_patch=w_patch)
    else:
      if position_ids.ndim == 2 and position_ids.shape[0] == 3 and position_ids.shape[1] == metadata.total_tokens:
        pos_3d = position_ids.T
      else:
        pos_3d = position_ids
    cos, sin = compute_3d_mrope_cos_sin(
        pos_3d, head_dim=self.head_dim, mrope_section=self.mrope_section, rope_theta=self.rope_theta,
    )
    cos = cos.astype(self.dtype)
    sin = sin.astype(self.dtype)
    if self.scan_layers:
      assert self.scanned_layers is not None
      remat = self.remat_policy not in ("none", "")
      und_seq, gen_seq = nnx_scan.apply_scanned_layers(
          self.scanned_layers, (und_seq, gen_seq), length=self.num_hidden_layers,
          param_scan_axis=self.param_scan_axis, apply_fn=lambda layer, carry: layer(carry[0], carry[1], metadata, cos, sin),
          remat=remat,
      )
    else:
      assert self.layers is not None
      for layer in self.layers:
        und_seq, gen_seq = layer(und_seq, gen_seq, metadata, cos, sin)
    und_seq = self.norm(und_seq)
    gen_seq = self.norm_moe_gen(gen_seq)
    gen_patches = self.proj_out(gen_seq)
    preds_vision = unpatchify_latents(
        gen_patches, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat, patch_size=p, latent_channels=c_lat,
    )
    if return_hidden_states:
      last_hidden_state = reinterleave_streams(
          und_tokens=und_seq, gen_tokens=gen_seq, packed_und_token_indexes=metadata.packed_und_token_indexes,
          packed_gen_token_indexes=metadata.packed_gen_token_indexes, total_tokens=metadata.total_tokens,
          use_ragged_ops=self.use_ragged_ops,
      )
      return preds_vision, last_hidden_state
    return preds_vision
