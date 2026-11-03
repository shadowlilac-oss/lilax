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

"""Patchification of Weaver diffusion latents to and from generation tokens."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def patchify_latents(latents: jax.Array, patch_size: int = 2) -> jax.Array:
  """Patchify latents from [B, C, T, H, W] to [B, S_gen, p*p*C]."""
  if latents.ndim != 5:
    raise ValueError(f"latents must be a 5D array [B, C, T, H, W], got shape {latents.shape}.")
  bsz, c_lat, t_lat, h_lat, w_lat = latents.shape
  if h_lat % patch_size != 0 or w_lat % patch_size != 0:
    raise ValueError(f"Latent spatial dimensions ({h_lat}, {w_lat}) must be divisible by patch_size ({patch_size}).")
  h_patch = h_lat // patch_size
  w_patch = w_lat // patch_size
  s_gen = t_lat * h_patch * w_patch
  patch_dim = patch_size * patch_size * c_lat
  reshaped = latents.reshape(bsz, c_lat, t_lat, h_patch, patch_size, w_patch, patch_size)
  permuted = jnp.einsum("bcthpwq->bthwpqc", reshaped)
  return permuted.reshape(bsz, s_gen, patch_dim)


def unpatchify_latents(
    patches: jax.Array, t_lat: int, h_lat: int, w_lat: int, *, patch_size: int = 2, latent_channels: int = 48,
) -> jax.Array:
  """Unpatchify generation tokens to [B, C, T, H, W]."""
  if h_lat % patch_size != 0 or w_lat % patch_size != 0:
    raise ValueError(f"Latent spatial dimensions ({h_lat}, {w_lat}) must be divisible by patch_size ({patch_size}).")
  h_patch = h_lat // patch_size
  w_patch = w_lat // patch_size
  s_gen = t_lat * h_patch * w_patch
  patch_dim = patch_size * patch_size * latent_channels
  if patches.shape[-1] != patch_dim:
    raise ValueError(f"Expected patch feature dim {patch_dim}, got {patches.shape[-1]}.")
  if patches.ndim == 2:
    if patches.shape[0] % s_gen != 0:
      raise ValueError(f"Total patch tokens ({patches.shape[0]}) must be divisible by s_gen ({s_gen}).")
    bsz = patches.shape[0] // s_gen
  elif patches.ndim == 3:
    bsz = patches.shape[0]
    if patches.shape[1] != s_gen:
      raise ValueError(f"Expected s_gen={s_gen} along axis 1, got {patches.shape[1]}.")
  else:
    raise ValueError(f"patches must be 2D or 3D, got shape {patches.shape}.")
  reshaped = patches.reshape(bsz, t_lat, h_patch, w_patch, patch_size, patch_size, latent_channels)
  permuted = jnp.einsum("bthwpqc->bcthpwq", reshaped)
  return permuted.reshape(bsz, latent_channels, t_lat, h_lat, w_lat)
