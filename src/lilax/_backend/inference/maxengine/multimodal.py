"""Prepare multimodal inputs for prefill."""

import jax
import jax.numpy as jnp


def prepare_images(images, image_masks):
  if images.ndim == 3:
    images = images[jnp.newaxis, jnp.newaxis, ...]
    image_masks = image_masks[jnp.newaxis, jnp.newaxis, ...] if image_masks is not None else None
  elif images.ndim == 4:
    images = images[jnp.newaxis, ...]
    image_masks = image_masks[jnp.newaxis, ...] if image_masks is not None else None
  return images, image_masks


def prepare_video_grid(video_grid_thw):
  flat_video_grid = jax.device_get(video_grid_thw).reshape(-1)
  if flat_video_grid.size != 3:
    raise ValueError(f"Decode currently supports one video grid with 3 values, got shape {video_grid_thw.shape}.")
  return tuple(int(dim) for dim in flat_video_grid)
