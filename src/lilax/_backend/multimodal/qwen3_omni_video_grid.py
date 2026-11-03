"""Video grid padding and mask reduction for Qwen3-Omni."""

import numpy as np
import jax.numpy as jnp


def maybe_pad_video_values_to_max_grid(
    video_values: np.ndarray,
    video_grid_thw: np.ndarray,
    config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
  """Pad Qwen3-Omni video pixels to configured static grid limits when enabled.

  Args:
    video_values: Video pixels of shape (batch, channels, T*tps, H*patch, W*patch).
    video_grid_thw: Actual video grid with shape (1, 3), in Qwen grid units.
    config: Config carrying video_max_grid_t/h/w and ViT patch sizes.

  Returns:
    Tuple of:
    - padded video pixels, or the input when no max grid is configured
    - input grid_thw
    - pixel-level mask of shape (batch, 1, max_T*tps, max_H*patch, max_W*patch), or None
  """
  max_grid = (
      getattr(config, "video_max_grid_t", None),
      getattr(config, "video_max_grid_h", None),
      getattr(config, "video_max_grid_w", None),
  )
  if all(dim is None for dim in max_grid):
    return video_values, video_grid_thw, None
  if any(dim is None for dim in max_grid):
    raise ValueError("video_max_grid_t, video_max_grid_h, and video_max_grid_w must be set together.")
  if video_values.ndim != 5:
    raise ValueError(f"video_values must have shape (batch, channels, time, height, width), got {video_values.shape}.")

  max_t, max_h, max_w = (int(dim) for dim in max_grid)  # pyrefly: ignore[bad-argument-type]
  actual_t, actual_h, actual_w = (int(dim) for dim in video_grid_thw[0])
  if actual_t > max_t or actual_h > max_h or actual_w > max_w:
    raise ValueError(
        f"video grid {video_grid_thw[0].tolist()} exceeds max grid {(max_t, max_h, max_w)}. "
        "Scale or resize the video before padding."
    )

  temporal_patch_size = config.temporal_patch_size_for_vit
  patch_size = config.patch_size_for_vit
  max_t_px = max_t * temporal_patch_size
  max_h_px = max_h * patch_size
  max_w_px = max_w * patch_size

  padded_video_values = np.zeros(
      (video_values.shape[0], video_values.shape[1], max_t_px, max_h_px, max_w_px),
      dtype=video_values.dtype,
  )
  patch_elements = video_values.shape[1] * temporal_patch_size * patch_size * patch_size
  valid_patches = actual_t * actual_h * actual_w
  padded_video_values.reshape((video_values.shape[0], -1, patch_elements))[:, :valid_patches] = video_values.reshape(
      (video_values.shape[0], valid_patches, patch_elements)
  )

  video_mask = np.zeros((video_values.shape[0], 1, max_t_px, max_h_px, max_w_px), dtype=np.int32)
  mask_patch_elements = temporal_patch_size * patch_size * patch_size
  video_mask.reshape((video_values.shape[0], -1, mask_patch_elements))[:, :valid_patches] = 1

  return padded_video_values, video_grid_thw, video_mask


def downsample_video_mask_to_tokens(video_mask, config):
  """Reduces a Qwen3 pixel mask to the post-projector video-token mask.

  Example: with patch size `(2, 16, 16)`, spatial merge size `2`, and padded
  grid `(3, 4, 4)`, a pixel mask `[1, 1, 6, 64, 64]` becomes 48 patch-mask
  values and then 12 projected-token-mask values. A valid grid `(2, 2, 4)`
  marks the first `2 * 2 * 4 / 2**2 = 4` projected tokens as valid.
  """
  if video_mask is None:
    return None

  patch_elements = config.temporal_patch_size_for_vit * config.patch_size_for_vit**2
  patch_mask = video_mask.reshape(video_mask.shape[0], -1, patch_elements).max(axis=-1)
  merge_elements = config.spatial_merge_size_for_vit**2
  if patch_mask.shape[-1] % merge_elements:
    raise ValueError(
        f"Video patch-mask length {patch_mask.shape[-1]} must be divisible by spatial merge area {merge_elements}."
    )
  return patch_mask.reshape(patch_mask.shape[0], -1, merge_elements).max(axis=-1).astype(jnp.int32)
