"""Image preprocessing and shared vision resizing for Qwen3-Omni."""

import math

import numpy as np
from PIL import Image

from lilax._backend.multimodal import utils as mm_utils

IMAGE_MEAN = 127.5
IMAGE_STD = 127.5
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
    max_h_pixels: int | None = None,
    max_w_pixels: int | None = None,
):
  """Rescales the image so that the following conditions are met:

  1. Both dimensions (height and width) are divisible by 'factor'.

  2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

  3. The aspect ratio of the image is maintained as closely as possible.

  4. (Optional) Neither dimension exceeds max_h_pixels / max_w_pixels. When either
     cap is exceeded, both dimensions are scaled down proportionally and re-aligned
     to 'factor'.

  """
  if max(height, width) / min(height, width) > MAX_RATIO:
    raise ValueError(
        f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
    )
  h_bar = round(height / factor) * factor
  w_bar = round(width / factor) * factor
  if h_bar * w_bar > max_pixels:
    beta = math.sqrt((height * width) / max_pixels)
    h_bar = max(factor, math.floor(height / beta / factor) * factor)
    w_bar = max(factor, math.floor(width / beta / factor) * factor)
  elif h_bar * w_bar < min_pixels:
    beta = math.sqrt(min_pixels / (height * width))
    h_bar = math.ceil(height * beta / factor) * factor
    w_bar = math.ceil(width * beta / factor) * factor
  # Apply per-dimension pixel caps, scaling down proportionally if either is exceeded.
  if (max_h_pixels is not None and h_bar > max_h_pixels) or (max_w_pixels is not None and w_bar > max_w_pixels):
    cap_h = max_h_pixels if max_h_pixels is not None else h_bar
    cap_w = max_w_pixels if max_w_pixels is not None else w_bar
    scale = min(cap_h / h_bar, cap_w / w_bar)
    h_bar = min(max(factor, round(h_bar * scale / factor) * factor), (cap_h // factor) * factor)
    w_bar = min(max(factor, round(w_bar * scale / factor) * factor), (cap_w // factor) * factor)
  return h_bar, w_bar


def pre_process_qwen3_image(image: np.ndarray | list[np.ndarray], config, force_resize=None):
  """Performs a bi-linear resize (with anti-aliasing) and normalizes the image."""
  patch_size = config.patch_size_for_vit
  merge_size = config.spatial_merge_size_for_vit
  temporal_patch_size = config.temporal_patch_size_for_vit
  resample_method = Image.BICUBIC

  images_in = [image] if isinstance(image, np.ndarray) else image
  images_out = []
  grids_thw = []

  for img in images_in:
    pil_img = Image.fromarray(img)
    if force_resize is not None:
      resized_height_2, resized_width_2 = force_resize
      resized_img_pil = pil_img.resize((resized_width_2, resized_height_2), resample=resample_method)
    else:
      # Qwen3-Omni performs one resize during fetch_image and another resize before patchify.
      resized_height_1, resized_width_1 = smart_resize(
          height=img.shape[0],
          width=img.shape[1],
          factor=IMAGE_FACTOR,
          min_pixels=MIN_PIXELS,
          max_pixels=MAX_PIXELS,
      )
      pil_img = pil_img.resize((resized_width_1, resized_height_1))
      resized_height_2, resized_width_2 = smart_resize(
          height=resized_height_1,
          width=resized_width_1,
          factor=patch_size * merge_size,
          min_pixels=MIN_PIXELS,
          max_pixels=MAX_PIXELS,
      )
      resized_img_pil = pil_img.resize((resized_width_2, resized_height_2), resample=resample_method)
    resized_img_np = np.array(resized_img_pil).astype(np.float32)

    img_np = mm_utils.normalize_images(resized_img_np, mean=IMAGE_MEAN, std=IMAGE_STD)
    img_np = np.permute_dims(img_np, (2, 0, 1))  # HWC to NCHW
    img_np = np.expand_dims(img_np, axis=(0, 1))  # add batch dimension
    img_np = np.repeat(img_np, temporal_patch_size, axis=1)  # add temporal dimension

    grid_t = 2 // temporal_patch_size
    grid_h, grid_w = resized_height_2 // patch_size, resized_width_2 // patch_size
    batch_size = img_np.shape[0]
    channel = img_np.shape[2]

    img_np = np.reshape(
        img_np,
        (
            batch_size,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        ),
    )
    img_np = np.permute_dims(img_np, (0, 1, 4, 7, 5, 8, 3, 2, 6, 9))  # HWC to CHW
    img_np = np.reshape(
        img_np, (batch_size, grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
    )
    img_grid_thw = np.asarray([grid_t, grid_h, grid_w], dtype=np.int32)
    images_out.append(img_np)
    grids_thw.append(img_grid_thw)

  # Images are concatenated along the sequence dimension e.g. (seq1 + seq2, 1536)
  concatenated_images = np.concatenate([img[0] for img in images_out], axis=0)
  return concatenated_images, np.stack(grids_thw)
