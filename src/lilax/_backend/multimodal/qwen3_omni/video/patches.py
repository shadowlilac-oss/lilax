"""Video frame resizing, normalization and patch flattening for Qwen3-Omni."""

import numpy as np
from PIL import Image

from lilax._backend.multimodal import utils as mm_utils
from lilax._backend.multimodal.qwen3_omni.video.sampling import FRAME_FACTOR
from lilax._backend.multimodal.qwen3_omni.vision import IMAGE_FACTOR, smart_resize

VIDEO_MIN_PIXELS = 128 * 28 * 28  # Minimum video pixels: 128 patches × patch_size².
VIDEO_MAX_PIXELS = 768 * 28 * 28  # Maximum video pixels: 768 patches × patch_size².
VIDEO_TOTAL_PIXELS = 128000 * 28 * 28 * 0.9  # Total video pixels budget: 128000 patches × patch_size² × 0.9.


def preprocess_video(video, config):
  """Preprocess the video for Qwen3-Omni model."""
  patch_size = config.patch_size_for_vit
  merge_size = config.spatial_merge_size_for_vit
  temporal_patch_size = config.temporal_patch_size_for_vit

  nframes, channel, height, width = video.shape
  max_pixels = max(min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / nframes * FRAME_FACTOR), int(VIDEO_MIN_PIXELS * 1.05))
  max_grid_h = getattr(config, "video_max_grid_h", None)
  max_grid_w = getattr(config, "video_max_grid_w", None)
  max_h_pixels = int(max_grid_h) * patch_size if max_grid_h is not None else None
  max_w_pixels = int(max_grid_w) * patch_size if max_grid_w is not None else None
  resized_height_1, resized_width_1 = smart_resize(
      height,
      width,
      factor=IMAGE_FACTOR,
      min_pixels=VIDEO_MIN_PIXELS,
      max_pixels=max_pixels,
  )

  # First resize - using PIL to match HuggingFace behavior
  resized_frames = []
  for frame_idx in range(nframes):
    # Convert from CHW to HWC for PIL
    frame = np.transpose(video[frame_idx], (1, 2, 0))
    pil_frame = Image.fromarray(frame.astype(np.uint8))
    pil_frame = pil_frame.resize((resized_width_1, resized_height_1), Image.BICUBIC)
    # Keep as float32 to preserve values outside [0, 255] from interpolation
    resized_frames.append(np.array(pil_frame, dtype=np.float32))

  resized_video = np.stack(resized_frames)

  # Second resize
  resized_height_2, resized_width_2 = smart_resize(
      resized_height_1,
      resized_width_1,
      factor=patch_size * merge_size,
      min_pixels=VIDEO_MIN_PIXELS,
      max_pixels=VIDEO_MAX_PIXELS,
      max_h_pixels=max_h_pixels,
      max_w_pixels=max_w_pixels,
  )

  # Second resize - process each channel separately to preserve float values
  final_frames = []
  for frame in resized_video:
    channels = []
    for c in range(frame.shape[2]):
      # Process each channel separately using PIL 'F' mode (float32)
      channel_data = frame[:, :, c]
      pil_frame = Image.fromarray(channel_data, mode="F")
      pil_frame = pil_frame.resize((resized_width_2, resized_height_2), Image.BICUBIC)
      channels.append(np.array(pil_frame, dtype=np.float32))
    final_frames.append(np.stack(channels, axis=2))

  resized_video = np.stack(final_frames)
  # Convert back to TCHW format
  resized_video = np.transpose(resized_video, (0, 3, 1, 2))

  resized_height, resized_width = resized_height_2, resized_width_2
  resized_video = mm_utils.normalize_images(
      resized_video,
      mean=127.5,
      std=127.5,
  )
  resized_video = np.expand_dims(resized_video, axis=0)  # Add batch dimension
  batch_size, grid_t, channel = resized_video.shape[:3]
  grid_t = grid_t // temporal_patch_size
  grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

  resized_video = np.reshape(
      resized_video,
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
  resized_video = np.permute_dims(resized_video, (0, 1, 4, 7, 5, 8, 3, 2, 6, 9))  # HWC to CHW
  resized_video = np.reshape(
      resized_video, (batch_size, grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
  )
  processed_grid = np.asarray([[grid_t, grid_h, grid_w]], dtype=np.int32)

  return resized_video[0, :, :], processed_grid
