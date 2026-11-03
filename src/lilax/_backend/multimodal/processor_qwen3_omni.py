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

"""Qwen3-Omni-specific preprocessing utilities for multimodal features.

Original implementation from HuggingFace: Qwen/Qwen3-Omni-30B-A3B-Instruct.
"""

from dataclasses import dataclass

import numpy as np

from lilax._backend.multimodal import utils as mm_utils
from lilax._backend.multimodal.qwen3_omni.audio import (
    SAMPLE_RATE,
    _get_feat_extract_output_lengths,
    pre_process_audio_qwen3_omni,
)
from lilax._backend.multimodal.qwen3_omni.tokens import QwenTokens
from lilax._backend.multimodal.qwen3_omni.video.grid import maybe_pad_video_values_to_max_grid
from lilax._backend.multimodal.qwen3_omni.video.loading import _read_video_decord, is_video_file, load_training_video
from lilax._backend.multimodal.qwen3_omni.video.patches import preprocess_video
from lilax._backend.multimodal.qwen3_omni.video.sampling import get_video_second_per_grid
from lilax._backend.multimodal.qwen3_omni.vision import pre_process_qwen3_image

QWEN3_TEMPORAL_PATCH_SIZE = 2
QWEN3_OMNI_IMAGE_SIZE = 768


@dataclass
class Qwen3OmniPreprocessorOutput(mm_utils.PreprocessorOutput):
  """Holds the output of Qwen3-Omni image preprocessor.

  Attributes:
    Inherited from `mm_utils.PreprocessorOutput`.
  """

  # Image attributes.
  num_images: int = 0
  pixel_values: None | np.ndarray = None
  pixel_grid_thw: None | np.ndarray = None
  # Video attributes.
  num_videos: int = 0
  video_values: None | np.ndarray = None
  video_grid_thw: None | np.ndarray = None
  video_second_per_grid: None | np.ndarray = None
  video_mask: None | np.ndarray = None
  # Audio attributes.
  num_audios: int = 0
  audio_values: None | np.ndarray = None
  audio_mask: None | np.ndarray = None
  audio_lengths: None | np.ndarray = None


def preprocess_mm_data_qwen3_omni_for_training(images, config):
  """Preprocesses image(s) or video for Qwen3-Omni SFT training using model config constants."""
  is_video = False
  if isinstance(images, str) and is_video_file(images):
    is_video = True
  elif isinstance(images, list) and len(images) > 0 and isinstance(images[0], str) and is_video_file(images[0]):
    is_video = True

  if is_video:
    video_array, sample_fps = load_training_video(images, config)
    video_processed, video_grid_thw = preprocess_video(video_array, config)
    video_values = np.reshape(
        video_processed,
        (
            1,
            config.num_channels_for_vit,
            config.temporal_patch_size_for_vit * video_grid_thw[0, 0],
            config.patch_size_for_vit * video_grid_thw[0, 1],
            config.patch_size_for_vit * video_grid_thw[0, 2],
        ),
    )
    video_values, video_grid_thw, video_mask = maybe_pad_video_values_to_max_grid(video_values, video_grid_thw, config)
    return Qwen3OmniPreprocessorOutput(
        num_videos=1,
        video_values=video_values,
        video_grid_thw=video_grid_thw,
        video_second_per_grid=np.asarray(
            [get_video_second_per_grid(sample_fps, config.temporal_patch_size_for_vit)], dtype=np.float32
        ),
        video_mask=video_mask,
    )
  else:
    images_in = [images] if isinstance(images, np.ndarray) else images
    if config.image_size_for_vit is None:
      force_resize = None
    elif isinstance(config.image_size_for_vit, (list, tuple)):
      force_resize = tuple(config.image_size_for_vit)
    else:
      force_resize = (config.image_size_for_vit, config.image_size_for_vit)
    pixel_values, pixel_grid_thw = pre_process_qwen3_image(images_in, config, force_resize=force_resize)
    pixel_values = np.reshape(
        pixel_values,
        (
            len(images_in),
            config.num_channels_for_vit,
            config.temporal_patch_size_for_vit * pixel_grid_thw[0, 0],
            config.patch_size_for_vit * pixel_grid_thw[0, 1],
            config.patch_size_for_vit * pixel_grid_thw[0, 2],
        ),
    )
    return Qwen3OmniPreprocessorOutput(
        num_images=len(images_in),
        pixel_values=pixel_values,
        pixel_grid_thw=pixel_grid_thw,
    )


def preprocess_mm_data_qwen3_omni(config):
  """Placeholder for multimodal data preprocessing."""
  processor_outputs = Qwen3OmniPreprocessorOutput()

  if config.image_path:
    images = [mm_utils.load_image_from_path(p) for p in config.image_path.split(",")]
    pixel_values, pixel_grid_thw = pre_process_qwen3_image(images, config)
    # Reshape to align with current Qwen3OmniMoeVisionEncoder, which carries grid_thw information
    pixel_values = np.reshape(
        pixel_values,
        (
            1,
            config.num_channels_for_vit,
            config.temporal_patch_size_for_vit * pixel_grid_thw[0, 0],
            config.patch_size_for_vit * pixel_grid_thw[0, 1],
            config.patch_size_for_vit * pixel_grid_thw[0, 2],
        ),
    )
    processor_outputs.pixel_values = pixel_values
    processor_outputs.pixel_grid_thw = pixel_grid_thw
    processor_outputs.num_images = len(images)

  if config.video_path:
    video_array, sample_fps = _read_video_decord(config.video_path)
    video_processed, video_grid_thw = preprocess_video(video_array, config)
    video_values = np.reshape(
        video_processed,
        (
            1,
            config.num_channels_for_vit,
            config.temporal_patch_size_for_vit * video_grid_thw[0, 0],
            config.patch_size_for_vit * video_grid_thw[0, 1],
            config.patch_size_for_vit * video_grid_thw[0, 2],
        ),
    )
    video_values, video_grid_thw, video_mask = maybe_pad_video_values_to_max_grid(video_values, video_grid_thw, config)
    processor_outputs.video_values = video_values
    processor_outputs.video_grid_thw = video_grid_thw
    processor_outputs.video_mask = video_mask
    processor_outputs.video_second_per_grid = np.asarray(
        [get_video_second_per_grid(sample_fps, config.temporal_patch_size_for_vit)], dtype=np.float32
    )
    processor_outputs.num_videos = 1  # Only one video for now.

  if config.video_path and config.use_audio_in_video:
    # TODO(hengtaoguo): add support for separate audio files. Now only extract audio from video files.
    mt_audio = mm_utils.load_audio(config.video_path, sample_rate=SAMPLE_RATE)
    mt_audio, mt_audio_mask = pre_process_audio_qwen3_omni(mt_audio)
    processor_outputs.audio_values = mt_audio
    processor_outputs.audio_mask = mt_audio_mask
    # Compute audio_lengths from audio_mask
    audio_mask_sum = np.sum(mt_audio_mask, axis=-1)
    audio_lengths = _get_feat_extract_output_lengths(audio_mask_sum)
    processor_outputs.audio_lengths = np.array(audio_lengths, dtype=np.int32)

  return processor_outputs


def add_extra_tokens_for_qwen3_omni(tokens, config, processor_output):
  """Add extra tokens for Qwen3-Omni multimodal sequences.

  Expands special tokens (<|image_pad|>, <|video_pad|>, <|audio_pad|>) into
  the correct number of placeholder tokens based on grid dimensions and merge size.

  For audio-in-video mode, interleaves audio and video tokens based on temporal ordering.

  Args:
    tokens: Input token sequence (1D array or list).
    image_grid_thw: Image dimensions (num_images, 3) with [temporal, height, width].
    video_grid_thw: Video dimensions (num_videos, 3) with [temporal, height, width].
    audio_lengths: Pre-computed audio token counts (num_audios,).
    spatial_merge_size: Number of patches merged spatially (e.g., 2 for 2x2→1).
    use_audio_in_video: If True, interleave audio and video tokens.
    second_per_grids: Time interval per temporal grid (num_videos,).
    position_id_per_seconds: Temporal granularity (tokens per second).

  Returns:
    Expanded token sequence with correct number of image/video/audio tokens.
  """
  image_grid_thw = getattr(processor_output, "pixel_grid_thw", None)
  video_grid_thw = getattr(processor_output, "video_grid_thw", None)
  audio_lengths = getattr(processor_output, "audio_lengths", None)
  second_per_grids = getattr(processor_output, "video_second_per_grid", None)
  spatial_merge_size = config.spatial_merge_size_for_vit
  position_id_per_seconds = config.position_id_per_seconds

  if not isinstance(tokens, np.ndarray):
    tokens = np.asarray(tokens)

  tokens = tokens.flatten()  # Ensure 1D

  # Get dynamic tokens based on config
  qwen_tokens = QwenTokens(config)

  # Merge lengths for computing number of tokens
  merge_length = spatial_merge_size**2

  # Convert to list for easier manipulation
  token_list = tokens.tolist()
  new_tokens = []

  image_idx = 0
  video_idx = 0
  audio_idx = 0

  i = 0
  while i < len(token_list):
    token = token_list[i]

    # Handle image tokens
    if token == qwen_tokens.image_pad and image_grid_thw is not None and image_idx < len(image_grid_thw):
      grid = image_grid_thw[image_idx]
      num_image_tokens = int((grid[0] * grid[1] * grid[2]) // merge_length)
      new_tokens.extend([qwen_tokens.image_pad] * num_image_tokens)
      image_idx += 1

    # Handle audio-in-video: <|vision_start|><|video_pad|><|vision_end|>
    elif (
        config.use_audio_in_video
        and token == qwen_tokens.vision_start
        and i + 2 < len(token_list)
        and token_list[i + 1] == qwen_tokens.video_pad
        and token_list[i + 2] == qwen_tokens.vision_end
        and video_grid_thw is not None
        and video_idx < len(video_grid_thw)
    ):

      if audio_lengths is None or audio_idx >= len(audio_lengths):
        raise ValueError("audio_lengths required for audio-in-video mode")
      if second_per_grids is None or video_idx >= len(second_per_grids):
        raise ValueError("second_per_grids required for audio-in-video mode")

      audio_length = audio_lengths[audio_idx]
      audio_token_indices = np.arange(audio_length)

      curr_video_grid = video_grid_thw[video_idx]
      height = curr_video_grid[1] // spatial_merge_size
      width = curr_video_grid[2] // spatial_merge_size
      num_frames = curr_video_grid[0]

      video_token_indices = np.arange(num_frames).reshape(-1, 1, 1)
      video_token_indices = np.broadcast_to(video_token_indices, (num_frames, height, width)).flatten()
      video_token_indices = video_token_indices * second_per_grids[video_idx] * position_id_per_seconds

      new_tokens.append(qwen_tokens.vision_start)
      new_tokens.append(qwen_tokens.audio_start)

      video_data_idx = 0
      audio_data_idx = 0

      while video_data_idx < len(video_token_indices) and audio_data_idx < len(audio_token_indices):
        if video_token_indices[video_data_idx] <= audio_token_indices[audio_data_idx]:
          new_tokens.append(qwen_tokens.video_pad)
          video_data_idx += 1
        else:
          new_tokens.append(qwen_tokens.audio_pad)
          audio_data_idx += 1

      while video_data_idx < len(video_token_indices):
        new_tokens.append(qwen_tokens.video_pad)
        video_data_idx += 1

      while audio_data_idx < len(audio_token_indices):
        new_tokens.append(qwen_tokens.audio_pad)
        audio_data_idx += 1

      new_tokens.append(qwen_tokens.audio_end)
      new_tokens.append(qwen_tokens.vision_end)

      video_idx += 1
      audio_idx += 1
      i += 2

    # Handle video tokens (without audio-in-video)
    elif token == qwen_tokens.video_pad and video_grid_thw is not None and video_idx < len(video_grid_thw):
      grid = video_grid_thw[video_idx]
      num_video_tokens = int((grid[0] * grid[1] * grid[2]) // merge_length)
      new_tokens.extend([qwen_tokens.video_pad] * num_video_tokens)
      video_idx += 1

    # Handle audio tokens (standalone, not in video)
    elif token == qwen_tokens.audio_pad and audio_lengths is not None and audio_idx < len(audio_lengths):
      num_audio_tokens = int(audio_lengths[audio_idx])
      new_tokens.extend([qwen_tokens.audio_pad] * num_audio_tokens)
      audio_idx += 1

    # All other tokens pass through unchanged
    else:
      new_tokens.append(token)

    i += 1

  return np.array(new_tokens, dtype=np.int32)


def get_dummy_image_shape_for_init_qwen3_omni(batch_size):
  """Return the shape of the dummy image for Qwen3-Omni model's initialization."""
  image_shape = (
      batch_size,
      mm_utils.NUM_IMAGE_CHANNELS,
      QWEN3_TEMPORAL_PATCH_SIZE,
      QWEN3_OMNI_IMAGE_SIZE,  # image_size_for_vit (height)
      QWEN3_OMNI_IMAGE_SIZE,  # video_num_frames
  )
  return image_shape


def get_dummy_audio_shape_for_init_qwen3_omni(config):
  """Return the shape of the dummy audio for Qwen3-Omni model's initialization."""
  # Audio shape: (batch, num_mel_bins, audio_length)
  # audio_length must be divisible by n_window * 2
  chunk_size = config.n_window_for_audio * 2
  audio_length = chunk_size * 4  # 4 chunks
  audio_shape = (config.micro_batch_size_to_train_on, config.num_mel_bins_for_audio, audio_length)
  return audio_shape


def reformat_prompt_qwen3_omni(
    prompt,
    image_placeholder="<|image|>",
    num_images=0,
    video_placeholder="<|video|>",
    num_videos=0,
):
  """Reformat the prompt for Qwen3-Omni model."""
  # Qwen3-Omni vision format: <|vision_start|><|image_pad|><|vision_end|>
  # Qwen3-Omni mm token order: image_pad, video_pad, audio_pad (standalone audios), then text tokens.
  # use_audio_in_video mode: such audio tokens are interleaved within video tokens.
  qwen3_image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
  qwen3_video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"

  if video_placeholder in prompt:
    prompt = prompt.replace(video_placeholder, qwen3_video_placeholder)
  video_placeholder_count = prompt.count(qwen3_video_placeholder)
  if video_placeholder_count < num_videos:
    prompt = qwen3_video_placeholder * (num_videos - video_placeholder_count) + prompt

  if image_placeholder in prompt:
    prompt = prompt.replace(image_placeholder, qwen3_image_placeholder)
  image_placeholder_count = prompt.count(qwen3_image_placeholder)
  if image_placeholder_count < num_images:
    prompt = qwen3_image_placeholder * (num_images - image_placeholder_count) + prompt

  # Qwen chat template
  formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
  return formatted_prompt


def get_mm_offsets_qwen3_omni(config, processor_output):
  """Calculate the token offsets for multimodal tokens in Qwen3-Omni model."""
  # Calculate token expansion for Qwen3-Omni multimodal inputs
  if processor_output is None:
    return 0

  total_offset = 0
  spatial_merge_size = config.spatial_merge_size_for_vit  # Default 2 for Qwen3-Omni
  merge_length = spatial_merge_size**2

  # Image tokens: <|image_pad|> expands to multiple image tokens
  if processor_output.pixel_grid_thw is not None:
    image_grid_thw = processor_output.pixel_grid_thw
    for grid in image_grid_thw:
      num_image_tokens = int((grid[0] * grid[1] * grid[2]) // merge_length)
      total_offset += num_image_tokens - 1  # -1 for the original <|image_pad|> token

  # Video tokens: <|video_pad|> expands to multiple video tokens
  if processor_output.video_grid_thw is not None:
    video_grid_thw = processor_output.video_grid_thw
    for grid in video_grid_thw:
      num_video_tokens = int((grid[0] * grid[1] * grid[2]) // merge_length)
      total_offset += num_video_tokens - 1  # -1 for the original <|video_pad|> token

  # Audio tokens: <|audio_pad|> expands based on audio_lengths
  if processor_output.audio_lengths is not None:
    audio_lengths = processor_output.audio_lengths
    for audio_len in audio_lengths:
      if getattr(config, "use_audio_in_video", False):
        total_offset += int(audio_len) + 2  # +2 for <|audio_start|> and <|audio_end|>, no <|audio_pad|> to remove
      else:
        total_offset += int(audio_len) - 1  # -1 for the original <|audio_pad|> token

  return total_offset
