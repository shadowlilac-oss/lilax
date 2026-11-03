"""Multimodal 3D RoPE position IDs for Qwen3-Omni."""

import numpy as np
import jax

from lilax._backend.multimodal.qwen3_omni.audio import _get_feat_extract_output_lengths
from lilax._backend.multimodal.qwen3_omni.tokens import QwenTokens


def get_llm_pos_ids_for_vision(
    start_idx: int | jax.Array,
    vision_idx: int,
    spatial_merge_size: int,
    t_index: jax.Array,
    grid_hs: jax.Array,
    grid_ws: jax.Array,
) -> jax.Array:
  """Computes 3D position IDs (temporal, height, width) for vision tokens.

  Creates position embeddings for a grid of vision tokens representing an
  image or video. For each temporal frame, generates a spatial grid of
  (height, width) positions.

  Args:
    start_idx: Starting position ID value to add as offset.
    vision_idx: Index of the current image/video being processed.
    spatial_merge_size: Number of patches merged spatially (e.g., 2 means 2x2 patches → 1 token).
    t_index: Temporal position for each frame. Shape: (num_frames,).
    grid_hs: Height dimensions for all images/videos. Shape: (num_images,).
    grid_ws: Width dimensions for all images/videos. Shape: (num_images,).

  Returns:
    3D position IDs with shape (3, num_vision_tokens) where:
      - dim 0: temporal positions
      - dim 1: height positions
      - dim 2: width positions

  Example:
    If spatial_merge_size=2, grid_h=4, grid_w=4, num_frames=2:
      - After merge: 2x2 grid per frame
      - Total tokens: 2 frames x 2 x 2 = 8 tokens
      - Output shape: (3, 8)
      - t_index: [0, 0, 0, 0, 50, 50, 50, 50]
      - h_index: [0, 0, 1, 1, 0, 0, 1, 1]
      - w_index: [0, 1, 0, 1, 0, 1, 0, 1]
  """
  # Calculate effective spatial dimensions after merging patches
  llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
  llm_grid_w = grid_ws[vision_idx] // spatial_merge_size

  # Create height indices: [0, 0, ..., 0 (W times), 1, 1, ..., 1 (W times), ...]
  # Shape: (num_frames, llm_grid_h, 1) -> expand -> (num_frames, llm_grid_h, llm_grid_w) -> flatten
  h_index = np.arange(llm_grid_h).reshape(1, -1, 1).repeat(len(t_index), axis=0).repeat(llm_grid_w, axis=2).flatten()

  # Create width indices: [0, 1, 2, ..., W-1, 0, 1, 2, ..., W-1, ...]
  # Shape: (num_frames, 1, llm_grid_w) -> expand -> (num_frames, llm_grid_h, llm_grid_w) -> flatten
  w_index = np.arange(llm_grid_w).reshape(1, 1, -1).repeat(len(t_index), axis=0).repeat(llm_grid_h, axis=1).flatten()

  # Create temporal indices: [t0, t0, ..., t0 (HxW times), t1, t1, ..., t1 (HxW times), ...]
  # Shape: (num_frames, 1) -> expand -> (num_frames, llm_grid_h * llm_grid_w) -> flatten
  t_index_expanded = t_index.reshape(-1, 1).repeat(llm_grid_h * llm_grid_w, axis=1).flatten()

  # Stack all three dimensions and add starting offset
  _llm_pos_ids = np.stack([t_index_expanded, h_index, w_index])
  llm_pos_ids = _llm_pos_ids + start_idx

  return llm_pos_ids


def get_chunked_index(token_indices: jax.Array, tokens_per_chunk: int, remove_index: int) -> list[tuple[int, int]]:
  """Splits token index list into chunks based on token value ranges.

  Given a list of monotonically increasing token indices, returns a list of
  (start, end) index tuples representing slices where token values fall within
  successive ranges of `tokens_per_chunk`.

  Args:
    token_indices: Monotonically increasing array of token index values. Shape: (seq_len,).
    tokens_per_chunk: Chunk size threshold (e.g., 100 means first chunk has values < 100).
    remove_index: Offset to subtract from token_indices before chunking.

  Returns:
    List of (start_idx, end_idx) tuples, each representing a chunk.

  Example:
    token_indices = [5, 10, 52, 105, 150, 250]
    tokens_per_chunk = 100
    remove_index = 0

    Result: [(0, 3), (3, 5), (5, 6)]
      - Chunk 0: indices 0-3 (values 5, 10, 52 are < 100)
      - Chunk 1: indices 3-5 (values 105, 150 are >= 100 and < 200)
      - Chunk 2: indices 5-6 (value 250 is >= 200)
  """
  chunks = []
  i = 0
  start_idx = 0
  current_chunk = 1

  while i < len(token_indices):
    if token_indices[i] - remove_index >= current_chunk * tokens_per_chunk:
      chunks.append((start_idx, i))
      start_idx = i
      current_chunk += 1
    i += 1

  # Append final chunk
  chunks.append((start_idx, len(token_indices)))

  return chunks


def get_rope_index(
    input_ids: np.ndarray,
    image_grid_thw: np.ndarray | None = None,
    video_grid_thw: np.ndarray | None = None,
    attention_mask: np.ndarray | None = None,
    use_audio_in_video: bool = False,
    audio_lengths: np.ndarray | None = None,
    second_per_grids: np.ndarray | None = None,
    spatial_merge_size: int = 2,
    position_id_per_seconds: int = 25,
    config=None,  # None defaults to Qwen3-Omni special tokens
) -> tuple[np.ndarray, np.ndarray]:
  """Calculate 3D RoPE position indices for multimodal sequences.

  This function computes position IDs that encode both sequential (text) and
  spatial-temporal (vision/audio) structure for Qwen3-Omni multimodal inputs.

  For pure text sequences:
    - All 3 dimensions receive the same sequential positions: [0, 1, 2, 3, 4]

  For multimodal sequences with vision:
    - Vision tokens get 3D positions (temporal, height, width)
    - Text tokens continue sequentially from max(vision_pos) + 1
    - Example with video (3 temporal patches, 2x2 spatial):
        Vision temporal: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
        Vision height:   [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        Vision width:    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        Text positions:  [101, 102, 103, 104, 105]

  Args:
    input_ids: Input token IDs. Shape: (batch, seq_len).
    image_grid_thw: Image dimensions (temporal, height, width). Shape: (num_images, 3).
    video_grid_thw: Video dimensions (temporal, height, width). Shape: (num_videos, 3).
    attention_mask: Padding mask (1 = real token, 0 = padding). Shape: (batch, seq_len).
    use_audio_in_video: If True, audio tokens are interleaved with video tokens.
    audio_lengths: Audio sequence lengths. Shape: (num_audios,).
    second_per_grids: Time interval per temporal grid (for videos). Shape: (num_videos,).
    spatial_merge_size: Number of patches merged spatially (e.g., 2 for 2x2→1).
    position_id_per_seconds: Temporal granularity (tokens per second, typically 25).

  Returns:
    A tuple of:
      - position_ids: 3D position IDs. Shape: (batch, seq_len, 3)
      - mrope_position_deltas: Position offset for each sequence. Shape: (batch, 1).

  Raises:
    ValueError: If multimodal tokens are present but grid info is missing.
  """
  qwen_tokens = QwenTokens(config)
  batch_size, seq_len = input_ids.shape

  # Handle text-only case (no multimodal content)
  if image_grid_thw is None and video_grid_thw is None:
    if attention_mask is None:
      attention_mask = np.ones_like(input_ids)

    # Create sequential 1D positions
    position_ids = np.cumsum(attention_mask.astype(np.float32), axis=-1) - 1
    position_ids = np.where(attention_mask == 0, 1.0, position_ids)

    # Expand to 3D (same value in all dimensions for text-only)
    position_ids = np.stack([position_ids, position_ids, position_ids], axis=-1)

    # Calculate deltas for each sequence
    max_position_ids = np.max(position_ids, axis=(1, 2), keepdims=True)  # (batch, 1, 1)
    mrope_position_deltas = max_position_ids.squeeze(-1) + 1 - np.sum(attention_mask, axis=-1, keepdims=True)

    return position_ids, mrope_position_deltas

  # Multimodal case: process each sequence in batch
  if attention_mask is None:
    attention_mask = np.ones_like(input_ids)

  attention_mask_bool = attention_mask == 1
  # Internally still build (3, batch, seq) then transpose to (batch, seq, 3).
  position_ids = np.zeros((3, batch_size, seq_len), dtype=np.float32)
  mrope_position_deltas = []

  # Process each sequence in the batch
  for i in range(batch_size):
    # Get valid tokens (non-padding)
    valid_input_ids = input_ids[i][attention_mask_bool[i]]

    # Count multimodal elements in this sequence
    vision_start_indices = np.where(valid_input_ids == qwen_tokens.vision_start)[0]
    vision_tokens = valid_input_ids[vision_start_indices + 1] if len(vision_start_indices) > 0 else np.array([])

    audio_nums = np.sum(valid_input_ids == qwen_tokens.audio_start).item()
    image_nums = np.sum(vision_tokens == qwen_tokens.image_pad).item() if len(vision_tokens) > 0 else 0
    video_nums = (
        (
            np.sum(vision_tokens == qwen_tokens.audio_start).item()
            if use_audio_in_video
            else np.sum(vision_tokens == qwen_tokens.video_pad).item()
        )
        if len(vision_tokens) > 0
        else 0
    )

    if image_grid_thw is not None:
      if image_grid_thw.ndim == 3 and image_grid_thw.shape[0] == batch_size:
        curr_image_grid_thw = image_grid_thw[i]
      elif image_grid_thw.ndim == 2 and image_grid_thw.shape[0] == batch_size and image_nums <= 1:
        curr_image_grid_thw = image_grid_thw[i : i + 1]
      else:
        curr_image_grid_thw = image_grid_thw
    else:
      curr_image_grid_thw = None

    if video_grid_thw is not None:
      if video_grid_thw.ndim == 3 and video_grid_thw.shape[0] == batch_size:
        curr_video_grid_thw = video_grid_thw[i]
      elif video_grid_thw.ndim == 2 and video_grid_thw.shape[0] == batch_size and (batch_size > 1 or video_nums <= 1):
        curr_video_grid_thw = video_grid_thw[i : i + 1]
      else:
        curr_video_grid_thw = video_grid_thw
    else:
      curr_video_grid_thw = None

    if second_per_grids is not None:
      if second_per_grids.ndim == 2 and second_per_grids.shape[0] == batch_size:
        curr_second_per_grids = second_per_grids[i]
      elif second_per_grids.ndim == 1 and len(second_per_grids) == batch_size and (batch_size > 1 or video_nums <= 1):
        curr_second_per_grids = second_per_grids[i : i + 1]
      else:
        curr_second_per_grids = second_per_grids
    else:
      curr_second_per_grids = None

    input_tokens = valid_input_ids.tolist()
    llm_pos_ids_list = []
    st = 0
    remain_images = image_nums
    remain_videos = video_nums
    remain_audios = audio_nums
    image_idx = 0
    video_idx = 0
    audio_idx = 0

    multimodal_nums = image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums

    # Process each multimodal element
    for _ in range(multimodal_nums):
      st_idx = llm_pos_ids_list[-1].max().item() + 1 if len(llm_pos_ids_list) > 0 else 0

      # Find next vision or audio start token
      if (qwen_tokens.image_pad in input_tokens or qwen_tokens.video_pad in input_tokens) and (
          remain_videos > 0 or remain_images > 0
      ):
        try:
          ed_vision_start = input_tokens.index(qwen_tokens.vision_start, st)
        except ValueError:
          ed_vision_start = len(input_tokens) + 1
      else:
        ed_vision_start = len(input_tokens) + 1

      if qwen_tokens.audio_pad in input_tokens and remain_audios > 0:
        try:
          ed_audio_start = input_tokens.index(qwen_tokens.audio_start, st)
        except ValueError:
          ed_audio_start = len(input_tokens) + 1
      else:
        ed_audio_start = len(input_tokens) + 1

      min_ed = min(ed_vision_start, ed_audio_start)

      # Add text tokens before multimodal element
      text_len = min_ed - st
      if text_len > 0:
        text_pos = np.arange(text_len).reshape(1, -1).repeat(3, axis=0) + st_idx
        llm_pos_ids_list.append(text_pos)
        st_idx += text_len

      # Determine BOS/EOS token lengths
      if min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
        bos_len, eos_len = 2, 2  # Audio in video
      else:
        bos_len, eos_len = 1, 1

      # Add BOS token(s)
      bos_pos = np.arange(bos_len).reshape(1, -1).repeat(3, axis=0) + st_idx
      llm_pos_ids_list.append(bos_pos)
      st_idx += bos_len

      # Process modality-specific content
      # Audio Only
      if min_ed == ed_audio_start:
        audio_len = _get_feat_extract_output_lengths(
            audio_lengths[audio_idx]  # pyrefly: ignore[unsupported-operation]
        ).item()  # pyrefly: ignore[unsupported-operation]
        audio_pos = np.arange(audio_len).reshape(1, -1).repeat(3, axis=0) + st_idx
        llm_pos_ids_list.append(audio_pos)

        st += int(text_len + bos_len + audio_len + eos_len)
        audio_idx += 1
        remain_audios -= 1

      # Image Only
      elif min_ed == ed_vision_start and input_tokens[ed_vision_start + 1] == qwen_tokens.image_pad:
        grid_t = curr_image_grid_thw[image_idx, 0].item()  # pyrefly: ignore[unsupported-operation]
        grid_hs = curr_image_grid_thw[:, 1]  # pyrefly: ignore[unsupported-operation]
        grid_ws = curr_image_grid_thw[:, 2]  # pyrefly: ignore[unsupported-operation]
        t_index = np.arange(grid_t, dtype=np.float32) * 1 * position_id_per_seconds

        image_pos = get_llm_pos_ids_for_vision(
            st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws  # pyrefly: ignore[bad-argument-type]
        )  # pyrefly: ignore[bad-argument-type]
        llm_pos_ids_list.append(image_pos)

        image_len = int(
            np.prod(curr_image_grid_thw[image_idx]).item()
            // (spatial_merge_size**2)  # pyrefly: ignore[unsupported-operation]
        )  # pyrefly: ignore[unsupported-operation]
        st += int(text_len + bos_len + image_len + eos_len)
        image_idx += 1
        remain_images -= 1

      # Video Only
      elif min_ed == ed_vision_start and input_tokens[ed_vision_start + 1] == qwen_tokens.video_pad:
        grid_t = curr_video_grid_thw[video_idx, 0].item()  # pyrefly: ignore[unsupported-operation]
        grid_hs = curr_video_grid_thw[:, 1]  # pyrefly: ignore[unsupported-operation]
        grid_ws = curr_video_grid_thw[:, 2]  # pyrefly: ignore[unsupported-operation]
        second_per_grid = (
            curr_second_per_grids[video_idx].item()
            if curr_second_per_grids is not None
            else (float(config.temporal_patch_size_for_vit) if config is not None else 1.0)
        )
        t_index = np.arange(grid_t, dtype=np.float32) * second_per_grid * position_id_per_seconds

        video_pos = get_llm_pos_ids_for_vision(
            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws  # pyrefly: ignore[bad-argument-type]
        )  # pyrefly: ignore[bad-argument-type]
        llm_pos_ids_list.append(video_pos)

        video_len = int(
            np.prod(curr_video_grid_thw[video_idx]).item()
            // (spatial_merge_size**2)  # pyrefly: ignore[unsupported-operation]
        )  # pyrefly: ignore[unsupported-operation]
        st += int(text_len + bos_len + video_len + eos_len)
        video_idx += 1
        remain_videos -= 1

      # Audio in Video (interleaved)
      elif min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
        audio_len = _get_feat_extract_output_lengths(
            audio_lengths[audio_idx]  # pyrefly: ignore[unsupported-operation]
        ).item()  # pyrefly: ignore[unsupported-operation]
        audio_llm_pos_ids = np.arange(audio_len).reshape(1, -1).repeat(3, axis=0) + st_idx

        grid_t = curr_video_grid_thw[video_idx, 0].item()  # pyrefly: ignore[unsupported-operation]
        grid_hs = curr_video_grid_thw[:, 1]  # pyrefly: ignore[unsupported-operation]
        grid_ws = curr_video_grid_thw[:, 2]  # pyrefly: ignore[unsupported-operation]
        second_per_grid = (
            curr_second_per_grids[video_idx].item()
            if curr_second_per_grids is not None
            else (float(config.temporal_patch_size_for_vit) if config is not None else 1.0)
        )
        t_index = np.arange(grid_t, dtype=np.float32) * second_per_grid * position_id_per_seconds

        video_llm_pos_ids = get_llm_pos_ids_for_vision(
            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws  # pyrefly: ignore[bad-argument-type]
        )  # pyrefly: ignore[bad-argument-type]

        # Interleave audio and video based on temporal ordering
        video_data_index = 0
        audio_data_index = 0
        while video_data_index < video_llm_pos_ids.shape[1] and audio_data_index < audio_llm_pos_ids.shape[1]:
          if video_llm_pos_ids[0, video_data_index] <= audio_llm_pos_ids[0, audio_data_index]:
            llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index : video_data_index + 1])
            video_data_index += 1
          else:
            llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index : audio_data_index + 1])
            audio_data_index += 1

        # Append remaining tokens
        if video_data_index < video_llm_pos_ids.shape[1]:
          llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index:])
        if audio_data_index < audio_llm_pos_ids.shape[1]:
          llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index:])

        video_len = int(
            np.prod(video_grid_thw[video_idx]).item() // (spatial_merge_size**2)  # pyrefly: ignore[unsupported-operation]
        )  # pyrefly: ignore[unsupported-operation]
        st += int(text_len + bos_len + audio_len + video_len + eos_len)

        audio_idx += 1
        video_idx += 1
        remain_videos -= 1
        remain_audios -= 1

      # Add EOS token(s)
      st_idx = llm_pos_ids_list[-1].max().item() + 1 if len(llm_pos_ids_list) > 0 else 0
      eos_pos = np.arange(eos_len).reshape(1, -1).repeat(3, axis=0) + st_idx
      llm_pos_ids_list.append(eos_pos)

    # Add any remaining text tokens
    if st < len(input_tokens):
      st_idx = llm_pos_ids_list[-1].max().item() + 1 if len(llm_pos_ids_list) > 0 else 0
      text_len = len(input_tokens) - st
      remaining_text_pos = np.arange(text_len).reshape(1, -1).repeat(3, axis=0) + st_idx
      llm_pos_ids_list.append(remaining_text_pos)

    # Concatenate all position IDs for this sequence
    llm_positions = np.concatenate(llm_pos_ids_list, axis=1)

    # Place into position_ids tensor at valid positions
    valid_positions = np.where(attention_mask_bool[i])[0]
    position_ids[:, i, valid_positions] = llm_positions

    # Calculate delta for this sequence
    mrope_position_deltas.append(llm_positions.max().item() + 1 - len(valid_input_ids))

  mrope_position_deltas = np.array(mrope_position_deltas).reshape(batch_size, 1)
  position_ids = np.transpose(position_ids, (1, 2, 0))  # (3, batch, seq) -> (batch, seq, 3)

  return position_ids, mrope_position_deltas
