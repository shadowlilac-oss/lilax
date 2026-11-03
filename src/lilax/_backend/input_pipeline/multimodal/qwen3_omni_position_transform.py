"""Grain transform computing Qwen3-Omni 3D multimodal position IDs."""

import dataclasses

import grain.python as grain
import numpy as np


@dataclasses.dataclass
class ComputeQwen3OmniPositions(grain.MapTransform):
  """Computes 3D position IDs for Qwen3-Omni multimodal sequences.

  This transform replaces the standard 1D sequential positions with 3D
  positions (temporal, height, width) for multimodal models like Qwen3-Omni.

  For text-only sequences, all 3 dimensions receive the same sequential values.
  For multimodal sequences with vision/audio, vision tokens get true 3D positions
  and text tokens continue sequentially from max(vision_pos) + 1.

  The actual position computation is delegated to multimodal_utils.get_rope_index(),
  which can be tested and modified independently.
  """

  def __init__(
      self,
      data_column: str = "inputs",
      spatial_merge_size: int = 2,
      position_id_per_seconds: int = 25,
      use_audio_in_video: bool = False,
      config=None,
      keep_aux_fields: bool = False,
  ):
    """Initialize the Qwen3-Omni position computation transform.

    Args:
      data_column: Name of the data column to compute positions for (default: "inputs").
      spatial_merge_size: Number of patches merged spatially (e.g., 2 for 2x2→1).
      position_id_per_seconds: Temporal granularity (tokens per second, typically 25).
      use_audio_in_video: If True, audio tokens are interleaved with video tokens.
      config: Optional model config for model-specific special token IDs.
      keep_aux_fields: If True, keep auxiliary (aux) fields — mrope deltas / grid
        metadata — on the element for inference. Training should leave this False
        so the batch matches get_shaped_batch.
    """
    self.data_column = data_column
    self.spatial_merge_size = spatial_merge_size
    self.position_id_per_seconds = position_id_per_seconds
    self.use_audio_in_video = use_audio_in_video
    self.config = config
    self.keep_aux_fields = keep_aux_fields

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute 3D position IDs for the batch element.

    Args:
      element: Dictionary containing:
        - {data_column}: Token IDs with shape (batch, seq_len)
        - {data_column}_segmentation: Attention mask (1=real, 0=padding)
        - image_grid_thw: Optional (num_images, 3) or (batch, num_images, 3) array
        - video_grid_thw: Optional (num_videos, 3) array
        - audio_lengths: Optional (num_audios,) array
        - second_per_grids: Optional (num_videos,) array

    Returns:
      element with {data_column}_position updated to shape (batch, seq_len, 3)
      for 3D positions (always 3D, even for text-only sequences).
    """
    input_ids = element[self.data_column]
    attention_mask = element.get(f"{self.data_column}_segmentation")
    image_grid_thw = element.get("image_grid_thw")
    video_grid_thw = element.get("video_grid_thw")
    audio_lengths = element.get("audio_lengths")
    second_per_grids = element.get("second_per_grids")
    if image_grid_thw is not None and image_grid_thw.ndim == 3:
      image_grid_thw = image_grid_thw[0]

    # Loads the decord/GCS video dependencies here, as the former processor import did.
    from lilax._backend.multimodal.qwen3_omni.video import loading  # pylint: disable=import-outside-toplevel,unused-import
    from lilax._backend.multimodal.qwen3_omni import positions  # pylint: disable=import-outside-toplevel

    position_ids, mrope_position_deltas = positions.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        use_audio_in_video=self.use_audio_in_video,
        audio_lengths=audio_lengths,
        second_per_grids=second_per_grids,
        spatial_merge_size=self.spatial_merge_size,
        position_id_per_seconds=self.position_id_per_seconds,
        config=self.config,
    )
    element[f"{self.data_column}_position"] = position_ids.astype(np.int32)
    if self.keep_aux_fields:
      element[f"{self.data_column}_mrope_deltas"] = mrope_position_deltas
    else:
      element.pop(f"{self.data_column}_mrope_deltas", None)
      element.pop("image_grid_thw", None)
      if getattr(self.config, "video_max_grid_t", None) is None:
        element.pop("video_grid_thw", None)
      element.pop("audio_lengths", None)
      element.pop("second_per_grids", None)
    return element
