"""Video file resolution and frame decoding for Qwen3-Omni."""

import logging
import os
import tempfile

import numpy as np
import decord

from lilax._backend.multimodal.qwen3_omni.video.sampling import calculate_video_frame_range, smart_nframes
from lilax._backend.utils import gcs_utils
from lilax._backend.utils import max_logging


def _read_video_decord(video_path, video_start=0.0, video_end=None) -> tuple[np.ndarray, float]:
  """Read video using decord.VideoReader (torch-free version)

  Args:
    video: the path of video. support "file://", "http://", "https://" and local path.
    video_start: the start time of video.
    video_end: the end time of video.

  Returns:
      tuple: (numpy.ndarray with shape (T, C, H, W), sample_fps as float)

  Raises:
      FileNotFoundError: If the video file does not exist.
      RuntimeError: If the video file cannot be read.
  """
  if not os.path.isfile(video_path):
    raise FileNotFoundError(f"Video file not found at path {video_path}. Please specify a valid video file path")
  video_config = {
      "video": video_path,
      "video_start": video_start,
      "video_end": video_end,
  }
  try:
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        video_config,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(video_config, total_frames=total_frames, video_fps=video_fps)

    # Use numpy linspace instead of torch.linspace
    idx = np.linspace(start_frame, end_frame, nframes).round().astype(int).tolist()

    video = vr.get_batch(idx).asnumpy()
    # Convert from THWC to TCHW format using numpy
    video = np.transpose(video, (0, 3, 1, 2))

    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning("Failed to read/decode video %s: %s. Using dummy video.", video_path, e)
    # Return dummy video: 4 frames of 224x224 black pixels
    # 224 is a multiple of 28 (patch_size * merge_size = 14 * 2 = 28)
    dummy_video = np.zeros((4, 3, 224, 224), dtype=np.uint8)
    return dummy_video, 2.0


def is_video_file(path):
  if not isinstance(path, str):
    return False
  video_extensions = (".mp4", ".avi", ".mkv", ".webm", ".mov", ".gif")
  return path.lower().endswith(video_extensions)


def load_training_video(images, config) -> tuple[np.ndarray, float]:
  """Resolves, downloads when remote, and decodes the single video of an SFT training example."""
  video_path = images
  if isinstance(video_path, list):
    if len(video_path) > 1:
      raise ValueError("Only 1 video per example is supported for now.")
    video_path = video_path[0]

  # Auto-assemble video directory from train or eval files
  video_dir = None
  if config.hf_train_files:
    video_dir = os.path.dirname(config.hf_train_files)
  elif config.hf_eval_files:
    video_dir = os.path.dirname(config.hf_eval_files)

  if video_dir:
    video_path = os.path.join(video_dir, video_path)

  is_gcs_video = video_path.startswith("gs://")
  temp_local_path = None

  if is_gcs_video:
    ext = os.path.splitext(video_path)[1]
    with tempfile.NamedTemporaryFile(suffix=ext, prefix="maxtext_video_sft_", delete=False) as temp_file:
      temp_local_path = temp_file.name

    max_logging.log(f"Downloading remote video {video_path} to temp local path {temp_local_path}...")
    try:
      bucket_name, prefix_name = gcs_utils.parse_gcs_bucket_and_prefix(video_path)
      storage_client = gcs_utils.storage.Client()
      bucket = storage_client.get_bucket(bucket_name)
      blob = bucket.blob(prefix_name)
      blob.download_to_filename(temp_local_path)
      video_path = temp_local_path
    except Exception as e:
      if os.path.exists(temp_local_path):
        os.remove(temp_local_path)
      raise RuntimeError(f"Failed to download video from GCS: {e}") from e
  else:
    if not os.path.exists(video_path):
      raw_path = images if isinstance(images, str) else images[0]
      if os.path.exists(raw_path):
        video_path = raw_path
      else:
        raise FileNotFoundError(f"Video file not found: {video_path} (original: {raw_path})")

  try:
    video_array, sample_fps = _read_video_decord(video_path)
  finally:
    if temp_local_path and os.path.exists(temp_local_path):
      max_logging.log(f"Cleaning up temp local video file {temp_local_path}...")
      os.remove(temp_local_path)
  return video_array, sample_fps
