"""Video frame range, frame count and temporal grid timing for Qwen3-Omni."""

import math

from lilax._backend.observability import max_logging

FRAME_FACTOR = 2  # Frame count must be divisible by this factor.
FPS = 2.0  # Default frames per second for video sampling.
FPS_MIN_FRAMES = 4  # Minimum number of frames to extract from video.
FPS_MAX_FRAMES = 768  # Maximum number of frames to extract from video.


def calculate_video_frame_range(
    ele: dict,
    total_frames: int,
    video_fps: float,
) -> tuple[int, int, int]:
  """
  Calculate the start and end frame indices based on the given time range.

  Args:
      ele (dict): A dictionary containing optional 'video_start' and 'video_end' keys (in seconds).
      total_frames (int): Total number of frames in the video.
      video_fps (float): Frames per second of the video.

  Returns:
      tuple: A tuple containing (start_frame, end_frame, frame_count).

  Raises:
      ValueError: If input parameters are invalid or the time range is inconsistent.
  """
  if video_fps <= 0:
    raise ValueError("video_fps must be a positive number")
  if total_frames <= 0:
    raise ValueError("total_frames must be a positive integer")

  video_start = ele.get("video_start", None)
  video_end = ele.get("video_end", None)
  if video_start is None and video_end is None:
    return 0, total_frames - 1, total_frames

  max_duration = total_frames / video_fps
  # Process start frame
  if video_start is not None:
    video_start_clamped = max(0.0, min(video_start, max_duration))
    start_frame = math.ceil(video_start_clamped * video_fps)
  else:
    start_frame = 0
  # Process end frame
  if video_end is not None:
    video_end_clamped = max(0.0, min(video_end, max_duration))
    end_frame = math.floor(video_end_clamped * video_fps)
    end_frame = min(end_frame, total_frames - 1)
  else:
    end_frame = total_frames - 1

  # Validate frame order
  if start_frame >= end_frame:
    raise ValueError(
        # pyrefly: ignore[unbound-name]
        f"Invalid time range: Start frame {start_frame} (at {video_start_clamped if video_start is not None else 0}s) "
        f"exceeds end frame {end_frame} (at {video_end_clamped if video_end is not None else max_duration}s). "
        f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)"
    )

  return start_frame, end_frame, end_frame - start_frame + 1


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
  """Calculate the number of frames for video used for model inputs.

  Args:
      ele (dict): a dict contains the configuration of video.
          support either `fps` or `nframes`:
              - nframes: the number of frames to extract for model inputs.
              - fps: the fps to extract frames for model inputs.
                  - min_frames: the minimum number of frames of the video, only used when fps is provided.
                  - max_frames: the maximum number of frames of the video, only used when fps is provided.
      total_frames (int): the original total number of frames of the video.
      video_fps (int | float): the original fps of the video.

  Returns:
      int: the number of frames for video used for model inputs.
  """

  def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

  def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

  def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

  assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
  if "nframes" in ele:
    nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
  else:
    fps = ele.get("fps", FPS)
    min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
    max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
    nframes = total_frames / video_fps * fps
    if nframes > total_frames:
      max_logging.log(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
    nframes = min(max(nframes, min_frames), max_frames, total_frames)
    nframes = floor_by_factor(nframes, FRAME_FACTOR)
  if not FRAME_FACTOR <= nframes <= total_frames:
    raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
  return nframes


def get_video_second_per_grid(sample_fps: float, temporal_patch_size: int) -> float:
  """Return elapsed seconds represented by one temporal video grid step."""
  if sample_fps <= 0:
    raise ValueError(f"sample_fps must be positive, got {sample_fps}.")
  return temporal_patch_size / sample_fps
