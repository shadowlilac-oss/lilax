"""Per-example prompt, response, and media preprocessing for multimodal SFT."""

import numpy as np

from lilax._backend.multimodal import processor as mm_processor
from lilax._backend.multimodal import utils as mm_utils


def is_video_file(path):
  if not isinstance(path, str):
    return False
  video_extensions = (".mp4", ".avi", ".mkv", ".webm", ".mov")
  return path.lower().endswith(video_extensions)


def reformat_prompt(
    example,
    column,
    image_placeholder,
    model_name,
    video_placeholder="<|video|>",
):
  """reformat prompt for multimodal SFT"""
  media = example.get("images")
  if media is None:
    media = []
  elif not isinstance(media, list):
    media = [media]

  contains_video = bool(media) and is_video_file(media[0])

  example[column] = mm_processor.reformat_prompt(
      prompt=example[column],
      image_placeholder=image_placeholder,
      model_name=model_name,
      num_images=0 if contains_video else len(media),
      video_placeholder=video_placeholder,
      num_videos=len(media) if contains_video else 0,
  )
  return example


def reformat_response(example, column, model_name):
  """reformat response for multimodal SFT"""
  val = example[column]
  if not val:
    raise ValueError(f"Response column '{column}' cannot be empty or None: {val}")
  response = val[0] if isinstance(val, (list, tuple)) else val

  example[column] = mm_processor.reformat_response(response, model_name)
  return example


def merge_image_columns(example, image_columns, max_num_images_per_example):
  """Merge multiple image columns into a single list of images."""
  images = []
  for col in image_columns:
    if isinstance(example[col], list):
      images.extend(example[col])
    else:
      images.append(example[col])

  example["images"] = images[:max_num_images_per_example] if max_num_images_per_example > 0 else images
  return example


def pre_process_image_sft(example, image_column, config):
  """pre-process image for multimodal SFT"""

  def _process_image_fn(image):
    is_video = is_video_file(image) or (isinstance(image, list) and image and is_video_file(image[0]))
    if is_video:
      return mm_processor.preprocess_image_for_training(image, config)

    if isinstance(image, list):
      image = [np.array(mm_utils.convert_to_RGB(img)) for img in image]
    else:
      image = np.array(mm_utils.convert_to_RGB(image))

    image = mm_processor.preprocess_image_for_training(image, config)
    return image

  example[image_column] = _process_image_fn(example[image_column])
  return example


def prepare_text_for_image_fusion(example, column_name, config):
  """prepare text for image fusion for multimodal SFT"""
  example[column_name] = mm_processor.prepare_text_for_image_fusion(
      tokens=example[column_name], config=config, processor_output=example["images"]
  )
  return example
