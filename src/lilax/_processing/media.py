"""Compose existing native media preprocessing into canonical model input arrays."""

from os import PathLike

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_transforms import to_channel_dimension_format
from transformers.image_utils import ChannelDimension, make_nested_list_of_images, to_numpy_array

from lilax._backend.input_pipeline.transforms.pad_or_trim import PadOrTrimToMaxLength
from lilax._backend.multimodal import processor as mm_processor
from lilax._backend.multimodal import utils as mm_utils


def fold_media(config, values):
  values = dict(values)
  for key, mask in (("images", "image_masks"), ("videos", "video_masks")):
    if values.get(key) is None:
      continue
    media = values[key]
    if media.ndim > len(mm_processor.get_dummy_image_shape_for_init(config)):
      values[key] = media.reshape(-1, *media.shape[2:])
      if values.get(mask) is not None:
        masks = values[mask]
        values[mask] = masks.reshape(-1, *masks.shape[2:])
      if key == "videos" and values.get("video_grid_thw") is not None:
        grid = values["video_grid_thw"]
        values["video_grid_thw"] = grid.reshape(-1, grid.shape[-1])
  if values.get("audio_values") is not None:
    audio = values["audio_values"]
    if audio.ndim > len(mm_processor.get_dummy_audio_shape_for_init(config)):
      values["audio_values"] = audio.reshape(-1, *audio.shape[2:])
  return values


def _media_paths(value, batch_size, name):
  if value is None:
    return [None] * batch_size
  values = [value] if isinstance(value, (str, PathLike)) else list(value)
  if len(values) != batch_size or any(not isinstance(item, (str, PathLike)) for item in values):
    raise ValueError(f"Native {name} preprocessing requires one media file path per text input.")
  return list(map(str, values))


def _load_images(value):
  if isinstance(value, (list, tuple)):
    return [_load_images(item) for item in value]
  if isinstance(value, PathLike):
    value = str(value)
  return mm_utils.convert_to_RGB(value)


def prepare_media_inputs(tokenizer, config, text, images, videos, audio, **kwargs):
  texts = [text] if isinstance(text, str) else list(text or [])
  if not texts or any(not isinstance(item, str) for item in texts):
    raise ValueError("Multimodal processing requires a text string or a batch of text strings.")
  image_groups = [[] for _ in texts] if images is None else make_nested_list_of_images(_load_images(images))
  if len(image_groups) != len(texts):
    raise ValueError("Supply one image group per text input; use nested image lists for a batch.")
  video_paths = _media_paths(videos, len(texts), "video")
  audio_paths = _media_paths(audio, len(texts), "audio")
  if videos is not None and config.vision_encoder_block not in ("qwen3_omni", "qwen3_vl", "qwen3_5"):
    raise ValueError("The selected native processor has no video preprocessing route.")
  if audio is not None and config.decoder_block != "ming":
    raise ValueError("Separate native audio files are supported by Ming; Qwen audio is extracted from video.")
  text_kwargs = kwargs.pop("text_kwargs", {})
  for name in ("images_kwargs", "videos_kwargs", "audio_kwargs"):
    if kwargs.pop(name, {}):
      raise ValueError(f"Native media preprocessing does not consume {name}; use the processor's native preferences.")
  kwargs = text_kwargs | kwargs
  if kwargs.get("return_offsets_mapping"):
    raise ValueError("Tokenizer offset mappings do not describe the native expanded media tokens.")
  return_attention_mask = kwargs.pop("return_attention_mask", True)
  return_tensors = kwargs.pop("return_tensors", None)
  padding = kwargs.pop("padding", False)
  max_length = kwargs.pop("max_length", None)
  input_length = config.max_target_length if max_length is None else max_length
  truncation = kwargs.pop("truncation", config.use_truncation)
  rows, processed_rows = [], []
  for prompt, image_group, video_path, audio_path in zip(texts, image_groups, video_paths, audio_paths):
    row_config = config.model_copy(update={"image_path": "", "video_path": video_path or "", "audio_path": audio_path or ""})
    processed = mm_processor.preprocess_mm_data(row_config) if video_path or audio_path else mm_utils.PreprocessorOutput()
    if image_group:
      limit = config.max_num_images_per_example
      if limit > 0:
        image_group = image_group[:limit]
      image_group = [to_channel_dimension_format(to_numpy_array(mm_utils.convert_to_RGB(image)), ChannelDimension.LAST)
                     for image in image_group]
      image_output = mm_processor.preprocess_image_for_training(image_group, row_config)
      if video_path or audio_path:
        for name in ("pixel_values", "pixel_mask", "pixel_grid_thw", "aspect_ratios", "num_images"):
          if hasattr(image_output, name):
            setattr(processed, name, getattr(image_output, name))
      else:
        processed = image_output
    offset = mm_processor.get_image_offsets(row_config, processed)
    text_limit = input_length - offset
    if text_limit <= 0:
      raise ValueError("The native media tokens exceed the requested input length.")
    encoded = tokenizer(prompt, truncation=truncation, max_length=text_limit,
                        return_token_type_ids=False, **kwargs)
    tokens = mm_processor.prepare_text_for_image_fusion(encoded["input_ids"], row_config, processed)
    if len(tokens) > config.max_target_length:
      raise ValueError("The expanded native media input exceeds the model context length.")
    rows.append({"input_ids": np.asarray(tokens).tolist()})
    if config.max_num_images_per_example > 0 and processed.pixel_values is not None:
      processed = PadOrTrimToMaxLength(
          input_length, tokenizer.pad_token_id, config=config,
          max_num_images_per_example=config.max_num_images_per_example,
      )._pad_image_and_mask(processed)
    processed_rows.append(processed)
  encoded = tokenizer.pad(rows, padding=padding, max_length=input_length if padding == "max_length" else max_length,
                          return_attention_mask=True)
  values = dict(encoded)
  for source, target in (("pixel_values", "images"), ("pixel_mask", "image_masks"),
                         ("video_values", "videos"), ("video_mask", "video_masks"),
                         ("video_grid_thw", "video_grid_thw"), ("audio_values", "audio_values")):
    arrays = [getattr(row, source, None) for row in processed_rows]
    if any(array is not None for array in arrays):
      if any(array is None for array in arrays) or len({np.shape(array) for array in arrays}) != 1:
        raise ValueError(f"Native {target} batches require matching media shapes on every input.")
      values[target] = np.stack(arrays)
  if config.use_mrope:
    # These imports retain the native decode route's platform-dependent import boundary.
    from lilax._backend.multimodal.qwen3_omni.video import loading  # pylint: disable=unused-import
    from lilax._backend.multimodal.qwen3_omni import positions

    position_rows = []
    for index, processed in enumerate(processed_rows):
      position_ids, _ = positions.get_rope_index(
          input_ids=np.asarray(values["input_ids"])[index:index + 1],
          image_grid_thw=getattr(processed, "pixel_grid_thw", None),
          video_grid_thw=getattr(processed, "video_grid_thw", None),
          attention_mask=np.asarray(values["attention_mask"])[index:index + 1],
          use_audio_in_video=config.use_audio and getattr(processed, "num_videos", 0) > 0,
          audio_lengths=getattr(processed, "audio_lengths", None),
          second_per_grids=getattr(processed, "video_second_per_grid", None),
          spatial_merge_size=config.spatial_merge_size_for_vit,
          position_id_per_seconds=config.position_id_per_seconds, config=config,
      )
      position_rows.append(position_ids[0])
    values["position_ids"] = np.stack(position_rows)
  if return_attention_mask is False:
    values.pop("attention_mask")
  return BatchFeature(values, tensor_type=return_tensors)
