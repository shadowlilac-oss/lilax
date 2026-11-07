"""Hugging Face processing factories for native token and media inputs."""

from transformers import AutoProcessor as HFAutoProcessor, AutoTokenizer as HFAutoTokenizer

from lilax._processing.processor import NativeProcessor


def _set_processing_options(tokenizer, values, *, media_defaults=None):
  defaults = {"tokenize": True, "use_chat_template": False, "use_truncation": True, **(media_defaults or {})}
  tokenizer.init_kwargs["lilax_processing"] = defaults | tokenizer.init_kwargs.get("lilax_processing", {}) | {
      name: value for name, value in values.items() if value is not None
  }


class AutoTokenizer:
  @staticmethod
  def from_pretrained(
      pretrained_model_name_or_path, *, revision=None, token=None, cache_dir=None, local_files_only=False,
      tokenize: bool | None = None, use_chat_template: bool | None = None, use_truncation: bool | None = None, **kwargs,
  ):
    tokenizer = HFAutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, revision=revision, token=token, cache_dir=cache_dir,
        local_files_only=local_files_only, **kwargs,
    )
    _set_processing_options(tokenizer, {
        "tokenize": tokenize, "use_chat_template": use_chat_template, "use_truncation": use_truncation,
    })
    return tokenizer


class AutoProcessor:
  @staticmethod
  def from_pretrained(
      pretrained_model_name_or_path, *, revision=None, token=None, cache_dir=None, local_files_only=False,
      tokenize: bool | None = None, use_chat_template: bool | None = None, use_truncation: bool | None = None,
      image_column: str | list[str] | None = None, image_placeholder: str | None = None,
      video_placeholder: str | None = None, max_num_images_per_example: int | None = None,
      audio_placeholder: str | None = None, use_audio_in_video: bool | None = None,
      video_max_grid_t: int | None = None, video_max_grid_h: int | None = None,
      video_max_grid_w: int | None = None, **kwargs,
  ):
    processor = HFAutoProcessor.from_pretrained(
        pretrained_model_name_or_path, revision=revision, token=token, cache_dir=cache_dir,
        local_files_only=local_files_only, **kwargs,
    )
    _set_processing_options(processor.tokenizer, {
        "tokenize": tokenize, "use_chat_template": use_chat_template, "use_truncation": use_truncation,
        "image_column": image_column, "image_placeholder": image_placeholder, "video_placeholder": video_placeholder,
        "max_num_images_per_example": max_num_images_per_example,
        "audio_placeholder": audio_placeholder, "use_audio_in_video": use_audio_in_video,
        "video_max_grid_t": video_max_grid_t, "video_max_grid_h": video_max_grid_h,
        "video_max_grid_w": video_max_grid_w,
    }, media_defaults={"image_column": "image", "image_placeholder": "<|image|>", "video_placeholder": "<|video|>",
                       "max_num_images_per_example": -1, "audio_placeholder": "<|audio|>",
                       "use_audio_in_video": False, "video_max_grid_t": None, "video_max_grid_h": None,
                       "video_max_grid_w": None})
    return NativeProcessor(
        processor, pretrained_model_name_or_path, revision=revision, token=token, cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


