"""ProcessorMixin integration for native token and media input representations."""

from pathlib import Path

from transformers import ProcessorMixin

from lilax.configuration import AutoConfig, MaxTextConfig
from lilax.configuration.model_mapping import project_model_config
from .media import prepare_media_inputs
from .preferences import processing_overrides


class NativeProcessor(ProcessorMixin):
  def __init__(self, processor, source, *, revision=None, token=None, cache_dir=None, local_files_only=False):
    self._processor = processor
    self._source = str(source)
    self.tokenizer = processor.tokenizer
    self.chat_template = getattr(processor, "chat_template", None)
    self.valid_processor_kwargs = processor.valid_processor_kwargs
    self._load_options = {
        "revision": revision, "token": token, "cache_dir": cache_dir, "local_files_only": local_files_only,
    }
    options = self.tokenizer.init_kwargs["lilax_processing"]
    options.setdefault("model_source", str(source))
    if revision is not None:
      if options.get("model_revision") != revision:
        options.pop("native_model_config", None)
      options["model_revision"] = revision

  def __getattr__(self, name):
    return getattr(self._processor, name)

  def get_attributes(self):
    return self._processor.get_attributes()

  @property
  def model_input_names(self):
    return ["input_ids", "attention_mask", "position_ids", "images", "image_masks", "videos",
            "video_masks", "video_grid_thw", "audio_values"]

  def __call__(self, images=None, text=None, videos=None, audio=None, **kwargs):
    if kwargs.pop("return_token_type_ids", False):
      raise ValueError("Native model inputs use attention_mask and segment_ids, not token_type_ids.")
    if images is None and videos is None and audio is None:
      return self.tokenizer(text, return_token_type_ids=False, **kwargs)
    config = project_model_config(self._model_config()).model_copy(update=processing_overrides(self))
    return prepare_media_inputs(self.tokenizer, config, text, images, videos, audio, **kwargs)

  def _model_config(self):
    options = self.tokenizer.init_kwargs["lilax_processing"]
    if (Path(self._source).expanduser() / "config.json").is_file():
      model_config = AutoConfig.from_pretrained(self._source, **self._load_options)
      options["native_model_config"] = model_config.to_dict()
    elif "native_model_config" in options:
      model_config = MaxTextConfig(**options["native_model_config"])
    else:
      load_options = self._load_options | {"revision": options.get("model_revision")}
      model_config = AutoConfig.from_pretrained(options["model_source"], **load_options)
      options["native_model_config"] = model_config.to_dict()
    return model_config

  def save_pretrained(self, save_directory, **kwargs):
    self._model_config()
    return self._processor.save_pretrained(save_directory, **kwargs)
