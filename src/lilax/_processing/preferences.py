"""Project persisted processing preferences into existing pipeline consumers."""


def processing_overrides(processing_class):
  if processing_class is None:
    return {}
  tokenizer = getattr(processing_class, "tokenizer", processing_class)
  options = tokenizer.init_kwargs.get("lilax_processing", {})
  values = {"tokenizer_type": "huggingface"}
  if "tokenize" in options:
    values.update(tokenize_train_data=options["tokenize"], tokenize_eval_data=options["tokenize"])
  if "use_chat_template" in options:
    values["use_chat_template"] = options["use_chat_template"]
  if "use_truncation" in options:
    values["use_truncation"] = options["use_truncation"]
  for source, targets in (
      ("image_column", ("train_image_column", "eval_image_column")),
      ("image_placeholder", ("image_placeholder",)), ("video_placeholder", ("video_placeholder",)),
      ("max_num_images_per_example", ("max_num_images_per_example",)),
      ("audio_placeholder", ("audio_placeholder",)), ("use_audio_in_video", ("use_audio_in_video",)),
      ("video_max_grid_t", ("video_max_grid_t",)), ("video_max_grid_h", ("video_max_grid_h",)),
      ("video_max_grid_w", ("video_max_grid_w",)),
  ):
    if source in options:
      values.update({target: options[source] for target in targets})
  return values
