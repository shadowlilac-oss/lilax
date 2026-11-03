"""Data option validation for post-training."""


def _check_post_training_data_options(config, runtime, args):
  explicit = runtime.public_config._explicit_keys | runtime._overrides.keys()
  if args is not None:
    explicit = explicit | args._overrides().keys()
  unsupported = []
  if config.dataset_type not in ("grain", "tfds") and "colocated_python_data_input" in explicit:
    unsupported.append("colocated_python_data_input")
  padding_ignored = (
      config.dataset_type in ("synthetic", "c4_mlperf")
      or (config.dataset_type in ("grain", "tfds") and config.colocated_python_data_input)
      or (config.dataset_type == "hf" and config.use_sft and config.use_multimodal)
  )
  if padding_ignored:
    unsupported.extend(name for name in ("generate_padding_batch_train", "generate_padding_batch_eval") if name in explicit)
  if unsupported:
    raise ValueError(f"Post-training data pipeline does not use these settings: {sorted(unsupported)}")
