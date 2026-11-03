"""Validation of native multimodal model support."""


def validate_multimodal_support(config):
  if config.use_multimodal:
    # Gemma 4 small (E2B / E4B) only supports text for now; multimodal
    # support is pending clipped-linears in the vision encoder.
    if config.model_name in ("gemma4-e2b", "gemma4-e4b"):
      raise ValueError(f"Multimodal is not yet supported for {config.model_name}; only text inputs are supported.")
    valid_mm_models = (
        "gemma3-4b", "gemma3-12b", "gemma3-27b", "gemma4-26b", "gemma4-31b", "llama4-17b-16e",
        "llama4-17b-128e", "qwen3-omni-30b-a3b", "qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-30b-a3b",
        "qwen3.5-9b", "qwen3.5-35b-a3b", "qwen3.5-397b-a17b", "maxtext-omni-gemma3-qwen3", "weaver-mini",
        "weaver-max", "weaver-mini-diffuser", "ming-3.0-audio-tiny",
    )
    if config.model_name not in valid_mm_models and config.model_name != "default":
      raise ValueError(f"Multimodal is only supported for {valid_mm_models}, not {config.model_name}")
    if config.use_sft:
      if not config.sft_train_on_completion_only:
        raise ValueError("For multimodal SFT, `sft_train_on_completion_only` must be True.")
      if config.packing:
        raise ValueError("For multimodal SFT, `packing` is not yet supported.")
