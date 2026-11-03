def validate_deepstack_options(self):
  if self.deepstack_visual_indexes_for_vit and self.scan_layers:
    raise ValueError(
        "Deepstack visual embedding injection requires scan_layers=False. "
        "Set scan_layers=False in your config to use deepstack features."
    )


def validate_multimodal_options(self):
  if self.use_multimodal:
    if self.model_name in ("gemma4-e2b", "gemma4-e4b"):
      raise ValueError(f"Multimodal is not yet supported for {self.model_name}; only text inputs are supported.")
    valid_mm_models = (
        "gemma3-4b", "gemma3-12b", "gemma3-27b", "gemma4-26b", "gemma4-31b", "llama4-17b-16e",
        "llama4-17b-128e", "qwen3-omni-30b-a3b", "qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-30b-a3b",
        "qwen3.5-9b", "qwen3.5-35b-a3b", "qwen3.5-397b-a17b", "maxtext-omni-gemma3-qwen3",
        "weaver-mini", "weaver-max", "weaver-mini-diffuser", "ming-3.0-audio-tiny",
    )
    if self.model_name not in valid_mm_models and self.model_name != "default":
      raise ValueError(f"Multimodal is only supported for {valid_mm_models}, not {self.model_name}")
    if self.use_sft:
      if not self.sft_train_on_completion_only:
        raise ValueError("For multimodal SFT, `sft_train_on_completion_only` must be True.")
      if self.packing:
        raise ValueError("For multimodal SFT, `packing` is not yet supported.")
