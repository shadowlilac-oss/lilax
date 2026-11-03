def validate_block_diffusion_options(self):
  if self.attention_type == "block_diffusion":
    if self.packing:
      raise ValueError("Block-diffusion attention does not support packing; set `packing=False`.")
    if self.attention not in ("autoselected", "dot_product", "flash"):
      raise ValueError("Block-diffusion attention is supported only by dot_product attention and TPU Splash attention.")
    if self.attention in ("autoselected", "flash") and self.hardware != "tpu":
      raise ValueError(
          "Block-diffusion attention with attention='autoselected' or attention='flash' requires hardware='tpu'; "
          "use attention='dot_product' on other hardware."
      )
  if self.training_objective == "block_diffusion":
    if self.attention_type != "block_diffusion":
      raise ValueError("`training_objective='block_diffusion'` requires `attention_type='block_diffusion'`.")
    if self.block_diffusion_mask_id < 0 or self.block_diffusion_mask_id >= self.vocab_size:
      raise ValueError(
          f"`block_diffusion_mask_id` ({self.block_diffusion_mask_id}) must satisfy "
          f"0 <= block_diffusion_mask_id < vocab_size ({self.vocab_size})."
      )
    if self.packing:
      raise ValueError("`training_objective='block_diffusion'` requires `packing=False`.")
    if self.mtp_num_layers > 0:
      raise ValueError("`training_objective='block_diffusion'` is not compatible with MTP.")
    if self.num_vocab_tiling > 1:
      raise ValueError("`training_objective='block_diffusion'` is not compatible with vocabulary tiling.")
    if self.dataset_type != "hf":
      raise ValueError("`training_objective='block_diffusion'` currently requires `dataset_type='hf'`.")
    if self.use_dpo:
      raise ValueError("`training_objective='block_diffusion'` is not compatible with DPO.")
    if self.use_multimodal or self.use_audio:
      raise ValueError("`training_objective='block_diffusion'` currently supports text-only training.")
    model_contract = (self.block_diffusion_logit_alignment, self.block_diffusion_canvas_policy)
    if model_contract not in {("same_position", "all_masked"), ("shifted", "seed_and_mask")}:
      raise ValueError(
          "Block-diffusion training supports only `same_position/all_masked` or `shifted/seed_and_mask`; "
          f"received `{model_contract[0]}/{model_contract[1]}`."
      )
    if self.block_diffusion_canvas_policy == "seed_and_mask" and self.causal_block_size < 2:
      raise ValueError("`block_diffusion_canvas_policy='seed_and_mask'` requires `causal_block_size >= 2`.")
  return self
