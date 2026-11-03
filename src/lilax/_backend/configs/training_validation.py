"""Validation of native training objectives."""

from lilax._backend.common.common_types import AttentionType


def validate_training_objective(config):
  if config.training_objective == "block_diffusion":
    if config.attention_type != AttentionType.BLOCK_DIFFUSION.value:
      raise ValueError("`training_objective='block_diffusion'` requires `attention_type='block_diffusion'`.")
    if config.block_diffusion_mask_id < 0 or config.block_diffusion_mask_id >= config.vocab_size:
      raise ValueError(
          f"`block_diffusion_mask_id` ({config.block_diffusion_mask_id}) must satisfy "
          f"0 <= block_diffusion_mask_id < vocab_size ({config.vocab_size})."
      )
    # Block-diffusion attention validation above rejects packing first.
    if config.packing:  # pragma: no cover
      raise ValueError("`training_objective='block_diffusion'` requires `packing=False`.")
    if config.mtp_num_layers > 0:
      raise ValueError("`training_objective='block_diffusion'` is not compatible with MTP.")
    if config.num_vocab_tiling > 1:
      raise ValueError("`training_objective='block_diffusion'` is not compatible with vocabulary tiling.")
    if config.dataset_type != "hf":
      raise ValueError("`training_objective='block_diffusion'` currently requires `dataset_type='hf'.")
    if config.use_dpo:
      raise ValueError("`training_objective='block_diffusion'` is not compatible with DPO.")
    if config.use_multimodal or config.use_audio:
      raise ValueError("`training_objective='block_diffusion'` currently supports text-only training.")
    valid_model_contracts = {
        ("same_position", "all_masked"),
        ("shifted", "seed_and_mask"),
    }
    model_contract = (config.block_diffusion_logit_alignment, config.block_diffusion_canvas_policy)
    if model_contract not in valid_model_contracts:
      raise ValueError(
          "Block-diffusion training supports only `same_position/all_masked` or `shifted/seed_and_mask`; "
          f"received `{model_contract[0]}/{model_contract[1]}`."
      )
    if config.block_diffusion_canvas_policy == "seed_and_mask" and config.causal_block_size < 2:
      raise ValueError("`block_diffusion_canvas_policy='seed_and_mask'` requires `causal_block_size >= 2`.")
