"""Validation of native multi-token prediction configuration."""


def validate_mtp_main_hidden_state(config):
  if config.mtp_reuse_input_embedding and config.use_multimodal:
    raise ValueError("`mtp_reuse_input_embedding` does not support multimodal inputs.")
  if config.mtp_norm_main_hidden_state:
    if config.num_vocab_tiling > 1:
      raise ValueError(
          "`mtp_norm_main_hidden_state` is not supported with `num_vocab_tiling > 1`: under vocab tiling the"
          " decoder skips `apply_output_head` in train mode and `decoder_norm` runs later inside"
          " `vocab_tiling_nnx_loss`, so the MTP block would still receive the un-normalized hidden state."
      )
    if config.mhc_expansion_rate > 1:
      raise ValueError(
          "`mtp_norm_main_hidden_state` is not supported with `mhc_expansion_rate > 1`: the MTP layers expect"
          " the expanded 4D hyper-connection state, which the decoder's final norm collapses to 3D."
      )
