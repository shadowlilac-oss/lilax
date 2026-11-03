from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt


class MTP(BaseModel):
  """Multi-Token Prediction Configs."""

  mtp_num_layers: NonNegativeInt = Field(0, description="The number of auxiliary prediction layers to use for MTP.")
  mtp_loss_scaling_factor: NonNegativeFloat = Field(0.1, description="The scaling factor (lambda) for the MTP auxiliary loss.")
  mtp_eval_target_module: NonNegativeInt = Field(0, description="Specifies which MTP layer is used to calculate metrics.")
  mtp_reuse_input_embedding: bool = Field(False, description="Reuse the main decoder's token embeddings instead of a second lookup.")
  mtp_norm_main_hidden_state: bool = Field(
      False, description=("Apply the decoder's final norm to the main hidden state before passing it to MTP, as in"
                          " Megatron-LM where `final_layernorm` precedes the MTP `hnorm`.")
  )


class LogitsAndLoss(BaseModel):
  """Configuration for the final logits computation."""

  logits_via_embedding: bool = Field(False, description="If True, tie the embedding and unembedding matrices.")
  normalize_embedding_logits: bool = Field(True, description="If logits_via_embedding is true, normalize pre-softmax logits.")
  logits_dot_in_fp32: bool = Field(False, description="Use fp32 for the logits dot product for stability.")
  cast_logits_to_fp32: bool = Field(True, description="Whether to cast the final logits to fp32.")
  tie_lm_head_cotangents: bool = Field(
      False,
      description=(
          "Release the LM head's input and weight cotangents together in the backward pass (an optimization "
          "barrier), so the [batch, length, vocab] dlogits is not kept alive until the weight-gradient matmul."
      ),
  )
  lm_head_weight_grad_in_kernel_order: bool | None = Field(
      None,
      description=(
          "Trace the untied LM head's dot inside a jax.sharding.auto_axes region so that XLA folds the "
          "transpose back into the gradient dot and the weight gradient comes out in the kernel's stored "
          "axis order. Recovers the weight-gradient reduce-scatter that shard_mode=explicit otherwise loses, "
          "leaving the stored kernel, its initialization and the arithmetic untouched. No effect under "
          "shard_mode=auto. None means on for an untied model under shard_mode=explicit and off everywhere else."
      ),
  )
  lm_head_fsdp_as_vocab_tp: bool = Field(
      False,
      description=(
          "Shard the untied LM head on its vocab dimension using the FSDP axes (vocab TP over FSDP) instead of "
          "sharding its embed dimension on FSDP, so that the per-step FSDP all-gather of the head kernel and the "
          "reduce-scatter of its gradient move onto the hidden state -- reducing collective traffic when "
          "fsdp*per_device_batch*length < vocab_size (small-batch regimes) -- and the weight gradient needs no "
          "collective at all. The logits come out sharded on vocab across FSDP, so the loss builds its one-hot "
          "targets against a vocab-sharded iota rather than gathering them. Requires an untied head without MTP, "
          "vocab tiling, fp8 head weights, or training_objective=block_diffusion, and a vocab size divisible by "
          "the combined tensor and FSDP axes. Defaults to False (opt-in)."
      ),
  )
  final_logits_soft_cap: NonNegativeFloat | None = Field(None, description="Soft-cap value for the final logits. None or 0.0 means no cap.")
  z_loss_multiplier: float = Field(0.0, description="The multiplier for the z-loss (e.g., 1e-4). 0.0 to disable.")
  num_vocab_tiling: int = Field(1, description="Enables memory-saving optimization by tiling cross-entropy loss computation. >1 to enable.")
  vocab_tiling_ag_once: bool = Field(False, description="All gather the output head weight once before the tiled loss so the backward reuses it.")
