# Logits and multi-token prediction

Return to the [configuration reference](configuration.md).

Import these configurations from `lilax.configuration`. Keep the settings loaded with the matching model family. See [model configuration](../../guides/models/configuration.md) for loading and updating a focused domain.

These are constructor defaults. A loaded checkpoint can supply different values. Changing architecture dimensions or routing does not convert existing weights.

## MTPConfig

Import from `lilax.configuration`. Pydantic model whose keyword fields configure auxiliary multi-token prediction.

<details>
<summary>Parameters</summary>

- `mtp_num_layers` (`NonNegativeInt`, optional, defaults to `0`). Number of auxiliary prediction layers.
- `mtp_loss_scaling_factor` (`NonNegativeFloat`, optional, defaults to `0.1`). Weight of the auxiliary loss.
- `mtp_eval_target_module` (`NonNegativeInt`, optional, defaults to `0`). MTP layer used for metrics.
- `mtp_reuse_input_embedding` (`bool`, optional, defaults to `False`). Decoder token embedding reuse instead of a second lookup.
- `mtp_norm_main_hidden_state` (`bool`, optional, defaults to `False`). Decoder final normalization before the main hidden state enters MTP.

</details>

## LogitsConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures output logits and loss computation. Vocabulary sharding over FSDP requires an untied head without MTP, vocabulary tiling, FP8 head weights, or block diffusion. Vocabulary size must divide evenly over the tensor and FSDP axes.

<details>
<summary>Parameters</summary>

- `logits_via_embedding` (`bool`, optional, defaults to `False`). Tied embedding and unembedding matrices.
- `normalize_embedding_logits` (`bool`, optional, defaults to `True`). Pre-softmax logit normalization when embeddings are tied.
- `logits_dot_in_fp32` (`bool`, optional, defaults to `False`). Float32 logits dot product. Combining `True` with [`QwixFP8FullConfig.quantize_logits_proj=True`](quantization.md#qwixfp8fullconfig) raises `ValueError`.
- `cast_logits_to_fp32` (`bool`, optional, defaults to `True`). Float32 casting of final logits.
- `tie_lm_head_cotangents` (`bool`, optional, defaults to `False`). Joint release of output-head input and weight gradients so the vocabulary-sized logit gradient need not remain live until weight-gradient computation.
- `lm_head_weight_grad_in_kernel_order` (`bool | None`, optional, defaults to `None`). Stored weight-axis order for the untied output-head gradient under explicit sharding. `None` enables this only for an untied, explicitly sharded head. No effect in auto mode.
- `lm_head_fsdp_as_vocab_tp` (`bool`, optional, defaults to `False`). Vocabulary sharding of the untied output head over FSDP, subject to the restrictions above.
- `final_logits_soft_cap` (`NonNegativeFloat | None`, optional, defaults to `None`). Final-logit soft cap. `None` or `0.0` disables it.
- `z_loss_multiplier` (`float`, optional, defaults to `0.0`). Weight of the logit-normalizer z-loss. Zero disables it.
- `num_vocab_tiling` (`int`, optional, defaults to `1`). Vocabulary tiles for cross-entropy computation.
- `vocab_tiling_ag_once` (`bool`, optional, defaults to `False`). Single all-gather of output-head weights before tiled loss for reuse in the backward pass.

</details>

