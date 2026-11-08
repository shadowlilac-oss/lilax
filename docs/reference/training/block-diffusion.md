# Block-diffusion arrays

Import from `lilax.generation.block_diffusion`. See [preparation guide](../../guides/training/block-diffusion.md).

## corrupt_tokens

```python
corrupt_tokens(
	inputs: np.ndarray, validity_mask: np.ndarray, rng: np.random.Generator, *, block_size: int,
	mask_id: int, min_noise: float = 1e-3, logit_alignment: str = "same_position",
	canvas_policy: str = "all_masked", axis: int = 1,
) -> BlockDiffusionCorruptionResult
```

Returns `BlockDiffusionCorruptionResult(inputs, corruption_mask, targets_loss_mask)`. All arrays preserve the input shape. Valid tokens are sampled independently within blocks. At least one eligible token is selected in each eligible block.

Shape mismatch, invalid axis, an empty sequence axis, nonpositive block size, negative mask ID, or noise outside `(0, 1]` raise `ValueError`.

Only `same_position/all_masked` and `shifted/seed_and_mask` are supported pairs. Seed-and-mask requires a block size of at least two.

<details>
<summary>Parameters</summary>

- `inputs` (`np.ndarray`). Input token IDs.
- `validity_mask` (`np.ndarray`). Eligible token mask with the input shape.
- `rng` (`np.random.Generator`). Corruption random generator.
- `block_size` (`int`). Bidirectional block size.
- `mask_id` (`int`). Nonnegative mask token ID.
- `min_noise` (`float`, optional, defaults to `1e-3`). Minimum corruption noise in `(0, 1]`.
- `logit_alignment` (`str`, optional, defaults to `"same_position"`). Logit alignment from a supported pair above.
- `canvas_policy` (`str`, optional, defaults to `"all_masked"`). Canvas policy from a supported pair above.
- `axis` (`int`, optional, defaults to `1`). Sequence axis.

</details>

## align_logits_to_targets

```python
align_logits_to_targets(
	logits: jax.Array, alignment: str, positions: jax.Array | None = None,
	validity_mask: jax.Array | None = None,
) -> jax.Array
```

Aligns logits to targets. Position arrays must match batch and sequence dimensions.

Returns aligned JAX logits. `same_position` returns the supplied logits. `shifted` gathers preceding positions using optional packed metadata.

<details>
<summary>Parameters</summary>

- `logits` (`jax.Array`). Model logits.
- `alignment` (`str`). `same_position` or `shifted`.
- `positions` (`jax.Array | None`, optional, defaults to `None`). Packed sequence positions.
- `validity_mask` (`jax.Array | None`, optional, defaults to `None`). Valid token positions.

</details>

## Trainer compatibility

`ObjectiveConfig(training_objective="block_diffusion")` requires `AttentionParameters(attention_type="block_diffusion")` within `AttentionConfig.general` and `SFTConfig(packing=False)`. `KernelConfig.attention` accepts `dot_product`, `autoselected`, or `flash` for this objective; `autoselected` and `flash` require TPU hardware.

Training requires HF text-only data, no MTP layers, no vocabulary tiling, and no DPO. `block_diffusion_mask_id` must be in `[0, vocab_size)`. Objective alignment and canvas choices accept only the pairs listed above; `seed_and_mask` also requires `AttentionParameters.causal_block_size >= 2`. See [SFT configuration composition](../../guides/training/block-diffusion.md#configure-sft-for-block-diffusion).

SFT requires zero `z_loss_multiplier`, `load_balance_loss_weight`, and `indexer_loss_scaling_factor`. Loaded model defaults can enable these losses. For example, MoE load balancing is configured through `config.moe_config.general.load_balance_loss_weight`; disable it before model construction for this objective.
