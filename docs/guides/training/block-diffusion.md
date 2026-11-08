# Prepare block-diffusion training tokens

Use block-diffusion helpers when preparing a token canvas and its supervision masks. These helpers operate on arrays independently of a trainer.

```python
import numpy as np
from lilax.generation.block_diffusion import corrupt_tokens

tokens = np.array([[1, 2, 3, 4]], dtype=np.int32)
valid = np.ones_like(tokens, dtype=bool)
result = corrupt_tokens(tokens, valid, np.random.default_rng(0), block_size=2, mask_id=0)
print(result.inputs)
print(result.targets_loss_mask)
```

Token IDs and mask ID above describe an illustrative vocabulary. For a real model, select its actual mask token ID.

## Preserve alignment

Use `same_position` with `all_masked`, or `shifted` with `seed_and_mask`. The second pair preserves seed tokens and requires blocks of at least two tokens.

`align_logits_to_targets` aligns model logits with clean targets. Keep the alignment consistent with the canvas and the trainer's `ObjectiveConfig`.

## Configure SFT for block diffusion

Compose the model's attention and kernel settings with the trainer objective. Here `model_id` is your supported text checkpoint whose tokenizer already defines a genuine mask token in the model vocabulary. Supply `train_dataset` with `text` or `prompt` and `completion` columns, as in [SFT](sft.md). Do not use this workflow with a tokenizer that has no mask token, and do not add or borrow a token to fill that gap.

Before loading weights, check that the selected configuration has no MTP layers or vocabulary tiling. Block-diffusion SFT also requires `z_loss_multiplier`, `load_balance_loss_weight`, and `indexer_loss_scaling_factor` to be zero. Disable nonzero values in the loaded configuration before `AutoModelForCausalLM.from_pretrained`, through `config.logits_config.z_loss_multiplier`, `config.moe_config.general.load_balance_loss_weight`, and `config.attention_config.indexer.indexer_loss_scaling_factor` in the domains that are present. Training below allocates model and optimizer state and writes under the configured output directory.

```python
from lilax import AutoConfig, AutoModelForCausalLM, AutoTokenizer, SFTConfig, SFTTrainer
from lilax.configuration import AttentionConfig, AttentionParameters, KernelConfig
from lilax.training import ObjectiveConfig

tokenizer = AutoTokenizer.from_pretrained(model_id)
mask_token_id = tokenizer.mask_token_id
config = AutoConfig.from_pretrained(model_id)
config.attention_config = AttentionConfig(general=AttentionParameters(attention_type="block_diffusion"))
config.kernel_config = KernelConfig(attention="dot_product")
model = AutoModelForCausalLM.from_pretrained(model_id, config=config)
args = SFTConfig(
	output_dir="./block-diffusion", max_steps=10, per_device_train_batch_size=1, packing=False,
	objective_config=ObjectiveConfig(
		training_objective="block_diffusion", block_diffusion_mask_id=mask_token_id,
		block_diffusion_logit_alignment="same_position", block_diffusion_canvas_policy="all_masked",
	),
)
trainer = SFTTrainer(model, args=args, train_dataset=train_dataset, processing_class=tokenizer)
trainer.train()
```

The mask ID must satisfy `0 <= mask_token_id < config.vocab_size`; the example does not add a token. Numerical choices illustrate API use.

Block diffusion requires text-only HF data, `packing=False`, no MTP layers, and no vocabulary tiling. It does not support DPO. The supported attention kernels are `dot_product`, `flash`, and `autoselected`. The latter two require TPU hardware; this example selects `dot_product` without making a performance recommendation.

The objective accepts only `same_position` with `all_masked`, or `shifted` with `seed_and_mask`. For the second pair, set `causal_block_size` to at least two on `AttentionParameters`. The default block size is 32. Keep these choices consistent with any separately prepared arrays.

See [block-diffusion reference](../../reference/training/block-diffusion.md) for array and training contracts.
