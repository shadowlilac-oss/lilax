# Train LoRA adapters

Configure an adapter, attach it to a base model, and pass the adapted model to a trainer.

## Configure and attach the adapter

```python
from lilax import AutoModelForCausalLM, AutoTokenizer, SFTConfig, SFTTrainer
from lilax.data import Dataset
from lilax.peft import LoraConfig, PeftModel, get_peft_model

adapter_config = LoraConfig(
	r=8, lora_alpha=16,
	target_modules=r"decoder/layers(?:_[0-9]+|/[0-9]+)?/.*self_attention/(query|key|value|out)",
)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
model = get_peft_model(base, adapter_config)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
```

Rank and alpha here are illustrative. `target_modules` is a module-path regular expression. This Qwen3 example selects query, key, value, and output attention projections, including scanned and indexed layer paths. It excludes the MLP projections.

With `target_modules=""`, Lilax selects a pattern by the loaded model-name prefix. Qwen3 and Llama3.1 defaults include attention and MLP projections. Other registered families use their own attention and feed-forward paths. An unmatched family uses a generic attention-and-MLP pattern and logs a warning. Custom expressions are used unchanged. Lists of Hugging Face layer names are not accepted.

## Use QLoRA instead

To quantize the selected base weights while training adapters, replace the adapter setup above with:

```python
adapter_config = LoraConfig(
	r=8, lora_alpha=16,
	target_modules=r"decoder/layers(?:_[0-9]+|/[0-9]+)?/.*self_attention/(query|key|value|out)",
	lora_weight_qtype="nf4",
)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
model = get_peft_model(base, adapter_config)
```

Continue with the same tokenizer and training code below. Lilax requires `qwix>=0.1.8`. The `"nf4"` selection follows the [Qwix LoRA/QLoRA guide](https://qwix.readthedocs.io/en/latest/lora.html). `lora_weight_qtype` selects Qwix's weight quantization type. Optional `lora_tile_size` sets its block size along the innermost contraction axis. Leaving it as `None` uses no subchannel tiling. Supported types and tile-shape constraints depend on the installed Qwix version.

## Train

```python
dataset = Dataset.from_list([{"text": "A checkpoint preserves training state."}])
trainer = SFTTrainer(
	model, args=SFTConfig(output_dir="./lora-run", max_steps=10, per_device_train_batch_size=1, packing=False),
	train_dataset=dataset, processing_class=tokenizer,
)
trainer.train()
```

## Save and reload the adapter

```python
model.save_pretrained("./adapter", format="huggingface")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
restored = PeftModel.from_pretrained(base, "./adapter", adapter_config, format="huggingface")
```

An adapter checkpoint does not replace the base weights. Reload the base model and supply the same adapter settings, including rank, alpha, target modules, and any QLoRA options. Hugging Face adapter conversion requires registered configuration and parameter mappings for the base architecture and selected modules. `format="maxtext"` selects Orbax adapter storage and requires a fresh save destination. Loading defaults to that format, while saving defaults to `"huggingface"`.

## Keep QAT separate

Applying PEFT LoRA to a Qwix QAT model or QAT checkpoint raises `ValueError`. Select LoRA or QAT for that model.

## Evaluate before and after SFT with the public demo

Use the fresh model, tokenizer, and arguments in the [public SFT demo workflow](sft.md#evaluate-before-and-after-sft-with-the-public-demo). Before calling the helper, attach the `adapter_config` above with `model = get_peft_model(model, adapter_config)`. You can reuse the adapter settings and training dataset. Reuse an existing tokenizer only for the same model and chat template. An already materialized adapted model must have compatible target and prefill lengths, batch size, accumulation mode, and mesh. Otherwise, create the fresh base and adapter rather than change arguments after training. The helper does not attach an adapter itself. See the [demo reference](../../reference/training/sft-demo.md) for its evaluation and scoring contracts.

See the [adapter reference](../../reference/training/adapters.md) for accepted formats and adapter validation.
