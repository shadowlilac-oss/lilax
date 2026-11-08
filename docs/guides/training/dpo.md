# Train on preferences with DPO or ORPO

Use preference pairs to train a model toward a chosen completion and away from a rejected completion.

Training writes checkpoints under `output_dir`. The final `save_model` call replaces Lilax-owned native weights and updates named configuration and processing files while preserving unrelated destination files.

Constructing `DPOTrainer` initializes runtime placement and the training mesh before `train()`, while weight restoration remains deferred. Set device and mesh options first. See [runtime controls](../../reference/execution/runtime.md).

```python
from lilax import AutoModelForCausalLM, AutoTokenizer, DPOConfig, DPOTrainer
from lilax.data import Dataset

model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = Dataset.from_list([
	{"prompt": "What is 2 + 2?", "chosen": "4.", "rejected": "5."},
	{"prompt": "What is 3 + 3?", "chosen": "6.", "rejected": "7."},
])
args = DPOConfig(output_dir="./dpo", max_steps=10, per_device_train_batch_size=1)
trainer = DPOTrainer(model, args=args, train_dataset=dataset, processing_class=tokenizer)
result = trainer.train()
trainer.save_model("./preference-model")
```

The return value has this illustrative shape. Angle brackets represent your run's values:

```text
TrainOutput(global_step=<completed optimizer steps>, training_loss=<mean loss>, metrics={"train_loss": <mean loss>})
```

Read the loss with `result.metrics["train_loss"]`. See [logged metrics](../../reference/training/trainers.md#logged-metrics) for history and tracker output.

## Prepare preference pairs

The model configuration's `max_position_embeddings` sets runtime `max_target_length`, the total prompt-and-response capacity. To change it, load configuration with `AutoConfig.from_pretrained(model_id)`, set `config.max_position_embeddings`, and pass `config=config` to `AutoModelForCausalLM.from_pretrained`. Set it before loading or materializing the model, rather than changing `model.config` after construction.

| Columns | Example record | Prompt boundary |
| --- | --- | --- |
| `prompt`, `chosen`, `rejected` | `{"prompt": "What is 2 + 2?", "chosen": "4.", "rejected": "5."}` | The prompt is separate from the two candidate responses. |
| `chosen`, `rejected` | `{"chosen": "What is 2 + 2? 4.", "rejected": "What is 2 + 2? 5."}` | The longest common token prefix becomes the prompt. Each remaining suffix becomes a response. |

With tokenization enabled, these columns contain strings. With `tokenize=False` on the tokenizer's Lilax processing options, they contain token ID sequences. The three-column layout takes precedence when all three columns exist. Prompts are left-padded or trimmed from the left to `max_prompt_length`. When `None`, this limit resolves to `max_target_length // 2`. It must be strictly less than the resolved `max_target_length`, otherwise configuration raises `ValueError`. Responses are right-padded or trimmed from the right to the remaining sequence budget.

DPO disables packing when the model runtime has not explicitly set it. An explicit `packing=True` is preserved and raises `ValueError`.

Lilax does not automatically format DPO message lists. Convert conversational records to strings before constructing the trainer, rather than assuming every format from [TRL's DPO guide](https://huggingface.co/docs/trl/dpo_trainer) is accepted.

## Load preference pairs from the Hub

[UltraFeedback binarized](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized) stores `chosen` and `rejected` conversations. Using the model, tokenizer, and arguments above, format both conversations with the same tokenizer:

```python
from lilax.data import load_dataset

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

def format_pair(record):
	return {
		"chosen": tokenizer.apply_chat_template(record["chosen"], tokenize=False),
		"rejected": tokenizer.apply_chat_template(record["rejected"], tokenize=False),
	}

dataset = dataset.map(format_pair).select_columns(["chosen", "rejected"])
trainer = DPOTrainer(model, args=args, train_dataset=dataset, processing_class=tokenizer)
result = trainer.train()
print(result.global_step, result.metrics["train_loss"])
```

This uses the two-column layout. Shared response tokens also enter the inferred prompt if the responses begin identically. To control that boundary exactly, prepare explicit `prompt`, `chosen`, and `rejected` strings instead.

## Select ORPO

Using the model, tokenizer, and dataset above:

```python
args = DPOConfig(output_dir="./orpo", algorithm="orpo", max_steps=10, per_device_train_batch_size=1)
trainer = DPOTrainer(model, args=args, train_dataset=dataset, processing_class=tokenizer)
trainer.train()
```

`beta` and `label_smoothing` configure DPO. `orpo_lambda` configures ORPO. `max_prompt_length` limits the prompt portion.

See [trainer reference](../../reference/training/trainers.md), [LoRA](lora.md), and [checkpointing](checkpoints.md).
