# Train a causal language model

Use `Trainer` for language-model training. Load pretrained weights to continue training, or initialize from configuration to train new weights.

Constructing `Trainer` initializes runtime placement and the training mesh before `train()`, while weight construction remains deferred. Set device and mesh options first. See [runtime controls](../../reference/execution/runtime.md).

```python
from lilax import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from lilax.data import Dataset

model_id = "Qwen/Qwen3-8B"
config = AutoConfig.from_pretrained(model_id)
model = AutoModelForCausalLM.from_config(config, seed=0)
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = Dataset.from_list([{"text": "A checkpoint preserves the state of a training run."}])
args = TrainingArguments(output_dir="./pretraining", max_steps=10, per_device_train_batch_size=1)
trainer = Trainer(model, args=args, train_dataset=dataset, processing_class=tokenizer)
output = trainer.train()
trainer.save_model("./pretrained-model")
```

The numerical choices illustrate API use. A production corpus and duration depend on your training task.

This model's runtime-source configuration (`model.runtime.public_config`) defaults to `packing=True`, so `Trainer` packs records into training sequences. `TrainingArguments` has no packing field. To disable packing, insert this block immediately after model creation, before resolving or initializing its runtime and before constructing `Trainer`:

```python
model.runtime.public_config = model.runtime.public_config.model_copy(update={"packing": False})
model.runtime.config = model.runtime.public_config
```

Use this setup only before runtime resolution or initialization. See [configuration objects](../../reference/execution/runtime.md#configuration-objects) for runtime-source and resolved configuration.

## Initialize through a function

```python
trainer = Trainer(
	args=args, model_init=lambda: AutoModelForCausalLM.from_config(config, seed=0),
	train_dataset=dataset, processing_class=tokenizer,
)
```

Supply `model` or `model_init`. Omitting both raises `ValueError`.

## Evaluate named datasets

```python
validation = Dataset.from_list([{"text": "Evaluation measures the model on held-out data."}])
trainer = Trainer(model, args=args, train_dataset=dataset, eval_dataset={"heldout": validation}, processing_class=tokenizer)
metrics = trainer.evaluate()
print(metrics)
```

Named datasets receive separate metric prefixes. Keep evaluation finite. See [optimization](optimization.md), [training arguments](../../reference/training/training-arguments.md), and [checkpointing](checkpoints.md).

The printed dictionary has the following illustrative shape. `<float>` marks a value computed by evaluation, not a measured result supplied by this guide:

```text
{'eval_heldout_loss': <float>, 'eval_heldout_perplexity': <float>,
 'eval_heldout_total_weights': <float>, 'eval_heldout_moe_lb_loss': <float>,
 'eval_heldout_indexer_loss': <float>, 'eval_heldout_mtp_loss': <float>,
 'eval_heldout_mtp_acceptance_rate_percent': <float>, 'eval_heldout_z_loss': <float>}
```
