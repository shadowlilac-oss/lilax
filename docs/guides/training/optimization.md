# Configure optimization and memory

Set optimizer and schedule choices on trainer arguments. Compose focused configurations for additional controls.

```python
from lilax import SFTConfig, TrainingArguments
from lilax.training import LRSchedulerConfig, OptimizerConfig, RematConfig

args = SFTConfig(
	output_dir="./training", max_steps=100, per_device_train_batch_size=1,
	max_length=256, packing=False,
	gradient_accumulation_steps=4, optim="adamw", learning_rate=3e-5,
	lr_scheduler_config=LRSchedulerConfig(final_fraction=0.1),
	optimizer_config=OptimizerConfig(adam_eps_root=0.0),
	remat_config=RematConfig(remat_policy="full"),
)
```

Pass `args` to [SFTTrainer](sft.md). For [Trainer](pretraining.md), use `TrainingArguments` with the shared optimizer, scheduler, and rematerialization fields. Numerical choices illustrate configuration.

## Supply an Optax optimizer

To supply your own optimizer, build its learning rate into the transformation and pass the same schedule as the second element. This example uses the arguments above with a fresh Qwen3 model and two text records:

```python
import optax
from lilax import AutoModelForCausalLM, AutoTokenizer, SFTTrainer
from lilax.data import Dataset

model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
train_dataset = Dataset.from_list([
	{"text": "Gradient accumulation combines gradients across microbatches."},
	{"text": "A checkpoint lets you resume a training run."},
])
schedule = optax.constant_schedule(args.learning_rate)
optimizer = optax.adamw(
	learning_rate=schedule, b1=args.adam_beta1, b2=args.adam_beta2,
	eps=args.adam_epsilon, weight_decay=args.weight_decay,
)
trainer = SFTTrainer(
	model, args=args, train_dataset=train_dataset, processing_class=tokenizer, optimizers=(optimizer, schedule),
)
result = trainer.train()
print(result.metrics["train_loss"])
```

This downloads the selected checkpoint and tokenizer, initializes runtime placement when constructing the trainer, and trains for at most 100 optimizer steps with sequence length 256. Training allocates model and optimizer state and writes under `./training`. The printed loss comes from your run.

The example reuses the illustrative learning rate above and selects a constant schedule. Passing the schedule separately does not change the supplied transformation. `Trainer`, SFT, and DPO default each `None` element independently. SFT without multimodal input, DiLoCo, or optimizer host offload adds global-norm clipping when `max_grad_norm > 0`, including for a supplied optimizer. See the [optimizer pair contract](../../reference/training/trainers.md#optimizers).

## Select Muon

```python
from lilax.training import MuonConfig

args = TrainingArguments(
	output_dir="./muon", optim="muon", optimizer_config=OptimizerConfig(muon=MuonConfig()),
)
```

Choose `optax_muon` or `maxtext_muon` through `MuonConfig.implementation`. `SpikeSkipConfig` controls gradient spike skipping.

## Select parameters

For a Qwen3 parameter tree, `token_embedder/embedding` is the token embedding path. To update only that parameter and exclude it from AdamW weight decay:

```python
args = TrainingArguments(
	output_dir="./embeddings", optim="adamw",
	optimizer_config=OptimizerConfig(
		trainable_parameters=[r"token_embedder/embedding"],
		adamw_mask=[r"token_embedder/embedding"],
	),
)
```

Pass these arguments to the trainer with your Qwen3 model. All other parameter paths are frozen. The embedding receives gradient updates without weight decay. This example illustrates selection, not a fine-tuning recommendation.

Patterns use Python `re.search` on slash-separated parameter paths. `trainable_parameters` includes matching paths for updates. `adamw_mask` excludes matching paths from decay without freezing them. `None` or an empty list disables that list's filtering. See [optimizer and scheduler meanings](../../reference/training/optimization.md) for schedule duration, final rate, and WSD phases.

## Offload memory

```python
args = TrainingArguments(
	output_dir="./offload", remat_config=RematConfig(optimizer_memory_host_offload=True),
)
```

`parameter_memory_host_offload` controls parameter offload. Gradient checkpointing defaults to enabled. Explicit rematerialization settings select more detailed policies.

## Select an objective or batch ramp

`ObjectiveConfig` selects causal language modeling or block diffusion. For block diffusion, compose the objective with block-diffusion attention, a supported kernel, and `packing=False`, as shown in [block-diffusion SFT](block-diffusion.md#configure-sft-for-block-diffusion). Supply the model's actual mask token and a supported canvas policy. `BatchRampupConfig` configures a changing batch size.

See the [optimization reference](../../reference/training/optimization.md) for these configurations and [distributed controls](../execution/runtime.md) for sharding and DiLoCo.
