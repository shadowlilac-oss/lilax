# Log metrics and control training

Pass callback objects to a trainer. Use `report_to` for built-in experiment tracking.

With its default settings, `NonFiniteLossCallback` exits the process with `SystemExit(1)` when a step loss is nonfinite.

```python
from lilax import AutoModelForCausalLM, AutoTokenizer, SFTConfig, SFTTrainer
from lilax.callbacks import MetricsCallback, NonFiniteLossCallback
from lilax.data import Dataset

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
dataset = Dataset.from_list([{"text": "A checkpoint preserves training state."}])
trainer = SFTTrainer(
	model, args=SFTConfig(
		output_dir="./logged", max_steps=10, per_device_train_batch_size=1,
		packing=False, report_to="tensorboard", logging_steps=1,
	),
	train_dataset=dataset, processing_class=tokenizer,
	callbacks=[MetricsCallback(metrics_file="./metrics.jsonl"), NonFiniteLossCallback()],
)
trainer.train()
```

## Stop through a callback

```python
from lilax.callbacks import TrainerCallback

class StopAfterStep(TrainerCallback):
	def on_step_end(self, args, state, control, **kwargs):
		if state.global_step >= 5:
			control.should_training_stop = True
		return control

trainer.add_callback(StopAfterStep())
```

Register the callback before calling `train()`. `TargetEvalLossCallback` stops after an evaluation reaches its target. It requires evaluation data and evaluation events.

## Capture profiles

```python
from lilax.callbacks import ProfilerCallback
from lilax.profiling import ProfilingConfig

trainer.add_callback(ProfilerCallback(ProfilingConfig(profiler="xplane", profiler_steps=5)))
```

`RLProfilerCallback` selects invocation-based RL profiling. `MLPerfCallback` selects MLPerf logging. See [callback reference](../../reference/training/callbacks.md) for reporting and profiling options.

## Profile synthetic training with M3

`profile_m3` allocates a separate initialized model, mesh, and optimizer. It uses synthetic token inputs and ignores the trainer's model weights and dataset. Plan for those accelerator resources before calling it. Gradient accumulation greater than one raises `ValueError`.

Create a public `Trainer` for the supported `qwen3-0.6b` model. This example runs six synthetic steps and captures five after skipping the first step:

```python
from lilax import AutoConfig, AutoModelForCausalLM, Trainer, TrainingArguments
from lilax.callbacks import ProfilerCallback
from lilax.profiling import ProfilingConfig, profile_m3

config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
config.max_position_embeddings = 256
model = AutoModelForCausalLM.from_config(config)
trainer = Trainer(
	model, args=TrainingArguments(
		output_dir="./m3-profile", max_steps=6, per_device_train_batch_size=1,
		gradient_accumulation_steps=1, logging_dir="./m3-profile/traces",
	),
	callbacks=[ProfilerCallback(ProfilingConfig(
		profiler="xplane", skip_first_n_steps_for_profiler=1, profiler_steps=5,
	))],
)
last_loss = profile_m3(trainer)
print(last_loss)
```

`last_loss` is the last synthetic training loss as a float, not a benchmark score or the mean loss. The loop logs each step's loss, elapsed time, and tokens per second. With this XPlane selection, traces are written under `./m3-profile/traces` on process zero. These are output contracts, not measured results.

The synthetic loop builds its mesh over `jax.devices()`, independently of the trainer's mesh. Its global batch must be positive and divisible by the synthetic mesh's `data` and `fsdp` shard count. See [profile_m3](../../reference/training/callbacks.md#profile_m3) for the result and configuration contract.
