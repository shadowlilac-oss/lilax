# Prepare convergence recipe settings

Calculate training settings from dataset and schedule records. This example uses the shipped C4/DeepSeek
benchmark records and its existing sequence length, device count, and batch size. These are recipe values,
not tuning recommendations for another run.

```python
from lilax.benchmarks.convergence import c4_mlperf_hp, deepseek_671b_hp, setup_model_convergence

settings = setup_model_convergence(
	{"max_target_length": 8192}, c4_mlperf_hp, deepseek_671b_hp, num_devices=256, global_batch_size=1024,
)
print(settings["steps"], settings["eval_steps"], settings["per_device_batch_size"])
```

For your own experiment, supply its tuning dictionary, `DatasetHParams`, and `ConvHParams` records instead.
The tuning dictionary must include `max_target_length`. The function calculates settings without loading
the dataset or starting training.

The result is a new dictionary. The [reference](../../reference/benchmarks/convergence-recipes.md) lists record arguments and the mutation contracts of the individual helpers. Pass model and training settings through the corresponding public configuration domains when preparing an ordinary [Trainer workflow](../training/pretraining.md).
