# Work with runtime and distributed controls

Models own their runtime. Trainers initialize the mesh and apply training settings before constructing model weights.

## Select sharding behavior

```python
from lilax import TrainingArguments
from lilax.distributed import MeshConfig, ParallelismConfig, ShardingConfig

args = TrainingArguments(
	output_dir="./distributed", parallelism_config=ParallelismConfig(
		mesh=MeshConfig(context_parallel_strategy="all_gather"),
		sharding=ShardingConfig(shard_optimizer_over_data=True),
	),
)
```

Pass these arguments to a trainer. Set physical process placement and parallelism degrees in an [execution profile](launch.md). A trainer cannot change the sharding of an already initialized mesh.

## Configure DiLoCo

```python
from lilax.distributed import DiLoCoConfig

args = TrainingArguments(output_dir="./diloco", diloco_config=DiLoCoConfig(enable_diloco=True))
```

`diloco_sync_period`, `diloco_outer_lr`, and `diloco_outer_momentum` control outer synchronization. Use them with a compatible distributed deployment.

## Run a forward pass in the model context

```python
from lilax import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
inputs = tokenizer("Explain sharding.", return_tensors="np", return_token_type_ids=False)
model.materialize(runtime=model.runtime)
with model.runtime.context(evaluation=True):
	logits = model(**inputs, enable_dropout=False)
print(logits.shape)  # (batch, sequence_length, vocab_size)
last_position_logits = logits[:, -1, :]
```

Materialization uses the original runtime. An evaluation context selects evaluation axis rules. This ordinary Qwen forward call returns the logits array directly. See [model return limits](../../reference/models/models.md#maxtextmodel) for configurations that return `None` or serving-cache tuples, and [runtime reference](../../reference/execution/runtime.md) for lifecycle methods.
