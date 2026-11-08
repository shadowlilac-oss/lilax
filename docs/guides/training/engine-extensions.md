# Drive a training engine

Use the public engine protocol when integrating an existing engine with a caller-owned training loop. For ordinary model training, use [Trainer](pretraining.md).

Run the snippets below in order for one language-model update. The example uses Lilax's existing
loss and payload adapter, with a tokenized prompt and completion. It does not define an RL objective.

## Construct the concrete engine

Use a public model's runtime as the configuration producer. Runtime initialization initializes JAX and device
resources. This example initializes a separate model from configuration, without loading pretrained weights.

```python
import jax
import numpy as np
from lilax import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from lilax.trainers.engine import (
	MaxTextTrainingEngine, RLTrainerPayload, make_router_replay_loss_fn, router_replay_gen_model_input_fn,
)

config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_config(config)
runtime = model.runtime.initialize()
engine = MaxTextTrainingEngine(runtime.config, mesh=runtime.mesh)
loss_fn = make_router_replay_loss_fn(
	runtime.config, dropout_rng=jax.random.PRNGKey(runtime.config.init_weights_seed),
)
engine.with_loss_fn(loss_fn, has_aux=True).with_gen_model_input_fn(router_replay_gen_model_input_fn)
```

The constructor accepts resolved configuration (`runtime.config`), not the protocol's `TrainingConfig` or the model's `config`. See [configuration objects](../../reference/execution/runtime.md#configuration-objects). LoRA is unsupported. The objective returns a `LossOutput` or `WeightedMetric`. A two-element loss and auxiliary-metrics result is also accepted when its first element is a `WeightedMetric` or its second element contains `xent_sum` and `total_weights`. A plain scalar alone is rejected. The engine does not inherit a wrapper's pending Hub checkpoint.

## Supply a payload

Tokenize one prompt and completion into batched arrays. This language-model loss ignores the required
`advantages` field, so the example supplies zeros. No expert-routing override is supplied.

```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
prompt = tokenizer(["A checkpoint lets you"], return_tensors="np")
completion = tokenizer([" resume a training run."], return_tensors="np")

payload = RLTrainerPayload(
	prompt_ids=prompt["input_ids"], prompt_mask=prompt["attention_mask"],
	completion_ids=completion["input_ids"], completion_mask=completion["attention_mask"],
	advantages=np.zeros_like(completion["input_ids"], dtype=np.float32),
)
```

The adapter joins prompt and completion rows and creates next-token targets with validity masks.
The existing loss trains on valid next-token targets across the joined row. For custom RL training,
replace the loss and adapter with your own objective and matching inputs.

<a id="configure-extension-functions"></a>

## Update model state

`with_loss_fn` selects the objective supplied above. `with_gen_model_input_fn` maps each payload to its keyword inputs. Using the configured engine and payload above, run one update and close the engine:

```python
try:
	engine.fwd_bwd(payload)
	step = engine.update()
	metrics = engine.get_metrics()
	print(step, metrics.weighted_metrics["loss"].unreduced_sum.shape)
finally:
	engine.close()
```

For a stream of matching payloads, repeat `fwd_bwd`, `update`, and `get_metrics` inside the `try` block. `compile(payload)` can prepare a representative shape before the first update. Actual compilation policy belongs to the engine implementation.

After one update, the printed structure is `1 (1,)`. The step is an integer and the loss buffer has one entry
for the single forward/backward call.

## Save and synchronize

Use `save_checkpoint` and `restore_checkpoint` for engine state. `prepare_weight_sync` stages weights and `release_weight_sync` releases staging resources. Keep these operations in the lifecycle that owns the engine.

See [engine reference](../../reference/training/engine-extensions.md) for the protocol and concrete constructor boundary.
