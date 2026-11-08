# Configure M3 component domains

Build component settings before you initialize a component or runtime. Configuration construction alone does not load weights.

## Compose execution and data settings

```python
from lilax.generation.m3_config import BaseConfig, DataConfig, DeviceConfig

config = BaseConfig(
	device=DeviceConfig(),
	data=DataConfig(max_sequence_length=512),
)
```

The sequence length is illustrative. Select dimensions for your data and component. Device mesh and output fields are listed in [execution settings](../../reference/components/m3-execution-settings.md).

## Set training and optimizer domains

Start from the public `BaseConfig` above. This composes the optimizer domain using its current learning rate and weight decay. Replace those values with choices for your component training task when needed.

```python
from lilax.generation.m3_config import OptimizerConfig, TrainerConfig

trainer = TrainerConfig(
	learning_rate=config.trainer.learning_rate,
	optimizer=OptimizerConfig(adam_weight_decay=config.trainer.optimizer.adam_weight_decay),
)
config = config.model_copy(update={"trainer": trainer})
```

These settings belong to component workflows. Use the [Trainer guide](../training/pretraining.md) for ordinary causal language model training.

## Change a diffusion domain

```python
from lilax.generation.wan import DiffusionConfig, load_wan_config

raw = load_wan_config()
config = DiffusionConfig.from_dict(raw)
config.scheduler.num_inference_steps = 50
```

This selects a schedule length without running diffusion. Keep the scheduler, latent shape, and model prediction contract consistent when assembling the [Wan components](wan-components.md). The [diffusion reference](../../reference/components/diffusion-settings.md) lists nested defaults.

Calling `process_config` resolves dtypes, dimensions, and output directories and initializes JAX. Keep that explicit step separate from inert configuration construction. The [M3 reference](../../reference/components/m3-components.md) describes the producer and resolver boundary.
