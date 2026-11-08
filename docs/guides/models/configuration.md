# Configure a model

Start from a supported model configuration so dimensions and family settings stay together.

```python
from lilax import AutoConfig, AutoModelForCausalLM
from lilax.configuration import PrecisionConfig

config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
config.precision_config = PrecisionConfig(weight_dtype="bfloat16")
model = AutoModelForCausalLM.from_config(config, seed=0)
```

`from_config` initializes new weights. Use `from_pretrained(..., config=config)` to load weights with compatible settings.

## Change a focused domain

```python
from lilax.configuration import ArchitectureConfig

architecture = config.architecture_config or ArchitectureConfig()
config.architecture_config = architecture.model_copy(update={"fused_qkv": True})
config.save_pretrained("./model-config")
```

Projection layout must match loaded weights. Use [checkpoint conversion](../training/checkpoints.md) for saved layouts.

## Scale a configuration for fresh training

```python
from lilax.configuration import ModelScalingConfig

scaled_config = ModelScalingConfig(global_parameter_scale=1).apply(config)
fresh_model = AutoModelForCausalLM.from_config(scaled_config, seed=0)
```

Scaling replaces effective dimensions with the scaling object's base dimensions. It does not resize pretrained weights. The scale must be a positive power of two.

Keep duration, optimizer settings, and output paths in trainer arguments. See the [configuration reference](../../reference/models/configuration.md).
