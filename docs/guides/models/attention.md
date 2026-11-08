# Configure attention and position encoding

Keep model attention semantics separate from kernel selection. Start with the loaded model's settings.

```python
from lilax import AutoConfig, AutoModelForCausalLM
from lilax.configuration import KernelConfig

config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
kernel = config.kernel_config or KernelConfig()
config.kernel_config = kernel.model_copy(update={"attention": "dot_product"})
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", config=config)
```

This selects dot-product attention while preserving loaded MoE and Splash kernel settings. See the [kernel choices and hardware limits](../../reference/models/configuration.md#attention-position-and-kernels) before selecting an accelerated kernel.

## Preserve position settings while changing one option

```python
from lilax.configuration import PositionConfig, RopeConfig

config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B")
position = config.position_config or PositionConfig()
rope = position.rope or RopeConfig()
config.position_config = position.model_copy(update={"rope": rope.model_copy(update={"rope_use_scale": False})})
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", config=config)
```

This disables the Llama3.1 frequency-scaling correction for a comparison with unscaled RoPE. It changes the checkpoint's position behavior, rather than extending its trained context. Apply position changes before model construction.

## Configure a family-specific attention branch

`AttentionConfig` composes general attention, MoBA, MLA, compressed attention, sparse indexing, and Llama4 settings. Select branches already supported by your loaded model. These objects do not turn Qwen3 into a different architecture.

See [attention reference](../../reference/models/attention.md) and [model configuration](../../reference/models/configuration.md).
