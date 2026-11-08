# Construct M3 model components

M3 is Lilax's family of directly constructed Flax NNX model components, with component configuration, scanning, and checkpoint tools. `create_model` returns a component rather than the model object returned by `AutoModelForCausalLM`. Use the component APIs when you need to construct and manipulate these NNX objects directly.

## Create a GLM component

```python
from flax import nnx
from lilax.generation.m3 import create_model
from lilax.generation.m3_config import Glm5Config

config = Glm5Config.mini_config()
model = create_model(config, None, rngs=nnx.Rngs(0))
```

The mini configuration reduces layer count but retains large architecture dimensions. Construction still requires substantial memory. It initializes weights and does not load a pretrained checkpoint.

## Resolve settings from a public model

For supported Qwen component workflows, the model factory can provide the configuration producer:

```python
from lilax import AutoConfig, AutoModelForCausalLM
from lilax.generation.m3 import resolve_runtime_config

config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
public_model = AutoModelForCausalLM.from_config(config)
resolved = resolve_runtime_config(public_model.runtime.public_config)
```

The resolver accepts the runtime's configuration object. Passing the root `MaxTextConfig` directly raises `TypeError`. Do not treat those configurations as interchangeable. Resolution can initialize distributed JAX and configure its compilation cache according to the runtime settings; it does not construct a training mesh.

## Apply layer scanning

Continue with the resolved Qwen settings above. Construct a Qwen component, then enable scanning with its parameter scan axis.

```python
from flax import nnx
from lilax.generation.m3 import create_model, maybe_apply_scan
from lilax.generation.m3_config import BaseConfig

qwen = create_model(resolved, None, rngs=nnx.Rngs(0))
scan_config = BaseConfig.from_dict({"scan_layers": True, "param_scan_axis": resolved.param_scan_axis})
original_layers = qwen.decoder.layers
scanned = maybe_apply_scan(qwen, scan_config)
print(scanned is qwen, scanned.decoder.layers is original_layers)
print(type(scanned.decoder.layers).__name__, scanned.decoder.layers.num_layers)
```

Expected output for `Qwen/Qwen3-0.6B`:

```text
True False
ScannedQwen3DecoderLayer 28
```

Construction initializes the full Qwen component's weights without loading a checkpoint. Allow memory for these weights and for stacking them during scanning.

Scanning replaces `nnx.Sequential` layer containers in place and returns the model. Keep the returned object because scanning a root `nnx.Sequential` returns a replacement object. Consecutive structurally identical layers share a scan body; distinct runs remain separate. Parameters stack on `param_scan_axis`, capped at each parameter's rank. Other state stacks on axis zero. A negative or non-integer parameter axis raises `ValueError` when a run is scanned.

`scan_layers=False` returns the input unchanged. The GLM mini component above has no `nnx.Sequential` layer container, so this helper does not convert it. GLM uses its own stacked super-block execution controlled by `Glm5Config.scan_layers`; the default five-layer mini configuration has no complete super-block to scan. Use the Qwen component above to apply the generic scanning helper.

See [M3 reference](../../reference/components/m3-components.md).
