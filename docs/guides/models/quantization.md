# Choose precision and quantization

Select a recipe when loading or constructing a model. Precision settings and quantization recipes are separate objects.

Check the [training recipe hardware requirements](precision-hardware.md) before selecting a mode.
The table lists documented NVIDIA SM targets.

For complete training examples, choose a task:

- [Train with NVFP4](nvfp4.md), including the default RHT recipe and NVFP4 without RHT.
- [Train with FP8](fp8.md), with Flax, Transformer Engine, JAX MXFP8, and Qwix alternatives.
- [Run quantization-aware training](qat.md), with Qwix rules, provider choices, and local saving.

`PrecisionConfig(dtype=...)` controls activation precision, while `weight_dtype` controls stored model weights.
`quantization_config` selects low-precision computation or QAT behavior. Selecting an NVFP4 or FP8 compute recipe
does not select a stored weight dtype. There is no `"nvfp4"` weight dtype in `PrecisionConfig`.

```python
from lilax import AutoModelForCausalLM
from lilax.configuration import PrecisionConfig, QwixQATConfig, QwixQATRule

qat = QwixQATConfig(rules=(QwixQATRule(weight_qtype="int8", act_qtype="int8"),))
qat_model = AutoModelForCausalLM.from_pretrained(
	"Qwen/Qwen3-8B", quantization_config=qat, precision_config=PrecisionConfig(weight_dtype="bfloat16"),
)
```

Pass `qat_model` to a [trainer](../training/sft.md). QAT inserts quantization behavior during model construction. The rule above is an illustrative selection, not an accuracy recommendation.

## Choose a compute recipe

```python
from lilax.configuration import AqtInt8Config

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", quantization_config=AqtInt8Config())
```

This selects AQT `int8`. For AQT mixed precision, pass `AqtMixedPrecisionConfig(quant_cfg_path=...)` with a nonempty path to an AQT JSON configuration. For Qwix model interception, select `QwixRecipeConfig` with an explicit mode, such as `quantization="int8"`; its default mode is empty.

Use `QwixFP8FullConfig` for Qwix `fp8_full` with the batch-split schedule disabled. With a configuration that already enables that schedule, select `BatchSplitQwixConfig` for explicitly wired Qwix operations or `BatchSplitManualConfig` for manual FP8 operations. The batch-split recipes reject a disabled schedule. A nonempty Qwix graph recipe rejects an enabled schedule.

Select a recipe supported by your hardware and model. See the [recipe modes, backend choices, and accepted compute values](../../reference/models/quantization.md#compute-recipes).

## Configure cache quantization or sparsity

```python
from lilax import AutoConfig
from lilax.configuration import KVCacheQuantizationConfig

config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
config.kv_cache_quantization_config = KVCacheQuantizationConfig(quantize_kvcache=True, kv_quant_dtype="int8")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", config=config)
```

`SparsityConfig` selects an N:M weight pattern and update timing through `config.sparsity_config`.

## Save QAT weights

`qat_model.save_pretrained("./qat-weights")` preserves QAT configuration for local reload. Hugging Face export and projection-layout conversion do not support QAT. PEFT LoRA cannot be attached to a QAT model or QAT checkpoint.

See [train and save a QAT model](qat.md) for a complete SFT example and provider limits.
