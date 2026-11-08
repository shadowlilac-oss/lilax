# Train with FP8

Fine-tune `Qwen/Qwen3-8B` with one FP8 compute recipe. Select one recipe, then use the
shared model-loading and SFT code below. These choices quantize computation while keeping stored model weights
in the floating-point dtype selected by `PrecisionConfig`.

## Prepare your environment

Check the [FP8 and related training recipe hardware requirements](precision-hardware.md).

Start with [environment setup](../../getting-started/environment.md). For NVIDIA recipes, use the checkout's
`26.10-py3` GPU container. It preserves TE `2.20.2+6ea2a74a9`, JAX `0.11.2.dev20260904+1b907ec45a`,
and JAXlib `0.11.2.dev20261013`, and installs Flax `0.12.12`.

Configure placement before constructing `SFTTrainer`. Allow memory for floating-point weights, gradients,
and optimizer state. See [runtime controls](../execution/runtime.md) and [SFT data formats](../training/sft.md).

## Choose one recipe

Run these shared imports. Execute exactly one recipe assignment from the alternatives below, then continue
with [Load the model and train](#load-the-model-and-train). Recipes are alternatives, not settings to stack.

```python
from lilax import AutoModelForCausalLM, AutoTokenizer, SFTConfig, SFTTrainer
from lilax.configuration import ComputeRecipeConfig, JaxMXConfig, PrecisionConfig, QwixFP8FullConfig, QwixRecipeConfig
from lilax.data import Dataset

model_id = "Qwen/Qwen3-8B"
```

### Use Flax FP8 operations

For E4M3FN forward operands and E5M2 output gradients:

```python
recipe = ComputeRecipeConfig(quantization="fp8")
```

The FNUZ variant targets AMD MI300/MI325 GPUs with a compatible ROCm JAX and Flax environment.
AMD documents these [CDNA3 FP8 formats](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html).
The project's GPU container uses NVIDIA CUDA; supply the ROCm environment separately.
For E4M3FNUZ forward operands and E5M2FNUZ output gradients:

```python
recipe = ComputeRecipeConfig(quantization="nanoo_fp8")
```

These paths use Flax delayed scaling with amax history.
Lilax rejects gradient accumulation greater than one for both selections. Configured `decode` rejects
`"fp8"` and `"nanoo_fp8"` with `AssertionError`; see [configured decoding](../../reference/components/configured-decoding.md#decode).

### Use Transformer Engine FP8

Use a documented [TE JAX hardware target](precision-hardware.md#transformer-engine-recipes) for per-tensor FP8.
Choose delayed scaling with amax history:

```python
recipe = ComputeRecipeConfig(quantization="te_fp8_delayedscaling")
```

Or choose current scaling, which computes scales from the current tensors:

```python
recipe = ComputeRecipeConfig(quantization="te_fp8_currentscaling")
```

For block-scaled MXFP8 on the listed Blackwell targets:

```python
recipe = ComputeRecipeConfig(quantization="te_mxfp8")
```

These map to `DelayedScaling`, `Float8CurrentScaling`, and `MXFP8BlockScaling`, respectively.
See the [recipe-specific hardware and library requirements](precision-hardware.md#transformer-engine-recipes).
Delayed scaling rejects gradient accumulation greater than one in Lilax. MXFP8 uses blocks of 32 elements.
TE's generic `einsum` path is not implemented here. These dense-model examples do not configure TE grouped MoE GEMMs.
`te_no_quant` selects TE without quantization and is not an FP8 recipe.

### Use JAX MXFP8

Use a JAX build with `jax.nn.scaled_dot_general` and `get_scaled_dot_general_config("mxfp8")`:

```python
recipe = JaxMXConfig(quantization="jax_mxfp8")
```

The gradient mode defaults to the operand mode, so this selection uses MXFP8 for operands and gradients.
Lilax requires automatic sharding, a dense model, and non-FP8 stored weights. Forward, input-gradient,
and weight-gradient contracting dimensions must each contain whole blocks of at least 32 elements.
This path supports training only and does not implement `einsum`.

`JaxMXConfig` exposes MXFP8 and MXFP4. Selecting `jax_mxfp4` or gradient mode `mxfp4` requires a JAX build
providing that scaled-dot mode; Lilax requires JAX 0.12.0 or newer for MXFP4. The NVIDIA `26.10-py3` environment
pins JAX `0.11.2.dev20260904`, so use a compatible JAX environment for MXFP4. Use [TE NVFP4](nvfp4.md) for Lilax's NVFP4 recipes.
In [JAX 0.12.0](https://docs.jax.dev/en/latest/changelog.html#jax-0-12-0-october-19-2026),
MXFP4 runs natively on AMD gfx950 GPUs and uses a dequantize-and-dot fallback elsewhere.

### Use Qwix graph interception

Use Qwix with a matching JAX environment. Qwix documents [XLA CPU, GPU, and TPU targets](https://github.com/google/qwix#readme).
Choose E4M3FN for weights, activations, and backward quantization:

```python
recipe = QwixRecipeConfig(quantization="fp8_e4m3")
```

Or choose E5M2 for all three:

```python
recipe = QwixRecipeConfig(quantization="fp8_e5m2")
```

The Qwix spelling `"fp8"` is also accepted and selects the same qtypes as `"fp8_e4m3"`:

```python
recipe = QwixRecipeConfig(quantization="fp8")
```

To use E4M3FN forward operands and E5M2 backward quantization across the configured transformer-layer operations:

```python
recipe = QwixFP8FullConfig()
```

For Qwix interception using Flax delayed scaling with E4M3FN and E5M2:

```python
recipe = QwixRecipeConfig(quantization="fp8_gpu")
```

For the corresponding E4M3FNUZ and E5M2FNUZ operations on AMD MI300/MI325, use the same ROCm environment described above:

```python
recipe = QwixRecipeConfig(quantization="fp8_nanoo")
```

`"nanoo_fp8"` belongs to `ComputeRecipeConfig`; `"fp8_nanoo"` belongs to `QwixRecipeConfig`.
They are not interchangeable literals. Qwix `"fp8"` and `"fp8_gpu"` reject gradient accumulation greater than one.
Graph recipes require the batch-split schedule to be disabled, as in this example.
See [Qwix quantized training](https://qwix.readthedocs.io/en/stable/qt.html) for forward and backward quantization semantics.

## Load the model and train

Continue with the one `recipe` selected above. Keep `gradient_accumulation_steps=1` so the shared example
also works with recipes that reject accumulation.

```python
model = AutoModelForCausalLM.from_pretrained(
	model_id, quantization_config=recipe,
	precision_config=PrecisionConfig(dtype="bfloat16", weight_dtype="bfloat16"),
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = Dataset.from_list([
	{"text": "Gradient accumulation combines gradients across microbatches."},
	{"text": "A checkpoint lets you resume a training run."},
])
args = SFTConfig(
	output_dir="./fp8-sft", max_steps=10, per_device_train_batch_size=1, gradient_accumulation_steps=1,
	max_length=256, packing=False,
)
trainer = SFTTrainer(model, args=args, train_dataset=dataset, processing_class=tokenizer)
result = trainer.train()
print(result.metrics["train_loss"])
trainer.save_model("./fp8-sft-model")
```

The records and step count illustrate API use, not convergence or performance. Select precision before model
materialization. A trainer cannot change precision settings on an already materialized graph.

## Keep specialized recipes separate

`BatchSplitQwixConfig()` selects explicitly wired Qwix `fp8_full` operations. `BatchSplitManualConfig()`
selects manual `fp8_full` operations. Both require a separately enabled, supported batch-split schedule.
They are not drop-in alternatives for the ordinary SFT continuation above. `QwixFP8FullConfig()` uses graph
interception without that schedule.

`ServeFP8WeightConfig` is a serving recipe requiring FP8 stored weights. It is not the training recipe shown here.
Setting `PrecisionConfig(weight_dtype="float8_e4m3fn")` alone does not select any FP8 training computation.
For forward quantization rules with floating-point backpropagation, use the [QAT guide](qat.md).
