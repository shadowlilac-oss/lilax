# Train with NVFP4

Fine-tune `Qwen/Qwen3-8B` with Transformer Engine NVFP4 computation. Choose either the default recipe with
random Hadamard transforms, RHT, or the recipe without RHT. Both keep model weights in the floating-point
dtype selected by `PrecisionConfig`.

## Prepare the environment

Check the [recipe hardware table](precision-hardware.md#transformer-engine-recipes).

Use a Blackwell data-center GPU, such as B200, and the [NVIDIA GPU container setup](../../getting-started/environment.md#set-up-an-nvidia-gpu-container).
The checkout pins NVIDIA JAX `26.10-py3`. Its installer preserves Transformer Engine `2.20.2+6ea2a74a9`,
JAX `0.11.2.dev20260904+1b907ec45a`, and JAXlib `0.11.2.dev20261013`, and installs Flax `0.12.12`.
An ordinary JAX installation or the TPU extra does not provide this TE environment.

TE's [JAX recipe guide](https://docs.nvidia.com/deeplearning/transformer-engine/examples/te_jax_integration.html#quantization-recipes-at-a-glance)
requires Blackwell for NVFP4. The example uses BF16 activations and weights because
[RHT requires BF16 inputs and gradients](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html#random-hadamard-transform).
Allow memory for the full model, gradients, and optimizer state. NVFP4 compute does not make the checkpoint or optimizer four-bit.
Set device placement and mesh options before constructing the trainer, as described in [runtime controls](../execution/runtime.md).

## Choose one NVFP4 recipe

Run the shared imports, then select exactly one of the following alternatives before loading the model.

```python
from lilax import AutoModelForCausalLM, AutoTokenizer, SFTConfig, SFTTrainer
from lilax.configuration import ComputeRecipeConfig, PrecisionConfig
from lilax.data import Dataset

model_id = "Qwen/Qwen3-8B"
```

For TE's default NVFP4 recipe with RHT:

```python
recipe = ComputeRecipeConfig(quantization="te_nvfp4")
```

To disable RHT instead:

```python
recipe = ComputeRecipeConfig(quantization="te_nvfp4_no_rht")
```

The second selection constructs `NVFP4BlockScaling(disable_rht=True)`. It changes only the RHT option.
Both selections retain TE's default stochastic rounding and 2D weight quantization.
Disabling RHT does not disable stochastic rounding or make this recipe a general consumer-Blackwell training path.
See [TE's NVFP4 recipe](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html).

## Load the model and train

Use the selected `recipe` in this shared continuation:

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
	output_dir="./nvfp4-sft", max_steps=10, per_device_train_batch_size=1, max_length=256, packing=False,
)
trainer = SFTTrainer(model, args=args, train_dataset=dataset, processing_class=tokenizer)
result = trainer.train()
print(result.metrics["train_loss"])
trainer.save_model("./nvfp4-sft-model")
```

The short data and step count demonstrate API use, not an accuracy recipe. The saved model uses Lilax's local
checkpoint format. To resume training with optimizer state, use [training checkpoints](../training/checkpoints.md).

## Keep the recipe within its supported scope

This example uses a supported dense Qwen3 model. TE's generic `einsum` path is unsupported.
TE grouped MoE GEMMs use separate recipe settings.

Pass one recipe through `quantization_config`. Do not combine the two NVFP4 selections or substitute
`PrecisionConfig(weight_dtype="nvfp4")`. For other compute backends, use the [FP8 training guide](fp8.md).
For explicit forward quantization rules, use [QAT](qat.md).
