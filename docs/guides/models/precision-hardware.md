# Training recipe hardware requirements

The table lists documented hardware targets for dense training.

## Version scope

These Transformer Engine recipes use the project's NVIDIA JAX `26.10-py3` environment with
TE `2.20.2+6ea2a74a9`. Use the [NVIDIA GPU container setup](../../getting-started/environment.md#set-up-an-nvidia-gpu-container).

## Transformer Engine recipes

Transformer Engine recipes require NVIDIA CUDA GPUs. Each row links to the recipe's supported-device list.
These targets apply to dense training. Grouped MoE operations and inference have separate requirements.

Per-tensor FP8 requires CUDA 12.1 or newer and cuBLASLt 12.1.3.x or newer, as specified by
[TE JAX's versioned requirements](https://github.com/NVIDIA/TransformerEngine/blob/6ea2a74a9/transformer_engine/jax/quantize/helper.py#L66-L90).
MXFP8 and both NVFP4 recipes require CUDA 12.8, cuBLASLt 12.8.0, and JAX 0.5.3 or newer,
according to the [same versioned requirements](https://github.com/NVIDIA/TransformerEngine/blob/6ea2a74a9/transformer_engine/jax/quantize/helper.py#L82-L125).

| `ComputeRecipeConfig.quantization` | Documented NVIDIA target | Material requirements |
| --- | --- | --- |
| `te_fp8_delayedscaling` | [Ada and later, SM8.9+](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html#supported-devices) | amax history; gradient accumulation 1. |
| `te_fp8_currentscaling` | [Ada and later, SM8.9+](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html#supported-devices) | Scales use current tensors. |
| `te_mxfp8` | [Blackwell SM10.0/10.3](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/mxfp8/mxfp8.html#supported-devices) | Tensor rank at least 2. Last dimension and product of remaining dimensions each divisible by 32. |
| `te_nvfp4` | [Blackwell SM10.0/10.3](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html#supported-devices) | BF16 inputs and gradients for RHT; RHT and stochastic rounding, SR, enabled. |
| `te_nvfp4_no_rht` | [Blackwell SM10.0/10.3](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html#supported-devices) | Disables only RHT. SR and 2D weight quantization remain enabled. |

## QAT providers

Qwix's [ODML provider](https://qwix.readthedocs.io/en/stable/odml.html#odml-qat) simulates quantization with
quantize-then-dequantize operations. It does not require native low-bit GEMMs. NNX support is experimental,
and strict mode can reject unsupported operations. Selecting `provider="qt"` uses
[forward-only QAT](https://qwix.readthedocs.io/en/stable/qt.html#training-with-qwix),
with quantized forward operations and floating-point backpropagation.

Return to the complete [NVFP4](nvfp4.md), [FP8](fp8.md), or [QAT](qat.md) SFT example after selecting a recipe.
