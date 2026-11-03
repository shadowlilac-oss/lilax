# Lilax

Lilax brings a Hugging Face-style Python API to MaxText, a JAX library for training
and running large models on TPUs and GPUs.

> **Work in progress:** The APIs are source-authored; verification is limited to selected paths.
> Installation, runtime behavior, and feature coverage may vary by model and platform.

## Install

Python 3.12 or newer is required. For TPU, install from the repository root:

```sh
uv pip install './lilax[tpu]' --resolution=lowest
```

The NVIDIA GPU image `sha256:a77a1ac9ad9d132ccd7683572e711dbc4232109573c9d27b48b3817b87a0b4bc`
completed a 100-step SFT run with batch size 4, finite losses, and final loss 1.3456947803497314.
The run logged cuDNN runtime 9.26 versus compiled 9.27 incompatibility; it does not establish
general compatibility. The exact pinned Tunix package was installed without dependencies as an
explicit exception to its incompatible JAX version bound; its other requirements were resolved
normally. See
[container installation](TRAINING_OPERATIONS.md#container-installation).

Model loading also requires the `generation` extra. Training input pipelines require the
`tfds` extra. Optional training dependencies have known version conflicts, and a complete
installation across all paths has not been verified. See [implementation status](IMPLEMENTATION.md) and
[coverage](COVERAGE.md) before choosing an environment.

## Load a model

```python
from lilax.models import AutoModelForCausalLM

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        conversion_output_dir="/shared/maxtext-conversion",
    )
```

Choose a directory visible to every JAX process when Hugging Face weights need conversion.
The model graph is constructed when first used. Lilax supports Hugging Face loading
for selected models, not every model on the Hub.

## Fine-tune a model

Use `SFTConfig` for training settings and pass the dataset and tokenizer to `SFTTrainer`:

```python
from datasets import load_dataset
from transformers import AutoTokenizer
from lilax.trainers.post_training.sft import SFTConfig, SFTTrainer

if __name__ == "__main__":
    model_id = "Qwen/Qwen3-0.6B"
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir="./checkpoints",
            max_steps=100,
            per_device_train_batch_size=1,
            max_length=1024,
            train_data_columns=["messages"],
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
```

This example uses the model from the preceding section. Dataset formatting, hardware settings,
and optional dependencies depend on the selected training path. The 100-step run used batch size 4;
the example's batch size 1 has not been execution-verified.

For Tunix SFT, DPO, and distillation, `save_only_lora_params` and
`restore_only_lora_params` on their training configs default to `None` (selection follows LoRA
enablement). Set either to `False` for full-model checkpointing. QAT with LoRA requires both
to be `False`; adapter-only checkpoints omit QAT statistics. Supply a matching base model
when restoring adapter-only checkpoints. The native SFT lifecycle does not support these controls.

## More

- See the [training operations guide](TRAINING_OPERATIONS.md) for multi-node GPU launch, quantization,
  reporting, and profiling.
- Use `DPOTrainer`, `GRPOTrainer`, or `DistillationTrainer` for other post-training workflows.
- See [coverage](COVERAGE.md) for available model, training, and infrastructure APIs and their limits.
