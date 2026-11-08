# Run quantization-aware training

Fine-tune `Qwen/Qwen3-8B` with Qwix quantization-aware training, QAT. This example keeps floating-point model
weights and applies int8 quantization to selected forward operations. It does not load an int8 checkpoint
or select an [FP8 compute recipe](fp8.md).

## Prepare the environment

Check the [QAT provider hardware requirements](precision-hardware.md#qat-providers).
Int8 QAT does not require FP8 hardware.

Use the [project environment](../../getting-started/environment.md) for your JAX accelerator.
Lilax requires `qwix>=0.1.8`. Qwix documents [XLA CPU, GPU, and TPU targets](https://github.com/google/qwix#readme).
Configure devices and the training mesh before constructing `SFTTrainer`.

## Choose one QAT provider

Run the shared setup below, then select one provider before loading the model.

```python
from lilax import AutoModelForCausalLM, AutoTokenizer, SFTConfig, SFTTrainer
from lilax.configuration import PrecisionConfig, QwixQATConfig, QwixQATRule
from lilax.data import Dataset

model_id = "Qwen/Qwen3-8B"
rules = (QwixQATRule(
	module_path="decoder/.*layers.*", op_names=("dot_general",), weight_qtype="int8", act_qtype="int8",
),)
```

For forward-only QAT using Qwix's quantized-operations provider:

```python
qat = QwixQATConfig(provider="qt", rules=rules)
```

This selection quantizes the selected forward operations and keeps backpropagation in floating point.
It uses Qwix's QAT mode, not quantized backpropagation. See
[Qwix's QAT example](https://qwix.readthedocs.io/en/stable/qt.html#training-with-qwix).

For ODML fake quantization instead, replace the assignment above with:

```python
qat = QwixQATConfig(provider="odml", rules=rules)
```

`"odml"` is the default provider. It inserts quantize-then-dequantize operations and defaults to strict mode,
which can reject unsupported operations in the selected graph. Qwix describes its NNX ODML support as experimental.
See [ODML QAT](https://qwix.readthedocs.io/en/stable/odml.html#odml-qat).
Selecting this provider does not add a Lilax LiteRT export workflow.

## Load the model and train

Continue with exactly one `qat` assignment above:

```python
model = AutoModelForCausalLM.from_pretrained(
	model_id, quantization_config=qat, precision_config=PrecisionConfig(weight_dtype="bfloat16"),
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = Dataset.from_list([
	{"text": "Gradient accumulation combines gradients across microbatches."},
	{"text": "A checkpoint lets you resume a training run."},
])
args = SFTConfig(output_dir="./qat-sft", max_steps=10, per_device_train_batch_size=1, max_length=256, packing=False)
trainer = SFTTrainer(model, args=args, train_dataset=dataset, processing_class=tokenizer)
result = trainer.train()
print(result.metrics["train_loss"])
trainer.save_model("./qat-sft-model")
```

Rules match module paths and operation names. This example targets transformer-layer `dot_general` operations.

## Save and reload locally

`save_model` above saves the model in Lilax's local checkpoint format, including its QAT configuration.
Reload that directory without replacing the saved recipe:

```python
model = AutoModelForCausalLM.from_pretrained("./qat-sft-model")
```

For model-only saving outside the trainer, `model.save_pretrained("./qat-weights")` preserves QAT configuration too.
Use [training checkpoints](../training/checkpoints.md) to resume optimizer state and training progress.

## QAT limits

Do not combine QAT with an existing nonempty Qwix precision recipe. PEFT LoRA cannot be attached to a QAT model
or QAT checkpoint. Hugging Face export and projection-layout conversion do not support QAT.

ODML rejects `tile_size`. The `qt` provider rejects non-default ODML options, including
`disable_per_channel_weights=True`, fixed input or output ranges, and `strict=False`.
Keep those defaults when using `provider="qt"`. See the [quantization reference](../../reference/models/quantization.md)
for all public rule fields.
