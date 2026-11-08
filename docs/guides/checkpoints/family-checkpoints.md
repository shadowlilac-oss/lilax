# Convert a family checkpoint

Use a family converter when you already have the original checkpoint files and need an Orbax checkpoint. For routine model loading and saving, use the [model workflow](../models/loading.md).

## Convert Qwen3 MoE files

Supply a local Hugging Face checkpoint directory. The converter reads its tensors and writes the destination checkpoint.

The conversion runs in a CPU subprocess and needs enough host memory for the model. This converter supports Qwen3 MoE layouts, including the selected 30B-A3B model, rather than dense Qwen3-8B.

```python
from lilax.checkpoints.conversion import convert_qwen3_moe

convert_qwen3_moe("/checkpoints/qwen3-30b-a3b-hf", "/checkpoints/qwen3-30b-a3b-orbax", "qwen3-30b-a3b")
```

The call returns `None` after writing the destination checkpoint.

## Select another family or layout

Use the matching [Qwen converter](../../reference/checkpoints/qwen-checkpoints.md), [Llama or Gemma converter](../../reference/checkpoints/llama-gemma-checkpoints.md), or [DeepSeek or GPT converter](../../reference/checkpoints/deepseek-gpt-checkpoints.md). Scanned and unscanned functions write different layer layouts. Match the output to the consuming model configuration.

The scanned Qwen3-Next file converter has a built-in architecture-key mismatch documented in its reference. Use the matching public model-loading workflow when that converter cannot satisfy your checkpoint task.

The Gemma, Gemma2, and Gemma3 converters read original Flax/Orbax checkpoints, not Hugging Face files.
For Hugging Face weights, use [`from_pretrained`](../models/loading.md) or
[`convert_checkpoint`](../../reference/training/checkpoints.md#convert_checkpoint).
For an original Gemma3 checkpoint, select the conversion with a size string:

```python
from lilax.checkpoints.conversion import convert_gemma3

convert_gemma3("/checkpoints/gemma3-4b-flax", "/checkpoints/gemma3-4b-orbax", "4b")
```

Both examples write checkpoint files. Use a separate destination for each conversion. A converter's accepted layout does not establish support for every training, generation, or export workflow.

## Convert weights without saving

The family weight functions return parameter dictionaries and allocate the converted weights in host memory. Use them only when your caller consumes that family's parameter layout. They do not return an `AutoModelForCausalLM` wrapper. The dimensions below match the converter's built-in `qwen3-30b-a3b` architecture and require that source model.

```python
from lilax.checkpoints.conversion import qwen3_moe_to_maxtext_weights

architecture = {
	"num_hidden_layers": 48, "num_attention_heads": 32, "num_key_value_heads": 4,
	"hidden_size": 2048, "head_dim": 128, "num_experts": 128, "moe_intermediate_size": 768,
}
parameters = qwen3_moe_to_maxtext_weights("/checkpoints/qwen3-30b-a3b-hf", architecture)
```

Here `architecture` is a caller-supplied mapping of model dimensions described in the [Qwen conversion reference](../../reference/checkpoints/qwen-checkpoints.md). File converters select the built-in architecture mapping for you.
