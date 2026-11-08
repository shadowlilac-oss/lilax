# Update model-family settings

Load a configuration for the supported family before changing its focused settings. Family-specific options preserve the architecture's own conventions.

## Change a Qwen recurrent-attention option

```python
from lilax import AutoConfig

config = AutoConfig.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")
config.qwen_config = config.qwen_config.model_copy(update={"use_qk_norm_in_gdn": True})
config.save_pretrained("./qwen-next-config")
```

This saves configuration only. It does not allocate the model's weights. Construct or load the model after selecting compatible settings. See [Qwen settings](../../reference/models/qwen.md).

## Configure expert routing and kernels

`MoEConfig.general` owns expert counts, routing precision, capacity, and communication choices. `MoEConfig.deepseek` owns family-specific shared experts, routing groups, and bias updates. `KernelConfig.moe` selects expert kernels and tiles.

Preserve these objects from the loaded configuration when changing a field. Changing expert counts or dimensions does not resize checkpoint weights. See [expert settings](../../reference/models/moe.md).

For ragged all-to-all routing, start with a supported MoE checkpoint whose loaded configuration has `moe_config.general` and `kernel_config.moe`. The example preserves the loaded capacity, ragged-buffer factor, and experts per token. The default negative ragged-buffer factor uses worst-case space. Layer-local fallback requires a workload-selected positive factor, as shown after the configuration setup.

Before constructing the model, meet these prerequisites.

- Enable sparse matmul and ragged sort, with no embedding chunks or random routing.
- Disable ring of experts and TE MoE for ragged all-to-all.
- Keep `num_moe_token_chunks=1`.
- Keep `moe_config.deepseek.defer_small_all_reduces=False` when that domain is present.
- Disable options that require ring of experts, including `moe_log_max_load_ratio` and `moe_tc_ragged_sort`.

```python
from lilax import AutoConfig

config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V3", model_name="deepseek3-671b")

general = config.moe_config.general.model_copy(update={
	"use_ring_of_experts": False,
	"te_moe_block": False,
	"use_ragged_sort": True,
	"use_random_routing": False,
})
config.moe_config = config.moe_config.model_copy(update={"general": general})
kernels = config.kernel_config.moe.model_copy(update={"sparse_matmul": True, "num_moe_emb_chunks": 0})
config.kernel_config = config.kernel_config.model_copy(update={"moe": kernels})
```

The result is the same model configuration with the updated routing domain and preserved DeepSeek settings. To enable layer-local fallback, supply `ragged_buffer_factor` as a positive factor selected for your workload, then update the prepared routing domain:

```python
config.moe_config.general = config.moe_config.general.model_copy(update={
	"ragged_buffer_factor": ragged_buffer_factor, "moe_dropless_fallback": "layer",
})
```

`config.save_pretrained("./moe-config")` writes configuration files only. Model construction still needs compatible weights and execution settings. Fallback also needs expert parallelism above one in the execution profile. With one expert shard, runtime resolution disables it.

A negative `capacity_factor` disables capacity-based token drops. A negative ragged-buffer factor allocates worst-case space. With a positive ragged-buffer factor, `moe_dropless_fallback=None` allows overflow drops, `"step"` replays the whole training step, and `"layer"` switches only the overflowing layer to dropless computation. See [fallback prerequisites](../../reference/models/moe.md#dropless-fallback-prerequisites) before selecting another branch.

## Freeze encoders for multimodal training

```python
from lilax.configuration import VisionConfig

config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", model_name="qwen3-vl-4b")
vision = config.vision_config or VisionConfig()
config.vision_config = vision.model_copy(update={"freeze_vision_encoder_params": True})
```

`AudioConfig.freeze_audio_encoder_params` controls the audio encoder for supported audio models. Match processor settings to the model. See [vision and audio fields](../../reference/models/media-models.md).

## Select auxiliary model behavior

`MTPConfig` selects multi-token prediction layers and loss weighting. `LogitsConfig` controls output precision, tying, soft caps, and vocabulary tiling. These settings must match the architecture and training objective. See [logits and prediction](../../reference/models/model-loss.md).

Supported families also expose [Engram](../../reference/models/engram.md), [Ming](../../reference/models/ming.md), and [hyperconnection](../../reference/models/hyperconnections.md) settings. Configure the matching model's domain, rather than attaching unrelated architecture components to a loaded model.
