# Runtime-source fields for configured workflows

Configured decoding, the benchmark server, inference benchmarks, and checkpoint utilities consume a runtime-source object. The public producer is a supported model's `model.runtime.public_config`. It is distinct from the root `AutoConfig` model configuration, which describes model architecture and rejects run fields such as `prompt` and checkpoint destinations.

```python
from lilax import AutoConfig, AutoModelForCausalLM
from lilax.runtime import Runtime

config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_config(config)
source = model.runtime.public_config.model_copy(update={
	"load_parameters_path": "/checkpoints/qwen3-8b/0/items",
	"load_full_state_path": "", "prompt": "Explain checkpoints.",
})
runtime = Runtime(source).resolve_config()
```

Replace the path with an actual compatible parameter checkpoint. Creating this wrapper from configuration does not download Hub weights. Configured engines load their own selected weights and do not adopt another wrapper's live parameters. Pass `source` to `decode`, `MaxTextGenerator`, and `run_offline_workflow`. The offline engine factory requires the source object's explicit field selections. Pass `runtime.config` to APIs that require a resolved configuration, such as `run_benchmarks`.

The field entries cover these deliberate public workflows, not every runtime implementation setting. Defaults are declaration defaults. The selected model, loaded checkpoint metadata, and active execution profile can change the produced values. Inspect `source` for the values produced for your model. `model_copy(update=...)` copies the source fields. Runtime resolution applies the cross-field constraints.

## Checkpoint sources and destinations

Resolution derives `checkpoint_dir` as `<base_output_directory>/<run_name>/checkpoints/`. It is an output of resolution, not the field to set for selecting an export destination. `generate_decode_checkpoint` deletes an existing destination directory before writing. Use a destination separate from the source. Retain the original trainer runtime for extraction so architecture and optimizer state match.

The [storage profile fields](execution-profile-recovery.md) control chunking, concurrency, replication, and recovery. Standalone save-loop controls are in [checkpoint utility fields](runtime-source-fields-checkpoint-loop.md).

Full-state and parameter-only sources are mutually exclusive. Full-state loading requires checkpointing and is rejected by configured decode engines. Dynamic safetensors is unsupported with Pathways single-controller execution. Quantization status must match the checkpoint. Asynchronous saves must finish before their outputs are used.

<details>
<summary>Parameters</summary>

- `load_parameters_path` (`str`, optional, defaults to `""`). Parameter-only source loaded by each configured engine.
- `load_full_state_path` (`str`, optional, defaults to `""`). Full training-state source for recovery or extraction.
- `source_checkpoint_layout` (`Literal["orbax", "safetensors", "safetensors_dynamic"]`, optional, defaults to `"orbax"`). Source layout.
- `base_output_directory` (`str`, optional, defaults to `""`). Base run-artifact destination.
- `run_name` (`str`, optional, defaults to `""`). Run subdirectory and artifact identity.
- `enable_checkpointing` (`bool`, optional, defaults to `True`). Enable managers and saving.
- `async_checkpointing` (`bool`, optional, defaults to `True`). Save asynchronously.
- `checkpoint_period` (`int`, optional, defaults to `10000`). Steps between persistent saves.
- `checkpoint_is_quantized` (`bool`, optional, defaults to `False`). Source contains AQT-quantized parameters.
- `save_quantized_params_path` (`str`, optional, defaults to `""`). Destination for parameters quantized during loading.
- `force_unroll` (`bool`, optional, defaults to `False`). Unroll scanned layers for parameter-only extraction.
- `lora_input_adapters_path` (`str`, optional, defaults to `""`). Adapter directory for extraction with separate base and adapter outputs.

</details>

## Tokenizer and prompt input

The prefill allocation is part of total capacity. Increasing it reduces the remaining decode steps at fixed target length. Tokenizer length includes special tokens and any applied chat template. Checkpoint vocabulary and tokenizer IDs must agree.

The encoded prompt must fit the prefill allocation. Multi-stream decoding requires at least five global slots. Generator construction requires at least one.

<details>
<summary>Parameters</summary>

- `tokenizer_path` (`str | None`, optional, defaults to `None`). Tokenizer file, directory, or HF ID. Model preparation normally selects the registered tokenizer when omitted.
- `tokenizer_type` (`TokenizerType`, optional, defaults to `"sentencepiece"`). `sentencepiece`, `huggingface`, or `tiktoken`. Automatic registered Hub selection uses `huggingface`.
- `hf_access_token` (`str | None`, optional, defaults to `None`). Token for authenticated HF tokenizer loading.
- `use_chat_template` (`bool`, optional, defaults to `False`). Chat-template tokenization for configured engines. Does not select conversational SFT processing.
- `add_bos` (`bool`, optional, defaults to `True`). Add BOS in configured decode without a chat template.
- `prompt` (`str`, optional, defaults to `"I love to"`). Decode, scoring, and microbenchmark text. Generator calls supply their prompt list instead.
- `system_prompt` (`str`, optional, defaults to `""`). System message for vLLM chat construction.
- `max_prefill_predict_length` (`int`, optional, defaults to `64`). Padded prompt capacity.
- `max_target_length` (`int`, optional, defaults to `2048`). Total capacity including prefill. The public model producer projects `max_position_embeddings` here.
- `per_device_batch_size` (`int | float`, optional, defaults to `12`). Configured global batch size is `int(per_device_batch_size * jax.device_count())`.
- `autoregressive_decode_assert` (`str`, optional, defaults to `""`). Required generated prefix. Empty accepts any prefix.

</details>

## Prompt media

These fields apply only to supported multimodal models. The model producer supplies architecture fields such as vision dimensions and MRoPE. A text-model runtime cannot gain a vision or audio encoder by changing a path.

`build_chat_messages` handles system text and image markers. It does not load media bytes, format video or audio messages, or apply a tokenizer template. See [multimodal processing](../../guides/data/processing.md) for model-specific input preparation.

Qwen video grid caps must be supplied together or all left unset.

<details>
<summary>Parameters</summary>

- `use_multimodal` (`bool`, optional, defaults to `False`). Use the model's multimodal processing.
- `use_audio` (`bool`, optional, defaults to `False`). Use the model's audio encoder.
- `image_path` (`str`, optional, defaults to `""`). Image source. vLLM counts comma-separated paths and inserts image markers without loading bytes.
- `video_path` (`str`, optional, defaults to `""`). Configured video source.
- `audio_path` (`str`, optional, defaults to `""`). Configured audio source.
- `image_placeholder` (`str`, optional, defaults to `"<|image|>"`). Image marker.
- `video_placeholder` (`str`, optional, defaults to `"<|video|>"`). Video marker.
- `audio_placeholder` (`str`, optional, defaults to `"<|audio|>"`). Audio marker.
- `use_audio_in_video` (`bool`, optional, defaults to `False`). Extract video audio when supported.
- `video_max_grid_t` (`int | None`, optional, defaults to `None`). Temporal patch-grid cap before spatial merge.
- `video_max_grid_h` (`int | None`, optional, defaults to `None`). Height patch-grid cap before spatial merge.
- `video_max_grid_w` (`int | None`, optional, defaults to `None`). Width patch-grid cap before spatial merge.

</details>

## Decode sampling

These fields configure the engine. They are separate from `GenerationConfig` arguments on the model's `generate` method and from HTTP completion-request options. See [scoring and benchmark source fields](runtime-source-fields-inference.md) for the remaining configured workflows.

<details>
<summary>Parameters</summary>

- `decode_sampling_strategy` (`SamplingStrategy`, optional, defaults to `"greedy"`). `greedy`, `weighted`, `nucleus`, `topk`, or `composite`.
- `decode_sampling_nucleus_p` (`int | float`, optional, defaults to `-1.0`). `-1` disables nucleus filtering. vLLM maps nonpositive values to `1.0`.
- `decode_sampling_top_k` (`int`, optional, defaults to `0`). Zero disables top-k filtering. vLLM maps nonpositive values to `-1`.
- `decode_sampling_temperature` (`float`, optional, defaults to `1.0`). Sampling temperature.

</details>
