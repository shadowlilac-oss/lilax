# Experimental conversion agents

Import from `lilax.experimental.agent`. See [agent guide](../../guides/checkpoints/conversion-agent.md).

Install `lilax[conversion-agent]` for the required `google-genai>=1.0.0` SDK.

## CheckpointConversionAgent

Import from `lilax.experimental.agent`. Class.

```python
CheckpointConversionAgent(
	directory, target_model="gemma3", max_retries=3, context_directory=None, resource_directory=None,
	model_id="gemini-2.5-pro",
)
```

Creates an external-service client. `GEMINI_API_KEY` is required or construction raises `ValueError`.
The model ID default is not a guarantee of service availability. Stage checks interpret generated text and do not establish code correctness.

<details>
<summary>Parameters</summary>

- `directory` (`str | os.PathLike`). The working directory containing generated `outputs`.
- `target_model` (`str`, optional, defaults to `"gemma3"`). Model prompt context.
- `max_retries` (`int`, optional, defaults to `3`). Limit on plan and mapping attempts.
- `context_directory` (`str | os.PathLike | None`, optional, defaults to `None`). Parameter JSON context. Omission uses the resource directory's model context.
- `resource_directory` (`str | os.PathLike | None`, optional, defaults to `None`). Prompts and DSL resources. Omission uses bundled resources.
- `model_id` (`str`, optional, defaults to `"gemini-2.5-pro"`). The external model.

</details>

### analyze_model_structures

```python
agent.analyze_model_structures()
```

Reads parameter context, calls the service, and writes `outputs/analysis.txt`.

Returns analysis text.

### verify_dsl

```python
agent.verify_dsl()
```

Requires saved analysis. Calls the service and writes `outputs/proposed_dsl.txt`.

Returns proposed DSL text.

### plan_conversion

```python
agent.plan_conversion()
```

Requires saved analysis. Calls the service to generate and check a plan. Writes `outputs/plan.json`.
Exhausted attempts raise `RuntimeError`.

Returns plan text.

### generate_param_mapping

```python
agent.generate_param_mapping()
```

Requires saved analysis and parameter context. Calls the service to generate and check mapping code.
Writes `outputs/param_mapping.py`. Exhausted attempts raise `RuntimeError`.

Returns mapping code.

### generate_shape_mapping

```python
agent.generate_shape_mapping()
```

Requires saved analysis and HF parameter context. Calls the service to generate and check shape code.
Writes `outputs/hf_shape.py`. Exhausted attempts raise `RuntimeError`.

Returns shape code.

### generate_hook_functions

```python
agent.generate_hook_functions()
```

Requires saved plan and parameter mapping. Calls the service and writes `outputs/hook_fn.py`.

Returns hook code.

### analyze

```python
agent.analyze()
```

Writes analysis and proposed DSL through the corresponding stages.

Returns proposed DSL text.

### generate

```python
agent.generate()
```

Requires prior analysis. Writes the plan and generated mapping files through the generation stages.

Returns hook code.

### generate_text

```python
agent.generate_text(prompt, tools=None)
```

Calls the external service.

Returns response text.

<details>
<summary>Parameters</summary>

- `prompt` (`str`). Service prompt contents.
- `tools` (`list[google.genai.types.Tool] | None`, optional, defaults to `None`). Service tool configuration.

</details>

## BaselinePromptChain

Import from `lilax.experimental.agent`. Class.

```python
BaselinePromptChain(
	directory, target_model="gemma3", max_retries=3, context_directory=None, resource_directory=None,
	model_id="gemini-2.5-pro",
)
```

Inherits the [constructor parameters and stages](#checkpointconversionagent) from `CheckpointConversionAgent`.

Its additional method generates mapping and hook files through a baseline prompt chain. Generated-text checks do not establish correctness.

### run_chain

```python
chain.run_chain(max_retries=3)
```

Calls the service and writes generated artifacts. Exhausted checks raise `RuntimeError`.

Returns `None`.

<details>
<summary>Parameters</summary>

- `max_retries` (`int`, optional, defaults to `3`). Limit on mapping attempts for this chain.

</details>

## Artifact evaluation

### compressed_size

```python
compressed_size(data: bytes, method: str = "gzip") -> int
```

Returns compressed bytes count. The `gzip` choice uses zlib compression, not a gzip file container.
Unknown methods raise `ValueError`.

<details>
<summary>Parameters</summary>

- `data` (`bytes`). Uncompressed data.
- `method` (`str`, optional, defaults to `"gzip"`). Compression method, `gzip`, `bz2`, or `lzma`.

</details>

### estimate_kolmogorov

```python
estimate_kolmogorov(filepath: str) -> dict
```

Returns `original_size`, `gzip_size`, `bz2_size`, `lzma_size`, and `approx_k_complexity`. The last is the minimum compressed size.

<details>
<summary>Parameters</summary>

- `filepath` (`str`). The binary file to compress.

</details>

### evaluate_conversion

```python
evaluate_conversion(ground_truth_path: str, generated_path: str, prompt_path: str | None = None) -> str
```

Reads artifacts and calls the external service. Requires `GEMINI_API_KEY`.

Returns response text.

<details>
<summary>Parameters</summary>

- `ground_truth_path` (`str`). Reference code.
- `generated_path` (`str`). Generated code.
- `prompt_path` (`str | None`, optional, defaults to `None`). A template with `ground_truth` and `dsl_chain` placeholders. Omission uses the bundled template.

</details>

## Parameter context generation

### save_parameter_context

```python
save_parameter_context(config, hf_model_config, agent_directory)
```

Loads weights, creates an HF model to collect shapes, and writes parameter JSON on process zero.
Changes JAX PRNG configuration and `TF_CPP_MIN_LOG_LEVEL`.

Returns the output path.

Its first argument requires engine-compatible resolved configuration (`model.runtime.resolve_config().config`), not the root HF-style model configuration. Full-state checkpoint input is rejected. This operation can allocate model memory and read checkpoints.

<details>
<summary>Parameters</summary>

- `config` (`object`). Resolved engine settings and checkpoint selection from `runtime.config`. See [configuration objects](../execution/runtime.md#configuration-objects).
- `hf_model_config` (`str`). Hugging Face repository ID or local configuration source passed to `transformers.AutoConfig.from_pretrained`, not an already loaded configuration object.
- `agent_directory` (`str | os.PathLike`). The fallback output root when `base_output_directory` is empty.

</details>
