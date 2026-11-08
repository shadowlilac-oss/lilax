# Generate conversion code with an experimental agent

The conversion agent calls an external text-generation service and writes conversion artifacts. Its output is generated code, not a converted checkpoint or a deterministic verification result.

## Prepare context and resources

Install the `conversion-agent` extra into your project environment with `python -m pip install -e '.[conversion-agent]'` from the checkout's `lilax` directory. It supplies `google-genai>=1.0.0`.

Provide a context directory containing `maxtext_params.json` and `hf_params.json`, plus the prompt templates, DSL, and pitfalls resources for your conversion task. Configure `GEMINI_API_KEY` in the environment before constructing an agent.

## Analyze and generate

Supply `work_directory` for generated output, the prepared `context_directory` and `resource_directory`, and your chosen external service's `service_model_id`. The following sends parameter context and prompt resources to that service and writes intermediate and generated files under `work_directory/outputs`.

```python
from lilax.experimental.agent import CheckpointConversionAgent

agent = CheckpointConversionAgent(
	work_directory, context_directory=context_directory, resource_directory=resource_directory,
	model_id=service_model_id,
)
agent.analyze()
hook_code = agent.generate()
```

All paths and the service model ID are caller inputs. Analysis writes intermediate files. Generation reads the analysis, writes mapping artifacts, and returns hook code. Inspect generated artifacts before using them.

## Run the baseline chain

`BaselinePromptChain` accepts the same constructor inputs and provides `run_chain(max_retries=3)`. It writes generated files and returns `None`.

## Compare generated artifacts

`evaluate_conversion(reference_path, generated_path, prompt_path)` requests a model's textual judgment. It is not a pass/fail proof. `estimate_kolmogorov(filepath)` reports compression-size proxies and does not call a model from its function body.

See [agent reference](../../reference/checkpoints/conversion-agent.md).
