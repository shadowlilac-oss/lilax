# Configure an MLPerf offline run

Prepare benchmark arguments for an existing processed dataset and a compatible inference engine configuration.

```python
from lilax.benchmarks.mlperf.offline import offline_arguments

args = offline_arguments("./processed-openorca.parquet", model="llama2-70b", test_run=True)
```

This constructs arguments without launching a run. The helper selects the source benchmark's model-specific defaults. It does not load a checkpoint. Set checkpoint and tokenizer selection in the compatible inference configuration before running the workflow.

The processed dataset must include `tok_input` token-ID lists, `tok_input_length`, and `tok_output_length`. Token IDs must match the selected tokenizer. Before running, provide your benchmark's `mlperf.conf` and `user.conf` in the working directory, or pass their paths as `mlperf_conf` and `user_conf` to `offline_arguments`.

## Integrate an existing benchmark configuration

Build a runtime-source object from a local configuration and pass it directly to the workflow. This continues the `llama2-70b` argument selection above. `llama2-70b` has no registered Hub import. Create `./llama2-config/config.json` with the following content to select its packaged model configuration without downloading Hub metadata:

```json
{"model_name": "llama2-70b"}
```

Replace the parameter path below with a matching quantized checkpoint and the tokenizer path with its tokenizer.

Calling `run_offline_workflow` constructs a separate inference engine and runs LoadGen and accelerator work, even with `test_run=True`. That option reduces sample count rather than validating configuration only.

```python
from lilax import AutoConfig, AutoModelForCausalLM
from lilax.benchmarks.mlperf.offline import run_offline_workflow

config = AutoConfig.from_pretrained("./llama2-config", local_files_only=True)
model = AutoModelForCausalLM.from_config(config)
source = model.runtime.public_config.model_copy(update={
	"load_parameters_path": "/checkpoints/llama2-70b/0/items",
	"load_full_state_path": "", "tokenizer_path": "/tokenizers/llama2/tokenizer.model",
	"tokenizer_type": "sentencepiece",
})
result = run_offline_workflow(source, args)
```

The engine loads the selected checkpoint rather than live wrapper weights. The argument preset selects quantized checkpoint loading, KV quantization, and model-specific batch buckets. The workflow can score an accuracy log. Performance runs return `None`. See [runtime-source fields](../../reference/execution/runtime-source-fields.md) for checkpoint, tokenizer, and length selection.

For new model inference outside MLPerf, use [batch generation](../inference/generation.md). See [offline reference](../../reference/benchmarks/mlperf-offline.md).

## Assemble the exported XLA flag set

```python
from lilax.benchmarks.mlperf.xla_flags import build_xla_flag_str

flags = build_xla_flag_str()
```

The function returns the flag string without applying it.
