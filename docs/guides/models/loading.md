# Load models from the Hub or disk

Load a supported repository ID or a local model directory with the same factory.

```python
from lilax import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-8B"
config = AutoConfig.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

## Choose a snapshot

Use the same revision for configuration, model, and tokenizer. Replace `revision` with a branch, tag, or commit from the repository.

```python
revision = "main"
config = AutoConfig.from_pretrained(model_id, revision=revision)
model = AutoModelForCausalLM.from_pretrained(model_id, config=config, revision=revision)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
```

`cache_dir` selects the Hub snapshot cache. `local_files_only=True` prevents downloads. `token` controls access to the model snapshot.

## Allow space for first-use conversion

The first operation that needs Hub weights converts them in a CPU subprocess and stores a full additional weight copy. Allow disk space for this converted copy as well as the downloaded snapshot.

The converted weights go to `<root>/lilax-models/<sha256>/0/items`. The root is the runtime's `base_output_directory` when set, otherwise `cache_dir`, otherwise `HF_HUB_CACHE`. A trainer's `output_dir` supplies `base_output_directory`. Reuse requires matching configuration, snapshot path, revision, weight dtype, simulated CPU-device count, and conversion settings. Missing or incompatible saved conversion metadata raises an error. An existing runtime `load_parameters_path` takes precedence over conversion.

For multihost conversion, use a shared runtime `base_output_directory`, for example through a shared trainer `output_dir`. Without this output directory, multihost Hub conversion raises `ValueError`. `cache_dir` alone does not satisfy this requirement.

## Inspect supported operations

```python
from lilax import get_model_capabilities

capabilities = get_model_capabilities(config.model_name)
print(capabilities.hf_import, capabilities.hf_export, capabilities.hf_repo_id)
```

Use `HF_IMPORT_MODELS`, `HF_EXPORT_MODELS`, and `HF_REPOS` to discover registered identities. Loading an arbitrary repository does not register a new architecture.

See the [supported-model table](../../reference/models/models.md#supported-model-identities) for the exact repository IDs and import and export capabilities.

## Disambiguate a shared repository

`Qwen/Qwen3-VL-4B-Instruct` is registered as both `qwen3-vl-4b` and `weaver-mini`. Select the identity through `AutoConfig`, then pass that configuration to the model loader. The model loader itself has no `model_name` argument.

```python
model_id = "Qwen/Qwen3-VL-4B-Instruct"
config = AutoConfig.from_pretrained(model_id, model_name="qwen3-vl-4b")
model = AutoModelForCausalLM.from_pretrained(model_id, config=config)
```

Likewise, `Qwen/Qwen3.8-Flash-Next` needs either `qwen3.8-flash-next` or `qwen3.8-flash-next-mini` selected through `AutoConfig`. Environment overrides can affect configuration loading or raise collisions. See [environment overrides](../../reference/models/models.md#environment-overrides).

## Initialize new weights

```python
fresh_model = AutoModelForCausalLM.from_config(config, seed=0)
```

This initializes weights instead of restoring pretrained weights. See [configuration](configuration.md) and the [model reference](../../reference/models/models.md).

## Construct a native-only model

For an identity without Hub import support, create a local directory such as `./llama2-config` containing this `config.json`.

```json
{"model_name": "llama2-7b"}
```

Load the native defaults through `AutoConfig`, then construct new weights through the public model factory.

```python
from lilax import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("./llama2-config", local_files_only=True)
model = AutoModelForCausalLM.from_config(config, seed=0)
```

Weights remain deferred until materialization. Passing the bare identity `"llama2-7b"` to `AutoConfig.from_pretrained` instead attempts a Hub snapshot lookup for that name. A native identity does not imply a Hub repository or pretrained-weight import support.
