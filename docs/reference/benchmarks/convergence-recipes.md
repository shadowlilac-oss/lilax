# Convergence recipe settings

Import from `lilax.benchmarks.convergence`. These are configuration recipes for specialized training runs, not measured convergence guarantees.

The module exposes dataset records `c4_mlperf_hp`, `c4_en_hp`, and `c4_mutil_hp`, convergence records `llama3_405b_hp` and `deepseek_671b_hp`, and `DEEPSEEK_V3_LIBTPU_INIT_ARGS`. These retain recipe-specific cloud paths and hardware choices.

See [configure convergence records](../../guides/benchmarks/convergence-recipes.md).

## DatasetHParams

Import from `lilax.benchmarks.convergence`. Dataclass.

```python
DatasetHParams(
	name: str, dataset_path: str, dataset_name: str, dataset_type: str, train_split: str,
	eval_split: str, eval_tokens: int, add_bos: bool, add_eos: bool,
	eval_dataset_name: str | None = None, grain_file_type: str | None = None,
)
```

Describes dataset settings in a dataclass record.

<details>
<summary>Parameters</summary>

- `name` (`str`). Recipe record name.
- `dataset_path` (`str`). The dataset location.
- `dataset_name` (`str`). The training dataset.
- `dataset_type` (`str`). The data implementation.
- `train_split` (`str`). The training split.
- `eval_split` (`str`). The evaluation split.
- `eval_tokens` (`int`). The dataset evaluation budget.
- `add_bos` (`bool`). Flag for beginning-of-sequence tokens.
- `add_eos` (`bool`). Flag for end-of-sequence tokens.
- `eval_dataset_name` (`str | None`, optional, defaults to `None`). Evaluation dataset override when nonempty.
- `grain_file_type` (`str | None`, optional, defaults to `None`). The Grain input format when nonempty.

</details>

## ConvHParams

Import from `lilax.benchmarks.convergence`. Dataclass.

```python
ConvHParams(
	global_batch_size: int, warmup_samples: int, decay_end_samples: int, total_tokens_to_train: int,
	learning_rate: float, eval_interval: int, training_scaleing_factor: float, eval_tokens: int = -1,
	seeds: int = 1238,
)
```

Describes the recipe's schedule. `training_scaleing_factor` and `global_batch_size` are stored in the record.
The transformation uses its explicit batch-size argument. The spelling `training_scaleing_factor` is part of the API.

<details>
<summary>Parameters</summary>

- `global_batch_size` (`int`). The recipe batch size.
- `warmup_samples` (`int`). Warmup sample count used with the explicit transformation batch size to calculate warmup steps.
- `decay_end_samples` (`int`). Sample count used to calculate learning-rate decay steps.
- `total_tokens_to_train` (`int`). Token count used to calculate the recipe step budget.
- `learning_rate` (`float`). The rate multiplied by explicit global batch size.
- `eval_interval` (`int`). The evaluation token interval.
- `training_scaleing_factor` (`float`). The recipe scaling factor.
- `eval_tokens` (`int`, optional, defaults to `-1`). Evaluation budget override when positive.
- `seeds` (`int`, optional, defaults to `1238`). The data shuffle seed.

</details>

## setup_dataset

```python
setup_dataset(tuning_params: dict, params: DatasetHParams)
```

Mutates the dictionary with selected dataset settings.

Returns `None`.

<details>
<summary>Parameters</summary>

- `tuning_params` (`dict`). Dictionary updated with dataset settings.
- `params` (`DatasetHParams`). The dataset recipe.

</details>

## load_checkpoint

```python
load_checkpoint(tuning_params: dict, checkpoint_path: str)
```

Sets the dictionary's full-state checkpoint path.

Returns `None`.

<details>
<summary>Parameters</summary>

- `tuning_params` (`dict`). Dictionary updated with `load_full_state_path`.
- `checkpoint_path` (`str`). The full-state checkpoint path.

</details>

## setup_convergence_configs

```python
setup_convergence_configs(tuning_params: dict, params: ConvHParams, num_devices: int, global_batch_size: int)
```

Mutates batch, learning-rate, warmup, schedule, step, evaluation, and seed settings using the recipe's arithmetic.
`max_target_length` and initial `eval_steps` must be available where used.

Returns `None`.

<details>
<summary>Parameters</summary>

- `tuning_params` (`dict`). Dictionary updated with computed settings.
- `params` (`ConvHParams`). The schedule recipe.
- `num_devices` (`int`). The divisor for per-device batch size.
- `global_batch_size` (`int`). The actual batch size for schedule arithmetic.

</details>

## setup_model_convergence

```python
setup_model_convergence(
	tuning_params: dict, dataset: DatasetHParams, convergence_configs: ConvHParams,
	num_devices: int, global_batch_size: int,
) -> dict
```

Copies the dictionary and applies both transformations.

Returns the copy.

<details>
<summary>Parameters</summary>

- `tuning_params` (`dict`). Initial settings, copied before modification.
- `dataset` (`DatasetHParams`). Dataset settings.
- `convergence_configs` (`ConvHParams`). Schedule settings.
- `num_devices` (`int`). The execution device count.
- `global_batch_size` (`int`). The batch size used in arithmetic.

</details>

## c4_deepseek_v3_ep_256_v5p_512_gbs_1024

```python
c4_deepseek_v3_ep_256_v5p_512_gbs_1024()
```

Returns the built-in C4 convergence tuning dictionary.

## c4_deepseek_v3_config

```python
c4_deepseek_v3_config(**overrides)
```

Wraps the C4 tuning values in a runtime-source configuration. This object is distinct from the root HF-style model configuration.
It is not an interchangeable Trainer argument object.

Returns runtime-source configuration, like `model.runtime.public_config`. See [configuration objects](../execution/runtime.md#configuration-objects).

<details>
<summary>Parameters</summary>

- `**overrides` (`dict[str, Any]`). Overrides for values in the built-in tuning dictionary.

</details>

