# Diffusion schedulers

Import from `lilax.generation.wan`. See [Wan component guide](../../guides/components/wan-components.md).

## Shared construction

```python
Scheduler.from_config(config: dict, **kwargs)
```

Returns `(scheduler, state)`. `Scheduler` denotes either public scheduler class below.
Keyword overrides win and unknown constructor keys are filtered. Reassign state returned by setup and stepping.

<details>
<summary>Parameters</summary>

- `config` (`dict`). Constructor settings.
- `**kwargs` (`dict[str, Any]`). Constructor overrides.

</details>

## FlaxFlowMatchScheduler

Import from `lilax.generation.wan`. Class.

```python
FlaxFlowMatchScheduler(
	num_train_timesteps: int = 1000, shift: float = 3.0, sigma_max: float = 1.0,
	sigma_min: float = 0.003 / 1.002, inverse_timesteps: bool = False, extra_one_step: bool = False,
	reverse_sigmas: bool = False, use_dynamic_shifting: bool = False, time_shift_type: str = "linear",
	dtype: jnp.dtype = jnp.float32,
)
```

Creates a continuous-time flow scheduler. `has_state` is true and `len(scheduler)` returns the training timestep count.

<details>
<summary>Parameters</summary>

- `num_train_timesteps` (`int`, optional, defaults to `1000`). Training schedule length.
- `shift` (`float`, optional, defaults to `3.0`). Sigma shift.
- `sigma_max` (`float`, optional, defaults to `1.0`). Maximum noise.
- `sigma_min` (`float`, optional, defaults to `0.003 / 1.002`). Minimum noise.
- `inverse_timesteps` (`bool`, optional, defaults to `False`). Flag for reversed initial sigma ordering.
- `extra_one_step` (`bool`, optional, defaults to `False`). Flag to omit the final point of an extended schedule.
- `reverse_sigmas` (`bool`, optional, defaults to `False`). Flag for replacement of sigmas with one minus sigma.
- `use_dynamic_shifting` (`bool`, optional, defaults to `False`). Dynamic shifting.
- `time_shift_type` (`str`, optional, defaults to `"linear"`). Exponential or linear dynamic shifting.
- `dtype` (`jnp.dtype`, optional, defaults to `jnp.float32`). Computation dtype.

</details>

### create_state

```python
flow_scheduler.create_state() -> FlowMatchSchedulerState
```

Returns the initial state from `FlowMatchSchedulerState.create()`.

### set_timesteps

```python
flow_scheduler.set_timesteps(
	state: FlowMatchSchedulerState, num_inference_steps: int = 100, shape: Tuple = None,
	denoising_strength: float = 1.0, training: bool = False, shift: Optional[float] = None,
) -> FlowMatchSchedulerState
```

Returns updated schedule state.

<details>
<summary>Parameters</summary>

- `state` (`FlowMatchSchedulerState`). The prior state.
- `num_inference_steps` (`int`, optional, defaults to `100`). Schedule points.
- `shape` (`Tuple`, optional, defaults to `None`). Unused shape argument.
- `denoising_strength` (`float`, optional, defaults to `1.0`). Starting noise.
- `training` (`bool`, optional, defaults to `False`). Flag for training weight calculation.
- `shift` (`Optional[float]`, optional, defaults to `None`). Override for the configured shift.

</details>

### step

```python
flow_scheduler.step(
	state: FlowMatchSchedulerState, model_output: jnp.ndarray, timestep: jnp.ndarray,
	sample: jnp.ndarray, to_final: bool = False, return_dict: bool = True, step_index: Optional[int] = None,
) -> Union[FlaxFlowMatchSchedulerOutput, Tuple]
```

Returns `FlaxFlowMatchSchedulerOutput(prev_sample, state)`, or `(prev_sample, state)` when `return_dict=False`.
Requires initialized schedule state.

<details>
<summary>Parameters</summary>

- `state` (`FlowMatchSchedulerState`). The initialized schedule.
- `model_output` (`jnp.ndarray`). The predicted flow.
- `timestep` (`jnp.ndarray`). The schedule position.
- `sample` (`jnp.ndarray`). Current latents.
- `to_final` (`bool`, optional, defaults to `False`). Flag for forced final sigma selection.
- `return_dict` (`bool`, optional, defaults to `True`). Flag for an output record instead of a tuple.
- `step_index` (`Optional[int]`, optional, defaults to `None`). Step index override for timestep lookup.

</details>

### sample_timesteps

```python
flow_scheduler.sample_timesteps(timestep_rng, batch_size)
```

Returns sampled training timesteps.

<details>
<summary>Parameters</summary>

- `timestep_rng` (`jax.Array`). The JAX random key.
- `batch_size` (`int`). Sample count.

</details>

### apply_flow_match

```python
flow_scheduler.apply_flow_match(
	noise: jnp.ndarray, batch_images: jnp.ndarray, timesteps: jnp.ndarray,
)
```

Returns a tuple of noisy latents, training targets, and training weights.

<details>
<summary>Parameters</summary>

- `noise` (`jnp.ndarray`). Random noise.
- `batch_images` (`jnp.ndarray`). Clean image latents.
- `timesteps` (`jnp.ndarray`). Per-example training timesteps.

</details>

### add_noise

```python
flow_scheduler.add_noise(
	state: FlowMatchSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray,
) -> jnp.ndarray
```

Returns noisy samples. An uninitialized sigma or timestep schedule raises `ValueError`.

<details>
<summary>Parameters</summary>

- `state` (`FlowMatchSchedulerState`). Initialized sigmas and timesteps.
- `original_samples` (`jnp.ndarray`). Clean samples.
- `noise` (`jnp.ndarray`). Noise.
- `timesteps` (`jnp.ndarray`). Noise schedule positions.

</details>

### training_target

```python
flow_scheduler.training_target(sample: jnp.ndarray, noise: jnp.ndarray, *args, **kwargs) -> jnp.ndarray
```

Returns `noise - sample`.

<details>
<summary>Parameters</summary>

- `sample` (`jnp.ndarray`). Clean samples.
- `noise` (`jnp.ndarray`). Noise.
- `*args` (`tuple[Any, ...]`). Unused argument.
- `**kwargs` (`dict[str, Any]`). Unused argument.

</details>

### training_weight

```python
flow_scheduler.training_weight(state: FlowMatchSchedulerState, timestep: jnp.ndarray) -> jnp.ndarray
```

Returns training weights read from state prepared with `training=True`.

<details>
<summary>Parameters</summary>

- `state` (`FlowMatchSchedulerState`). Calculated training weights.
- `timestep` (`jnp.ndarray`). Schedule positions.

</details>

### return_to_timestep

```python
flow_scheduler.return_to_timestep(
	state: FlowMatchSchedulerState, timestep: jnp.ndarray, sample: jnp.ndarray, sample_stablized: jnp.ndarray,
) -> jnp.ndarray
```

Returns `(sample - sample_stablized) / sigma` at the matching schedule position. The spelling `sample_stablized` is part of the API.

<details>
<summary>Parameters</summary>

- `state` (`FlowMatchSchedulerState`). Initialized schedule arrays.
- `timestep` (`jnp.ndarray`). Timestep for sigma lookup.
- `sample` (`jnp.ndarray`). The current sample.
- `sample_stablized` (`jnp.ndarray`). The stabilized sample.

</details>

### FlowMatchSchedulerState

Import from `lilax.generation.wan`. Flax struct.

`FlowMatchSchedulerState.create()` returns initial state with all three arrays `None`, `training=False`, and zero steps.
Both scheduler states support [Flax struct replacement](https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html). Keep returned state.

<details>
<summary>Parameters</summary>

- `sigmas` (`jnp.ndarray`). Sigma values.
- `timesteps` (`jnp.ndarray`). Schedule positions.
- `linear_timesteps_weights` (`Optional[jnp.ndarray]`). Training weights.
- `training` (`bool`). Training mode.
- `num_inference_steps` (`int`). Schedule length.

</details>

### FlowMatchSchedulerState.create

```python
FlowMatchSchedulerState.create()
```

Returns initial state with all three arrays `None`, training disabled, and zero inference steps.

## FlaxUniPCMultistepScheduler

Import from `lilax.generation.wan`. Class.

```python
FlaxUniPCMultistepScheduler(
	num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02,
	beta_schedule: str = "linear", trained_betas: Optional[Union[jnp.ndarray, List[float]]] = None,
	solver_order: int = 2, prediction_type: str = "epsilon", thresholding: bool = False,
	dynamic_thresholding_ratio: float = 0.995, sample_max_value: float = 1.0, predict_x0: bool = True,
	solver_type: str = "bh2", lower_order_final: bool = True, disable_corrector: List[int] = [],
	solver_p: Optional[FlaxSchedulerMixin] = None, use_flow_sigmas: Optional[bool] = False,
	flow_shift: Optional[float] = 1.0, timestep_spacing: str = "linspace", steps_offset: int = 0,
	final_sigmas_type: Optional[str] = "zero", rescale_zero_terminal_snr: bool = False,
	dtype: jnp.dtype = jnp.float32,
)
```

Creates a UniPC multistep solver. Solver type accepts `"bh1"` or `"bh2"`. Other types raise `NotImplementedError`.
`has_state` is true and `len(scheduler)` returns training timestep count.

<details>
<summary>Parameters</summary>

- `num_train_timesteps` (`int`, optional, defaults to `1000`). Training schedule length.
- `beta_start` (`float`, optional, defaults to `0.0001`). Initial beta.
- `beta_end` (`float`, optional, defaults to `0.02`). Final beta.
- `beta_schedule` (`str`, optional, defaults to `"linear"`). Beta schedule, `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
- `trained_betas` (`Optional[Union[jnp.ndarray, List[float]]]`, optional, defaults to `None`). Beta values used instead of generated betas.
- `solver_order` (`int`, optional, defaults to `2`). Solver history order.
- `prediction_type` (`str`, optional, defaults to `"epsilon"`). Prediction conversion.
- `thresholding` (`bool`, optional, defaults to `False`). Dynamic thresholding flag. `True` raises `NotImplementedError` during output conversion.
- `dynamic_thresholding_ratio` (`float`, optional, defaults to `0.995`). The threshold ratio.
- `sample_max_value` (`float`, optional, defaults to `1.0`). The threshold maximum.
- `predict_x0` (`bool`, optional, defaults to `True`). Clean-sample prediction.
- `solver_type` (`str`, optional, defaults to `"bh2"`). Solver type, `bh1` or `bh2`.
- `lower_order_final` (`bool`, optional, defaults to `True`). Flag for lower solver order near the end.
- `disable_corrector` (`List[int]`, optional, defaults to `[]`). Steps without correction.
- `solver_p` (`Optional[FlaxSchedulerMixin]`, optional, defaults to `None`). Nested predictor. A non-`None` value raises `NotImplementedError` during predictor updates.
- `use_flow_sigmas` (`Optional[bool]`, optional, defaults to `False`). A flow schedule.
- `flow_shift` (`Optional[float]`, optional, defaults to `1.0`). Flow sigma shift.
- `timestep_spacing` (`str`, optional, defaults to `"linspace"`). Timestep spacing, `linspace`, `leading`, or `trailing`.
- `steps_offset` (`int`, optional, defaults to `0`). Leading timestep offset.
- `final_sigmas_type` (`Optional[str]`, optional, defaults to `"zero"`). Final sigma type, `zero` or `sigma_min`.
- `rescale_zero_terminal_snr` (`bool`, optional, defaults to `False`). Flag for terminal signal-to-noise rescaling.
- `dtype` (`jnp.dtype`, optional, defaults to `jnp.float32`). Computation dtype.

</details>

### set_timesteps

```python
unipc_scheduler.set_timesteps(
	state: UniPCMultistepSchedulerState, num_inference_steps: int, shape: Tuple,
) -> UniPCMultistepSchedulerState
```

Returns schedule state with initialized solver history. Invalid spacing or final sigma type raises `ValueError`.

<details>
<summary>Parameters</summary>

- `state` (`UniPCMultistepSchedulerState`). Scheduler-created state.
- `num_inference_steps` (`int`). Schedule length.
- `shape` (`Tuple`). Sample shape for solver history allocation.

</details>

### step

```python
unipc_scheduler.step(
	state: UniPCMultistepSchedulerState, model_output: jnp.ndarray, timestep: Union[int, jnp.ndarray],
	sample: jnp.ndarray, return_dict: bool = True, generator: Optional[jax.random.PRNGKey] = None,
) -> Union[FlaxUniPCMultistepSchedulerOutput, Tuple[jnp.ndarray]]
```

Requires timestep setup first or raises `ValueError`.

Returns `FlaxUniPCMultistepSchedulerOutput` with `prev_sample` and updated `state`.
False `return_dict` returns `(prev_sample, state)`. Output mapping conveniences follow [Diffusers BaseOutput](https://huggingface.co/docs/diffusers/api/outputs).

<details>
<summary>Parameters</summary>

- `state` (`UniPCMultistepSchedulerState`). Initialized solver history.
- `model_output` (`jnp.ndarray`). The current prediction.
- `timestep` (`Union[int, jnp.ndarray]`). The schedule position.
- `sample` (`jnp.ndarray`). Current latents.
- `return_dict` (`bool`, optional, defaults to `True`). Flag for an output record instead of a tuple.
- `generator` (`Optional[jax.random.PRNGKey]`, optional, defaults to `None`). Unused argument.

</details>

### scale_model_input

```python
unipc_scheduler.scale_model_input(state: UniPCMultistepSchedulerState, sample: jnp.ndarray, *args, **kwargs) -> jnp.ndarray
```

Returns the sample unchanged.

<details>
<summary>Parameters</summary>

- `state` (`UniPCMultistepSchedulerState`). Unused argument.
- `sample` (`jnp.ndarray`). Model input.
- `*args` (`tuple[Any, ...]`). Unused argument.
- `**kwargs` (`dict[str, Any]`). Unused argument.

</details>

### add_noise

```python
unipc_scheduler.add_noise(
	state: UniPCMultistepSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray,
	timesteps: jnp.ndarray,
) -> jnp.ndarray
```

Returns noisy samples using the common schedule.

<details>
<summary>Parameters</summary>

- `state` (`UniPCMultistepSchedulerState`). The common schedule.
- `original_samples` (`jnp.ndarray`). Clean samples.
- `noise` (`jnp.ndarray`). Added noise.
- `timesteps` (`jnp.ndarray`). Schedule positions.

</details>

## UniPC state and solver operations

### create_state

```python
unipc_scheduler.create_state(common: Optional[CommonSchedulerState] = None) -> UniPCMultistepSchedulerState
```

Returns state with a common schedule. Use scheduler-produced state rather than constructing its common state manually.

<details>
<summary>Parameters</summary>

- `common` (`Optional[CommonSchedulerState]`, optional, defaults to `None`). A common schedule. Omission builds it.

</details>

### set_begin_index

```python
unipc_scheduler.set_begin_index(state: UniPCMultistepSchedulerState, begin_index: int = 0) -> UniPCMultistepSchedulerState
```

Returns state with the requested starting index.

<details>
<summary>Parameters</summary>

- `state` (`UniPCMultistepSchedulerState`). Prior state.
- `begin_index` (`int`, optional, defaults to `0`). The starting schedule index.

</details>

### convert_model_output

```python
unipc_scheduler.convert_model_output(state: UniPCMultistepSchedulerState, model_output: jnp.ndarray, sample: jnp.ndarray) -> jnp.ndarray
```

Returns the prediction converted to solver representation. Dynamic thresholding raises `NotImplementedError`.
Prediction types are `epsilon`, `sample`, `v_prediction`, and, with `predict_x0=True`, `flow_prediction`.

<details>
<summary>Parameters</summary>

- `state` (`UniPCMultistepSchedulerState`). The current initialized step index and sigmas.
- `model_output` (`jnp.ndarray`). The model prediction.
- `sample` (`jnp.ndarray`). Current latents.

</details>

### index_for_timestep

```python
unipc_scheduler.index_for_timestep(
	state: UniPCMultistepSchedulerState, timestep: Union[int, jnp.ndarray],
	schedule_timesteps: Optional[jnp.ndarray] = None,
) -> int
```

Returns the first matching index or the final index when no match exists.

<details>
<summary>Parameters</summary>

- `state` (`UniPCMultistepSchedulerState`). Schedule timesteps.
- `timestep` (`Union[int, jnp.ndarray]`). The position to find.
- `schedule_timesteps` (`Optional[jnp.ndarray]`, optional, defaults to `None`). Schedule timesteps used instead of `state.timesteps` for lookup.

</details>

### multistep_uni_p_bh_update

```python
unipc_scheduler.multistep_uni_p_bh_update(
	state: UniPCMultistepSchedulerState, model_output: jnp.ndarray, sample: jnp.ndarray, order: int,
) -> jnp.ndarray
```

Returns the predictor sample using initialized solver history. A configured nested `solver_p` raises `NotImplementedError`.

<details>
<summary>Parameters</summary>

- `state` (`UniPCMultistepSchedulerState`). Initialized output and timestep history.
- `model_output` (`jnp.ndarray`). Unused prediction argument. Prediction uses the stored history.
- `sample` (`jnp.ndarray`). Current latents.
- `order` (`int`). The predictor order.

</details>

### multistep_uni_c_bh_update

```python
unipc_scheduler.multistep_uni_c_bh_update(
	state: UniPCMultistepSchedulerState, this_model_output: jnp.ndarray, last_sample: jnp.ndarray,
	this_sample: jnp.ndarray, order: int,
) -> jnp.ndarray
```

Returns the corrector sample from prior output history. Normal callers use `step`, which updates history and step index together.

<details>
<summary>Parameters</summary>

- `state` (`UniPCMultistepSchedulerState`). Initialized solver history.
- `this_model_output` (`jnp.ndarray`). The current prediction.
- `last_sample` (`jnp.ndarray`). The preceding sample.
- `this_sample` (`jnp.ndarray`). The current sample.
- `order` (`int`). Corrector order.

</details>

### UniPCMultistepSchedulerState

Import from `lilax.generation.wan`. Flax struct.

Tracks schedules, output history, order, samples, and indexes.

<details>
<summary>Parameters</summary>

- `common` (`CommonSchedulerState`). Common schedule arrays.
- `sigmas` (`jnp.ndarray`). Noise sigmas.
- `alpha_t` (`jnp.ndarray`). Signal coefficients.
- `sigma_t` (`jnp.ndarray`). Noise coefficients.
- `lambda_t` (`jnp.ndarray`). Log coefficient ratios.
- `init_noise_sigma` (`float`). Initial noise scale.
- `timesteps` (`jnp.ndarray`, optional, defaults to `None`). Schedule positions.
- `model_outputs` (`jnp.ndarray`, optional, defaults to `None`). Prediction history.
- `timestep_list` (`jnp.ndarray`, optional, defaults to `None`). Timestep history.
- `lower_order_nums` (`int`, optional, defaults to `0`). Available solver order.
- `last_sample` (`Optional[jnp.ndarray]`, optional, defaults to `None`). Prior latents.
- `step_index` (`Optional[int]`, optional, defaults to `None`). The current step.
- `begin_index` (`Optional[int]`, optional, defaults to `None`). The initial step.
- `this_order` (`int`, optional, defaults to `0`). Current solver order.

</details>

### UniPCMultistepSchedulerState.create

```python
UniPCMultistepSchedulerState.create(
	common_state: CommonSchedulerState, alpha_t: jnp.ndarray, sigma_t: jnp.ndarray,
	lambda_t: jnp.ndarray, sigmas: jnp.ndarray, init_noise_sigma: jnp.ndarray,
)
```

Returns state with initial history and indexes.

<details>
<summary>Parameters</summary>

- `common_state` (`CommonSchedulerState`). Common schedule arrays.
- `alpha_t` (`jnp.ndarray`). Signal coefficients.
- `sigma_t` (`jnp.ndarray`). Noise coefficients.
- `lambda_t` (`jnp.ndarray`). Log coefficient ratios.
- `sigmas` (`jnp.ndarray`). Schedule sigmas.
- `init_noise_sigma` (`jnp.ndarray`). Initial noise scale.

</details>
