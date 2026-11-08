# Optimization and objectives

Import these configurations from `lilax.training`. See [optimization guide](../../guides/training/optimization.md).

## LRSchedulerConfig

Import from `lilax.training`. Pydantic model with keyword fields.

`TrainingArguments.lr_scheduler_type` selects `"cosine"` or `"wsd"`. `learning_rate` is the peak rate. `warmup_ratio` sets the fraction of the schedule spent increasing linearly from zero to that rate.

Warmup lasts `int(schedule_steps * warmup_ratio)` steps. Cosine decay uses the remaining schedule steps. For WSD, decay lasts `int(schedule_steps * wsd_decay_steps_fraction)` steps. The warmup and WSD decay fractions must sum to at most one.

Distillation's default optimizer uses warmup and cosine decay over the training duration, with `warmup_ratio` and `final_fraction`. It does not use the WSD fields or a separate `schedule_steps` duration. A supplied `optimizers` schedule replaces the default schedule when Lilax builds the optimizer. With a supplied transformation, Lilax does not attach the schedule to it. See the [optimizer pair contract](trainers.md#optimizers).

The [Hugging Face optimizer and scheduler reference](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules) describes upstream schedules. Lilax accepts the fields and scheduler choices listed here.

<details>
<summary>Parameters</summary>

- `schedule_steps` (`int`, optional, defaults to `-1`). Schedule duration. `-1` uses training duration. Steps after schedule completion use zero learning rate.
- `final_fraction` (`float`, optional, defaults to `0.1`). Final decay rate divided by peak learning rate.
- `wsd_decay_steps_fraction` (`float`, optional, defaults to `0.1`). WSD decay fraction. The stable phase uses steps left after warmup and decay.
- `wsd_decay_style` (`Literal["linear", "cosine"]`, optional, defaults to `"linear"`). WSD decay curve.

</details>

## OptimizerConfig

Import from `lilax.training`. Pydantic model with keyword fields.

Both pattern lists use Python regular expressions with `re.search`, not glob patterns or Hugging Face parameter names. Matching uses slash-separated paths in the model parameter tree, such as `token_embedder/embedding`. A match to any expression selects the path. Anchors `^` and `$` restrict a pattern to the whole path.

The lists select opposite sets. An `adamw_mask` match excludes weight decay. A `trainable_parameters` match includes parameter updates. See [parameter selection example](../../guides/training/optimization.md#select-parameters).

<details>
<summary>Parameters</summary>

- `adam_eps_root` (`float`, optional, defaults to `0.0`). Adam stability term inside the square root. `TrainingArguments.adam_epsilon` is outside it.
- `adamw_mask` (`list[str] | None`, optional, defaults to `None`). Paths excluded from AdamW or Adam-pax weight decay. `None` or `[]` adds no exclusions. Does not freeze parameters.
- `mu_dtype` (`str`, optional, defaults to `""`). First-moment dtype. Empty uses model weight dtype.
- `muon` (`MuonConfig | None`, optional, defaults to `None`). Muon settings.
- `spike_skip` (`SpikeSkipConfig | None`, optional, defaults to `None`). Loss or gradient spike update-skipping settings.
- `trainable_parameters` (`list[str] | None`, optional, defaults to `None`). Paths allowed to update. Others are frozen. `None` or `[]` leaves all trainable.

</details>

### MuonConfig

Import from `lilax.training`. Pydantic model with keyword fields.

Configures Muon optimization.

<details>
<summary>Parameters</summary>

- `implementation` (`Literal["optax_muon", "maxtext_muon"]`, optional, defaults to `"optax_muon"`). Muon implementation.
- `beta` (`float`, optional, defaults to `0.95`). Momentum coefficient.
- `weight_decay` (`float`, optional, defaults to `0.0`). Muon weight decay.
- `consistent_rms` (`float | None`, optional, defaults to `None`). Consistent RMS setting.
- `include_routers` (`bool`, optional, defaults to `True`). Router parameter inclusion.
- `use_all_to_all` (`bool`, optional, defaults to `False`). All-to-all communication.

</details>

### SpikeSkipConfig

Import from `lilax.training`. Pydantic model with keyword fields.

Configures optimizer update skipping on loss or gradient spikes.

<details>
<summary>Parameters</summary>

- `enabled` (`bool`, optional, defaults to `False`). Spike update skipping.
- `interval` (`int`, optional, defaults to `128`). Spike-history interval.
- `scaling_factor` (`float`, optional, defaults to `6.0`). Spike threshold scaling.

</details>

## RematConfig

Import from `lilax.training`. Pydantic model with keyword fields.

Configures rematerialization and host offload.

<details>
<summary>Parameters</summary>

- `remat_policy` (`str`, optional, defaults to `"full"`). Decoder rematerialization policy.
- `remat_policy_for_vit` (`str`, optional, defaults to `"minimal"`). Vision rematerialization policy.
- `optimizer_memory_host_offload` (`bool`, optional, defaults to `False`). Optimizer memory offload to the host.
- `parameter_memory_host_offload` (`bool`, optional, defaults to `False`). Parameter memory offload to the host.

</details>

## ObjectiveConfig

Import from `lilax.training`. Pydantic model with keyword fields.

Selects the training objective. Block diffusion requires a valid mask ID from the model vocabulary.
See [block-diffusion compatibility](block-diffusion.md#trainer-compatibility).

<details>
<summary>Parameters</summary>

- `training_objective` (`Literal["causal_lm", "block_diffusion"]`, optional, defaults to `"causal_lm"`). Objective.
- `block_diffusion_mask_id` (`int`, optional, defaults to `-1`). Mask token ID.
- `block_diffusion_min_noise` (`float`, optional, defaults to `1e-3`). Minimum corruption noise.
- `block_diffusion_logit_alignment` (`Literal["same_position", "shifted"]`, optional, defaults to `"same_position"`). Logit alignment.
- `block_diffusion_canvas_policy` (`Literal["all_masked", "seed_and_mask"]`, optional, defaults to `"all_masked"`). Corruption canvas policy.

</details>

## BatchRampupConfig

Import from `lilax.training`. Pydantic model with keyword fields.

Configures batch-size rampup.

All configurations on this page reject extra fields. They are composed through [training arguments](training-arguments.md).

<details>
<summary>Parameters</summary>

- `enable_rampup_batch_size` (`bool`, optional, defaults to `False`). Batch-size rampup.
- `per_device_batch_size_start` (`float`, optional, defaults to `4.0`). Initial batch per device.
- `per_device_batch_size_increment` (`float`, optional, defaults to `2.0`). Batch increment per device.
- `global_rampup_samples` (`int`, optional, defaults to `500`). Samples governing rampup.

</details>
