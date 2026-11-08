# M3 training settings

Import `TrainerConfig` and `OptimizerConfig` from `lilax.generation.m3_config`. Constructor values below describe component defaults, not training recommendations. The common HF-style Trainer uses [TrainingArguments](../training/training-arguments.md) instead.

## TrainerConfig

Import from `lilax.generation.m3_config`. Pydantic model.

These fields store component training preferences. Constructing `TrainerConfig` does not start or configure the public HF-style Trainer. A field described as retained has no training loop attached by this configuration object.

<details>
<summary>Parameters</summary>

- `learning_rate` (`float`, optional, defaults to `1e-5`). The base optimizer learning rate.
- `scale_lr` (`bool`, optional, defaults to `False`). Retained batch-based learning rate scaling.
- `max_train_samples` (`int`, optional, defaults to `-1`). Retained sample limit.
- `max_train_steps` (`int`, optional, defaults to `1500`). Retained optimizer-step limit.
- `num_train_epochs` (`int`, optional, defaults to `1`). Retained epoch count.
- `seed` (`int`, optional, defaults to `0`). The random seed.
- `per_device_batch_size` (`float`, optional, defaults to `1.0`). Examples per device. Processing derives global load and training sizes from device count.
- `global_batch_size` (`int`, optional, defaults to `0`). Retained explicit global batch size.
- `disable_training_weights` (`bool`, optional, defaults to `False`). Retained disabled weight updates.
- `train_text_encoder` (`bool`, optional, defaults to `False`). Retained text-encoder training.
- `text_encoder_learning_rate` (`float`, optional, defaults to `4.25e-6`). Retained text-encoder learning rate.
- `snr_gamma` (`float`, optional, defaults to `-1.0`). Retained signal-to-noise loss weighting.
- `remat_policy` (`str`, optional, defaults to `"NONE"`). Retained activation recomputation policy.
- `names_which_can_be_saved` (`list[str]`, optional, defaults to `[]`). Names of activations eligible for rematerialization retention. Each instance receives a fresh value.
- `names_which_can_be_offloaded` (`list[str]`, optional, defaults to `[]`). Names of activations eligible for host offload. Each instance receives a fresh value.
- `enable_tile_search` (`bool`, optional, defaults to `False`). Retained kernel tile search.
- `tile_search_mode` (`str`, optional, defaults to `"smart"`). Retained tile-search strategy.
- `tile_search_iters` (`int`, optional, defaults to `10`). Retained tile-search iterations.
- `tile_search_out` (`str`, optional, defaults to `""`). Retained tile-search output.
- `optimizer` (`OptimizerConfig`, optional, defaults to `OptimizerConfig()`). Optimizer settings. Each instance receives a fresh value.
- `eval_every` (`int`, optional, defaults to `-1`). Retained evaluation interval.
- `eval_data_dir` (`str`, optional, defaults to `""`). Retained evaluation input.
- `enable_generate_video_for_eval` (`bool`, optional, defaults to `False`). Retained video generation during evaluation.
- `eval_max_number_of_samples_in_bucket` (`int`, optional, defaults to `60`). Retained per-bucket evaluation sample limit.
- `enable_eval_timesteps` (`bool`, optional, defaults to `False`). Retained evaluation of selected diffusion timesteps.
- `timesteps_list` (`list[int]`, optional, defaults to `[125, 250, 375, 500, 625, 750, 875]`). Retained evaluation timesteps. Each instance receives a fresh value.
- `num_eval_samples` (`int`, optional, defaults to `420`). Retained evaluation sample count.
- `metrics_file` (`str`, optional, defaults to `""`). Retained metrics output.
- `write_metrics` (`bool`, optional, defaults to `True`). Retained metrics writing.
- `timing_metrics_file` (`str`, optional, defaults to `""`). Retained timing output.
- `write_timing_metrics` (`bool`, optional, defaults to `True`). Retained timing-metrics writing.
- `gcs_metrics` (`bool`, optional, defaults to `False`). Retained GCS metrics output.
- `log_period` (`int`, optional, defaults to `100`). Retained logging interval.
- `enable_mllog` (`bool`, optional, defaults to `False`). Retained MLPerf logging.
- `enable_ssim` (`bool`, optional, defaults to `False`). Retained structural-similarity metrics.
- `enable_ml_diagnostics` (`bool`, optional, defaults to `False`). Retained ML diagnostics.

</details>

## OptimizerConfig

Import from `lilax.generation.m3_config`. Pydantic model.

See [configure components](../../guides/components/m3-settings.md).

<details>
<summary>Parameters</summary>

- `adam_b1` (`float`, optional, defaults to `0.9`). First-moment decay.
- `adam_b2` (`float`, optional, defaults to `0.999`). Second-moment decay.
- `adam_eps` (`float`, optional, defaults to `1e-8`). Numerical epsilon.
- `adam_weight_decay` (`float`, optional, defaults to `0.0`). Weight decay.
- `opt_enable_grad_clipping` (`bool`, optional, defaults to `False`). Retained elementwise clipping.
- `max_grad_value` (`float`, optional, defaults to `1.0`). The elementwise clipping bound.
- `opt_enable_grad_global_norm_clipping` (`bool`, optional, defaults to `False`). Retained global norm clipping.
- `max_grad_norm` (`float`, optional, defaults to `1.0`). The global norm bound.
- `warmup_steps_fraction` (`float`, optional, defaults to `0.1`). Retained warmup fraction.
- `learning_rate_schedule_steps` (`int`, optional, defaults to `-1`). Retained schedule steps.
- `save_optimizer` (`bool`, optional, defaults to `False`). Retained optimizer state saving.

</details>
