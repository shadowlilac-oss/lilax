# Common training arguments

## TrainingArguments

Import from `lilax`. This Pydantic model configures training duration, optimization, reporting, evaluation, and saving through keyword fields. It requires `output_dir` and forbids unknown fields. Task configurations extend these fields unless their reference explicitly replaces them. Focused configuration objects supply scheduling, optimization, checkpoints, parallelism, DiLoCo, rematerialization, objectives, and batch rampup. See the [optimization guide](../../guides/training/optimization.md) for composition examples. Positive evaluation cadence requires evaluation data.

Training checkpoints use `<output_dir>/<run_name>/checkpoints/<step>/`. Resume with `True` uses this run's checkpoint root.
Set a stable `run_name` to resume across processes.

Unsupported `report_to` names raise `ValueError`. The stored result is a list of selected trackers.
TensorBoard appends the resolved run name to its base directory. Default event files therefore use
`<output_dir>/<run_name>/tensorboard/<run_name>/`, or `<logging_dir>/<run_name>/` with an override.
See [reporting callbacks](callbacks.md) for metrics files and Vertex upload behavior.

<details>
<summary>Parameters</summary>

<a id="duration-and-optimization"></a>

**Duration and optimization fields.**

- `output_dir` (`str`). Output root.
- `run_name` (`str`, optional, defaults to `""`). Run label. Empty uses `JOBSET_NAME` when set, otherwise `<model_name>_<YYYY-MM-DD-HH-MM>`.
- `max_steps` (`int`, optional, defaults to `150001`). Training step limit.
- `num_train_epochs` (`int`, optional, defaults to `1`). Dataset epochs.
- `per_device_train_batch_size` (`int | float`, optional, defaults to `12`). Training batch per device.
- `per_device_eval_batch_size` (`int | float`, optional, defaults to `0.0`). Evaluation batch per device. Zero uses the training value.
- `gradient_accumulation_steps` (`int`, optional, defaults to `1`). Microbatch accumulation.
- `learning_rate` (`float`, optional, defaults to `3e-5`). Peak learning rate.
- `optim` (`Literal["adamw", "adam_pax", "sgd", "muon"]`, optional, defaults to `"adamw"`). Optimizer.
- `adam_beta1` (`float`, optional, defaults to `0.9`). First Adam moment coefficient.
- `adam_beta2` (`float`, optional, defaults to `0.95`). Second Adam moment coefficient.
- `adam_epsilon` (`float`, optional, defaults to `1e-8`). Adam epsilon outside the square root.
- `weight_decay` (`float`, optional, defaults to `0.1`). Weight decay.
- `max_grad_norm` (`float`, optional, defaults to `1.0`). Gradient clipping threshold.
- `lr_scheduler_type` (`Literal["cosine", "wsd"]`, optional, defaults to `"cosine"`). Schedule type.
- `warmup_ratio` (`float`, optional, defaults to `0.1`). Warmup fraction.

<a id="logging-evaluation-and-saving"></a>

**Logging, evaluation, and saving fields.**

- `logging_steps` (`int`, optional, defaults to `100`). Logging cadence.
- `eval_steps` (`int`, optional, defaults to `-1`). Evaluation cadence. Positive values require evaluation data.
- `save_steps` (`int`, optional, defaults to `10000`). Checkpoint cadence.
- `eval_on_start` (`bool`, optional, defaults to `False`). Evaluation before training.
- `report_to` (`str | list[str]`, optional, defaults to `"none"`). `none`, `all`, `tensorboard`, `wandb`, `trackio`, or a list of tracker names.
- `logging_dir` (`str | None`, optional, defaults to `None`). TensorBoard base directory. `None` selects `<output_dir>/<run_name>/tensorboard/`.

<a id="focused-configurations"></a>

**Focused configuration fields.**

- `lr_scheduler_config` (`LRSchedulerConfig | None`, optional, defaults to `None`). [Schedule settings](optimization.md).
- `optimizer_config` (`OptimizerConfig | None`, optional, defaults to `None`). [Optimizer settings](optimization.md).
- `checkpoint_config` (`TrainingCheckpointConfig | None`, optional, defaults to `None`). [Checkpoint settings](checkpoints.md).
- `parallelism_config` (`ParallelismConfig | None`, optional, defaults to `None`). [Parallelism settings](../execution/runtime.md).
- `diloco_config` (`DiLoCoConfig | None`, optional, defaults to `None`). [DiLoCo settings](../execution/runtime.md).
- `remat_config` (`RematConfig | None`, optional, defaults to `None`). [Rematerialization settings](optimization.md).
- `objective_config` (`ObjectiveConfig | None`, optional, defaults to `None`). [Objective settings](optimization.md).
- `batch_rampup_config` (`BatchRampupConfig | None`, optional, defaults to `None`). [Batch rampup settings](optimization.md).
- `gradient_checkpointing` (`bool`, optional, defaults to `True`). Rematerialization.
- `jax_distributed_initialization_timeout` (`int`, optional, defaults to `300`). Distributed initialization timeout.

</details>
