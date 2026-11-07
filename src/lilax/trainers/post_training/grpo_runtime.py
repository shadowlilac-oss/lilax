"""Install GRPO role allocation before the supplied model is materialized."""

from pathlib import Path

import jax
import yaml

from lilax._backend.configs.inert import RLConfig
from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.trainers.post_train.rl.setup import device_allocation
from lilax._backend.utils.globals import MAXTEXT_CONFIGS_DIR
from lilax._processing.preferences import processing_overrides
from lilax.callbacks._configuration import callback_overrides
from lilax.runtime import Runtime
from ..lifecycle._post_base_precision import check_precision_overrides


def install_grpo_runtime(model, args, processing_class, train_dataset, eval_dataset, rollout, agentic, callbacks):
  runtime = model.runtime
  values = runtime.public_config.get_source_values()
  if not values.get("generation_configs"):
    with (Path(MAXTEXT_CONFIGS_DIR) / "post_train/rl.yml").open(encoding="utf-8") as stream:
      values["generation_configs"] = yaml.safe_load(stream)["generation_configs"]
  overrides = args._overrides() | processing_overrides(processing_class) | callback_overrides(callbacks)
  options = getattr(train_dataset, "_lilax_rl_options", None)
  if options is not None:
    overrides["train_fraction"] = options.train_fraction
  overrides["num_test_batches"] = int(eval_dataset is not None)
  if rollout is not None:
    for section in (rollout.cluster, rollout.weight_sync):
      if section is not None:
        overrides.update(section.model_dump(exclude_unset=True))
    if rollout.vllm is not None:
      overrides.update(rollout.vllm._source_values())
    if rollout.max_prompt_length is not None:
      overrides["max_prefill_predict_length"] = rollout.max_prompt_length
    if rollout.generation_config is not None:
      generation = rollout.generation_config
      overrides.update(decode_sampling_strategy=generation.sampling_options()["algorithm"],
                       decode_sampling_temperature=generation.temperature if generation.do_sample else 0.0,
                       decode_sampling_top_k=generation.top_k,
                       decode_sampling_nucleus_p=1.0 if generation.top_p < 0 else generation.top_p)
      if generation.max_new_tokens is not None:
        prompt_length = overrides.get("max_prefill_predict_length", values["max_prefill_predict_length"])
        overrides["max_target_length"] = prompt_length + generation.max_new_tokens
    overrides["reshard_chunk_size"] = rollout.reshard_chunk_size
    if rollout.engine == "maxtext":
      overrides.update(use_grpo=True, train_data_columns=["prompts"], return_log_prob=True)
  if agentic is not None:
    overrides.update(agentic._source_values())
  values.update(runtime._overrides | overrides)
  source = RLConfig.model_validate(values)
  source._explicit_keys = runtime.public_config._explicit_keys | overrides.keys()
  source.learning_rate_schedule_steps = values["learning_rate_schedule_steps"]
  profile = runtime._profile
  selected = {} if profile is None else {
      name: value for name, value in profile.source_values(rl=True).items()
      if name not in source._explicit_keys or name == "hardware"
  }
  native = resolve_runtime_config(source, overrides=selected)
  if rollout is not None and rollout.engine == "maxtext":
    count = native.inference_devices_per_replica * native.inference_replicas
    devices = jax.devices()
    if count >= len(devices):
      raise ValueError("The MaxText GRPO engine requires inference devices followed by at least one training device.")
    trainer_devices, sampler_devices = devices[count:], devices[:count]
    trainer_config = native
    sampler_config = resolve_runtime_config(source, overrides=selected | {
        "ici_data_parallelism": native.inference_replicas,
        "ici_tensor_parallelism": native.inference_devices_per_replica, "ici_fsdp_parallelism": 1,
    })
  else:
    trainer_config, sampler_config, trainer_devices, sampler_devices = device_allocation.setup_configs_and_devices_from_config(
        native, resolve_config=lambda updates: resolve_runtime_config(source, overrides=selected | updates),
        devices=runtime.devices,
    )
  if model.existing_graph_config is not None:
    check_precision_overrides(model.existing_graph_config, selected | overrides)
  if runtime.mesh is not None:
    if any(getattr(trainer_config, name) != getattr(runtime.config, name) for name in (
        "mesh_axes", "ici_parallelism", "dcn_parallelism", "data_sharding", "logical_axis_rules",
    )) or list(runtime.mesh.devices.flat) != list(trainer_devices):
      raise ValueError("GRPO role allocation cannot change the supplied model's initialized mesh.")
  else:
    placement = Runtime(source, devices=trainer_devices)
    placement.config = trainer_config
    placement.initialize()
    runtime.mesh = placement.mesh
  runtime.config, runtime.devices = trainer_config, trainer_devices
  return trainer_config, sampler_config, sampler_devices
