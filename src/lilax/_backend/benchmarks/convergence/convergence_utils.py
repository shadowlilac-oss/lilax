"""Dataset and training settings for convergence benchmarks."""

from dataclasses import dataclass
import math


@dataclass
class DatasetHParams:
  name: str
  dataset_path: str
  dataset_name: str
  dataset_type: str
  train_split: str
  eval_split: str
  eval_tokens: int
  add_bos: bool
  add_eos: bool
  eval_dataset_name: str | None = None
  grain_file_type: str | None = None


@dataclass
class ConvHParams:
  global_batch_size: int
  warmup_samples: int
  decay_end_samples: int
  total_tokens_to_train: int
  learning_rate: float
  eval_interval: int
  training_scaleing_factor: float
  eval_tokens: int = -1
  seeds: int = 1238


def load_checkpoint(tuning_params: dict, checkpoint_path: str):
  tuning_params["load_full_state_path"] = checkpoint_path


def setup_dataset(tuning_params: dict, params: DatasetHParams):
  tuning_params["reuse_example_batch"] = 0
  tuning_params["dataset_path"] = params.dataset_path
  tuning_params["dataset_name"] = params.dataset_name
  tuning_params["dataset_type"] = params.dataset_type
  if params.grain_file_type:
    tuning_params["grain_file_type"] = params.grain_file_type
  if params.eval_dataset_name:
    tuning_params["eval_dataset_name"] = params.eval_dataset_name
  tuning_params["train_split"] = params.train_split
  tuning_params["eval_split"] = params.eval_split
  tuning_params["add_bos"] = params.add_bos
  tuning_params["add_eos"] = params.add_eos
  tuning_params["eval_steps"] = params.eval_tokens


def setup_convergence_configs(tuning_params: dict, params: ConvHParams, num_devices: int, global_batch_size: int):
  gbs = global_batch_size
  total_steps = params.total_tokens_to_train / gbs
  warmup_steps = math.ceil(params.warmup_samples / gbs - 1e-6)
  decay_end_step = math.ceil(params.decay_end_samples / gbs - 1e-6)
  max_target_length = tuning_params["max_target_length"]

  tuning_params["per_device_batch_size"] = float(gbs / num_devices)
  tuning_params["learning_rate"] = params.learning_rate * gbs
  tuning_params["warmup_steps_fraction"] = float(warmup_steps / decay_end_step)
  tuning_params["learning_rate_schedule_steps"] = decay_end_step
  tuning_params["steps"] = int(total_steps)
  eval_tokens = params.eval_tokens if params.eval_tokens > 0 else tuning_params["eval_steps"]
  tuning_params["eval_steps"] = int(math.ceil(eval_tokens / max_target_length / gbs))
  tuning_params["eval_interval"] = int(math.ceil(params.eval_interval / max_target_length / gbs))
  tuning_params["data_shuffle_seed"] = params.seeds


def setup_model_convergence(
    tuning_params: dict, dataset: DatasetHParams, convergence_configs: ConvHParams,
    num_devices: int, global_batch_size: int,
) -> dict:
  result = tuning_params.copy()
  setup_dataset(result, dataset)
  setup_convergence_configs(result, convergence_configs, num_devices, global_batch_size)
  return result
